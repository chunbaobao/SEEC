import torch
import torch.nn as nn
from torch.nn import functional as F
from compressai.layers import GDN, conv3x3, subpel_conv3x3, ResidualBlock, conv1x1
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from .custom_layers import SWin_Attention, downsample_conv1x1, subpel_conv1x1, ResBlock_1x1, maskedconv_3p
from .lmm import LogisticMixtureModel as Lmm


class AnalysisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=2)
        self.gdn = GDN(out_ch)
        self.skip = conv3x3(in_ch, out_ch, stride=2)
        self.rb = ResidualBlock(out_ch, out_ch)

    def forward(self, input):
        out = self.conv1(input)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out = out + self.skip(input)

        out = self.rb(out)
        return out


class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb = ResidualBlock(in_ch, out_ch)
        self.conv_up = subpel_conv3x3(out_ch, out_ch, r=2)
        self.igdn = GDN(out_ch, inverse=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.upsample = subpel_conv3x3(out_ch, out_ch, r=2)

    def forward(self, input):
        out1 = self.rb(input)

        out = self.conv_up(out1)
        out = self.igdn(out)
        out = self.conv(out)
        out = self.leaky_relu(out)

        out = out + self.upsample(out1)

        return out


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            AnalysisBlock(in_ch, out_ch),
            AnalysisBlock(out_ch, out_ch),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=8),
            AnalysisBlock(out_ch, out_ch),
            conv3x3(out_ch, out_ch, stride=2),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=4),
        )

    def forward(self, input):
        out = self.layers(input)
        return out


class Decoder(nn.Module):
    def __init__(self, in_ch, prior_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            SWin_Attention(dim=in_ch, num_heads=8, window_size=4),
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch),
            SWin_Attention(dim=in_ch, num_heads=8, window_size=8),
            SynthesisBlock(in_ch, in_ch),
        )
        self.conv_prior = subpel_conv3x3(in_ch, prior_ch, r=2)

    def forward(self, input):
        out = self.layers(input)
        prior = self.conv_prior(out)
        return prior


class HyperEncoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

    def forward(self, input):
        out = self.layers(input)
        return out


class HyperDecoder(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers_mu = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

        self.layers_sigma = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

    def forward(self, input):
        mu = self.layers_mu(input)
        sigma = self.layers_sigma(input)
        return mu, sigma


class PatchedEntropyBottleneck(EntropyBottleneck):
    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        B, C, H, W = x.shape
        return super(EntropyBottleneck, self).compress(
            x.reshape(1, B * C, H, W), indexes.reshape(1, B * C, H, W), medians.reshape(1, B * C, 1, 1)
        )

    def decompress(self, strings, size, BC):
        output_size = (BC[0], self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(BC[0], *([-1] * (len(size) + 1)))

        z_hat = super(EntropyBottleneck, self).decompress(
            strings, indexes.reshape(1, BC[0] * BC[1], *size), torch.float, medians.reshape(1, BC[0] * BC[1], 1, 1)
        )
        return z_hat.reshape(BC[0], BC[1], size[0], size[1])


class SegImageCompressor(CompressionModel):
    def __init__(self, num_ch, prior_ch):
        super().__init__()
        self.entropy_bottleneck = PatchedEntropyBottleneck(num_ch)

        self.encoder = Encoder(3, num_ch)
        self.decoder = Decoder(num_ch, prior_ch)
        self.hyperencoder = HyperEncoder(num_ch)
        self.hyperdecoder = HyperDecoder(num_ch)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.encoder(x)  # yscale : 1
        z = self.hyperencoder(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma_hat, means=mu_hat)
        prior = self.decoder(y_hat)

        return {"prior": prior, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x):
        y = self.encoder(x)
        z = self.hyperencoder(y)

        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], z.size()[:2])

        mu_hat, sigma_hat = self.hyperdecoder(z_hat)

        B, C, H, W = mu_hat.shape
        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, B * C, H, W))
        y_strings = self.gaussian_conditional.compress(
            y.reshape(1, B * C, H, W), indexes, means=mu_hat.reshape(1, B * C, H, W)
        )

        return {"strings": [y_strings, z_strings], "shape": z.size()}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[2:], shape[:2])
        mu_hat, sigma_hat = self.hyperdecoder(z_hat)

        B, C, H, W = mu_hat.shape
        indexes = self.gaussian_conditional.build_indexes(sigma_hat.reshape(1, B * C, H, W))
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=mu_hat.reshape(1, B * C, H, W))
        y_hat = y_hat.reshape(B, C, H, W)
        return {"prior": self.decoder(y_hat)}


class EntropyModel(nn.Module):
    def __init__(self, ep_ch, num_cls, num_mix=5, no_multichannel_lmm=False):

        super().__init__()
        self.num_cls = num_cls
        self.num_mix = num_mix
        self.out_channels = 10 * num_mix if no_multichannel_lmm else 12 * num_mix
        # The output conv layers for each class
        self.conv_outs = nn.ModuleList(
            [
                nn.Sequential(
                    conv1x1(ep_ch, ep_ch),
                    ResBlock_1x1(ep_ch),
                    conv1x1(ep_ch, ep_ch),
                    nn.LeakyReLU(inplace=True),
                    conv1x1(ep_ch, self.out_channels),
                )
                for _ in range(num_cls)
            ]
        )

    def forward(self, fusion_context, seg):  # for inference
        seg = seg.squeeze(1)
        seg_one_hot = F.one_hot(seg.to(torch.int64), num_classes=self.num_cls)  # only LongTensor
        seg_one_hot = seg_one_hot.permute(0, 3, 1, 2).float()  # Shape: B*num_cls*H*W
        conv_outs = torch.stack([conv(fusion_context) for conv in self.conv_outs], dim=1)  # Shape: B*num_cls*C_out*H*W

        out = (conv_outs * seg_one_hot.unsqueeze(2)).sum(dim=1)  # Shape: B*C_out*H*W

        return out

    # less cuda memory usage but slower than the above implementation
    # for training
    # def forward(self, fusion_context, seg):
    #     seg = seg.squeeze(1)
    #     B, _, H, W = fusion_context.shape
    #     out = torch.zeros(B, self.out_channels, H, W, device=fusion_context.device)
    #     for cls_idx in range(self.num_cls):
    #         mask = seg == cls_idx
    #         if mask.any():
    #             conv_out = self.conv_outs[cls_idx](fusion_context)
    #             mask = mask.unsqueeze(1).expand_as(conv_out)
    #             out[mask] = conv_out[mask]

    #     return out


class SeecNet(nn.Module):
    def __init__(
        self,
        num_ch,
        num_cls=2,
        prior_ch=256,
        context_ch=256,
        ep_ch=256,
        num_mix=5,
        noseg=False,
        no_multichannel_lmm=False,
    ):
        super().__init__()

        self.register_buffer("num_ch", torch.tensor(num_ch))
        self.register_buffer("num_mix", torch.tensor(num_mix))
        self.register_buffer("prior_ch", torch.tensor(prior_ch))
        self.register_buffer("context_ch", torch.tensor(context_ch))
        self.register_buffer("ep_ch", torch.tensor(ep_ch))
        self.register_buffer("num_cls", torch.tensor(num_cls))
        self.register_buffer("no_seg", torch.tensor(noseg))
        self.register_buffer("no_multichannel_lmm", torch.tensor(no_multichannel_lmm))

        self.seg_img_compressor = SegImageCompressor(num_ch, prior_ch)
        self.fusion = conv1x1(prior_ch + context_ch, ep_ch)
        self.mask_conv = maskedconv_3p(3, context_ch, kernel_size=7, padding=3)

        self.ep = (
            EntropyModel(ep_ch, num_cls, num_mix, no_multichannel_lmm)
            if not noseg
            else EntropyModel_noseg(ep_ch, 1, num_mix, no_multichannel_lmm)
        )

    def forward(self, x, seg):

        seg_out = self.seg_img_compressor(x)
        x = x * 2
        context = self.mask_conv(x)
        fusion_context = self.fusion(torch.cat([seg_out["prior"], context], dim=1))

        lmm_params = self.ep(fusion_context, seg)
        mu, log_sigma, coeffs, weights = torch.split(lmm_params, self.num_mix * 3, dim=1)
        lmm = Lmm(mu, log_sigma, weights, coeffs, self.no_multichannel_lmm)
        x_likelihoods = lmm(x)

        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": seg_out["likelihoods"]["y"],
                "z": seg_out["likelihoods"]["z"],
            },
        }

    @classmethod
    def from_state_dict(cls, stata_dict):
        num_ch = stata_dict["num_ch"].item()
        num_cls = stata_dict["num_cls"].item()
        num_mix = stata_dict["num_mix"].item()
        prior_ch = stata_dict["prior_ch"].item()
        context_ch = stata_dict["context_ch"].item()
        ep_ch = stata_dict["ep_ch"].item()
        noseg = stata_dict["no_seg"].item()
        no_multichannel_lmm = stata_dict["no_multichannel_lmm"].item()

        model = cls(num_ch, num_cls, prior_ch, context_ch, ep_ch, num_mix, noseg, no_multichannel_lmm)
        model.load_state_dict(stata_dict)
        return model


class EntropyModel_noseg(EntropyModel):
    def __init__(self, ep_ch, num_cls=1, num_mix=5, no_multichannel_lmm=False):
        super().__init__(ep_ch, num_cls, num_mix, no_multichannel_lmm)

    def forward(self, fusion_context, seg):
        del seg
        conv_outs = torch.stack([conv(fusion_context) for conv in self.conv_outs], dim=1)
        out = conv_outs.squeeze(1)
        return out


# class SeecNet_noseg(SeecNet):
#     def __init__(self, num_ch, num_cls=1, prior_ch=256, context_ch=256, ep_ch=256):
#         num_cls = 1
#         super().__init__(num_ch, num_cls, prior_ch, context_ch, ep_ch)
#         self.ep = EntropyModel_noseg(ep_ch, num_cls)

#     def forward(self, x, seg):
#         return super().forward(x, seg)


class SeecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log2 = torch.log(torch.tensor(2.0))

    def forward(self, x, output):
        assert x.ndim == 4  # B, C, H, W
        num_pixels = x.numel() / x.shape[1]
        out = {}
        out["z_bpp"] = -torch.log2(output["likelihoods"]["z"]).sum() / num_pixels
        out["y_bpp"] = -torch.log2(output["likelihoods"]["y"]).sum() / num_pixels
        out["latent_bpp"] = out["z_bpp"] + out["y_bpp"]
        out["x_bpp"] = -output["likelihoods"]["x"].sum() / (self.log2 * num_pixels)
        out["loss"] = out["x_bpp"] + out["latent_bpp"]

        return out
