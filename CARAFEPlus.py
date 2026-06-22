# carafe_plus.py
# Implementation of CARAFE++ (unified upsampling / downsampling operator)
# Reference: "CARAFE++: Unified Content-Aware ReAssembly of FEatures" (Wang et al., TPAMI 2022).
# Key design: channel compressor (1x1), content encoder (k_encoder), kernel normalizer (softmax),
# and content-aware reassembly via unfold + weighted sum.
#
# Paper defaults used: Cm=16 (downsample), Cm=64 (upsample), k_encoder=3, k_reassembly=5, normalize=softmax.
# See paper sections 3.2-3.3 for formulation and hyperparameter choices.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFEPlus(nn.Module):
    """
    CARAFE++ operator (upsample or downsample)
    Args:
        channels: input feature channels (C)
        scale: integer scale factor s (e.g., 2)
        mode: 'upsample' or 'downsample'
        cm: compressed channels Cm (if None uses paper defaults: 64 for up, 16 for down)
        k_encoder: kernel size for content encoder (suggested kencoder = kreassembly - 2)
        k_reassembly: reassembly kernel size (kreassembly), e.g. 5
        norm: 'softmax' or 'sigmoid_normalize' (paper found softmax or normalized sigmoid good)
    """

    def __init__(self,
                 channels,
                 scale=2,
                 mode='upsample',
                 cm=None,
                 k_encoder=3,
                 k_reassembly=5,
                 norm='softmax'):

        super().__init__()
        assert mode in ('upsample', 'downsample')
        assert k_reassembly % 2 == 1
        self.C = channels
        self.s = scale
        self.mode = mode
        self.k_enc = k_encoder
        self.k_rea = k_reassembly
        self.norm = norm

        # default Cm per paper
        if cm is None:
            cm = 64 if mode == 'upsample' else 16
        self.Cm = cm

        # Channel compressor: 1x1 conv -> BN -> ReLU (paper uses BN+ReLU after compressor)
        self.channel_compress = nn.Sequential(
            nn.Conv2d(self.C, self.Cm, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.Cm),
            nn.ReLU(inplace=True)
        )

        # Content encoder:
        # - for upsample: output channels = s*s * (k_rea*k_rea)
        # - for downsample: use stride = s, output channels = k_rea*k_rea
        K2 = self.k_rea * self.k_rea
        if self.mode == 'upsample':
            out_ch = (self.s * self.s) * K2
            stride = 1
        else:  # downsample
            out_ch = K2
            stride = self.s

        self.content_encoder = nn.Conv2d(self.Cm, out_ch,
                                         kernel_size=self.k_enc,
                                         stride=stride,
                                         padding=self.k_enc // 2,
                                         bias=True)

        # small optional projection on input side for numerical stability (not in paper but often used)
        # (not necessary—commented out)
        # self.in_proj = nn.Identity()

    def _kernel_normalize(self, kernel: torch.Tensor) -> torch.Tensor:
        """
        kernel shape depends on mode:
         - upsample: (N, s*s, K2, H, W)
         - downsample: (N, K2, H', W')
        normalize over the K2 dimension for each output location and subpixel (if upsample).
        """
        if self.norm == 'softmax':
            # softmax over kernel positions (dim=2)
            k = F.softmax(kernel, dim=2)
            return k
        elif self.norm == 'sigmoid_normalize':
            k = torch.sigmoid(kernel)
            # normalize sum to 1 over K2
            s = k.sum(dim=2, keepdim=True)
            k = k / (s + 1e-6)
            return k
        else:
            raise ValueError('Unsupported norm: ' + str(self.norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W)
        returns:
          - if upsample: (N, C, H*s, W*s)
          - if downsample: (N, C, ceil(H/s), ceil(W/s))
        """
        N, C, H, W = x.shape
        s = self.s
        k_rea = self.k_rea
        K2 = k_rea * k_rea
        pad = k_rea // 2

        # 1) channel compress and encode kernels
        x_comp = self.channel_compress(x)  # (N, Cm, H, W)  or (N, Cm, H, W) depending on stride in encoder

        if self.mode == 'upsample':
            # predict kernels: (N, s*s*K2, H, W)
            kernels = self.content_encoder(x_comp)  # (N, s*s*K2, H, W)
            N, total_ch, Hk, Wk = kernels.shape
            assert Hk == H and Wk == W, "content encoder for upsample should preserve spatial dims"

            # reshape to (N, s*s, K2, H, W)
            kernels = kernels.view(N, s * s, K2, H, W)
            kernels = self._kernel_normalize(kernels)  # norm over K2
            # kernels: (N, s*s, K2, H, W)

            # 2) extract patches from input x with padding (reflect) and stride=1
            # unfold returns (N, C*K2, H*W)
            x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            x_unfold = F.unfold(x_pad, kernel_size=k_rea, stride=1)  # (N, C*K2, H*W)
            x_unfold = x_unfold.view(N, C, K2, H, W)  # (N, C, K2, H, W)

            # 3) expand for s*s subpixels
            # (N, 1, C, K2, H, W) -> expand along dim=1 to s*s
            x_unfold = x_unfold.unsqueeze(1).expand(-1, s * s, -1, -1, -1, -1)  # (N, s*s, C, K2, H, W)

            # kernels: (N, s*s, K2, H, W) -> need to broadcast into channel dim
            kernels_exp = kernels.unsqueeze(2)  # (N, s*s, 1, K2, H, W)

            # multiply and sum over K2 -> (N, s*s, C, H, W)
            out_sub = (x_unfold * kernels_exp).sum(dim=3)  # sum over K2

            # reorganize: (N, s*s, C, H, W) -> (N, C, s, s, H, W)
            out_sub = out_sub.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, s*s, H, W)
            out_sub = out_sub.view(N, C, s, s, H, W)               # (N, C, s, s, H, W)
            out_sub = out_sub.permute(0, 1, 4, 2, 5, 3).contiguous()  # (N, C, H, s, W, s)
            out = out_sub.view(N, C, H * s, W * s)
            return out

        else:  # downsample
            # content encoder has stride s, so it outputs spatial dims H' = ceil(H/s) maybe floor depending on conv behavior
            kernels = self.content_encoder(x_comp)  # (N, K2, H', W')
            N, total_ch, H_out, W_out = kernels.shape
            # kernels shape: (N, K2, H_out, W_out)
            kernels = kernels.view(N, K2, H_out, W_out)
            kernels = self._kernel_normalize(kernels)  # normalize over K2

            # extract patches from input x with padding and stride=s.
            # unfold with stride=s produces patches centered for each output location.
            x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            # using unfold with stride=s:
            x_unfold = F.unfold(x_pad, kernel_size=k_rea, dilation=1, padding=0, stride=s)  # (N, C*K2, H_out*W_out)
            x_unfold = x_unfold.view(N, C, K2, H_out, W_out)  # (N, C, K2, H_out, W_out)

            # kernels: (N, K2, H_out, W_out) -> expand channel dim
            kernels_exp = kernels.unsqueeze(1)  # (N, 1, K2, H_out, W_out)

            # weighted sum over K2 -> (N, C, H_out, W_out)
            out = (x_unfold * kernels_exp).sum(dim=2)  # sum over K2
            return out
