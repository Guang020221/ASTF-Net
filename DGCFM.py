# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialFusion(nn.Module):
    def __init__(self, nf=96):
        super(DifferentialFusion, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        mid = max(nf // 4, 1)
        self.se_fc1    = nn.Conv2d(nf, mid, kernel_size=1, bias=True)
        self.se_relu   = nn.ReLU(inplace=True)
        self.se_fc2    = nn.Conv2d(mid, nf, kernel_size=1, bias=True)
        self.se_sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        w1     = self.act(self.global_avgpool(x1 - x2))
        w2     = self.act(self.global_avgpool(x2 - x1))
        F_init = w1 * x1 + w2 * x2
        se     = self.global_avgpool(F_init)
        se     = self.se_relu(self.se_fc1(se))
        se     = self.se_sigmoid(self.se_fc2(se))
        F_refined = F_init * se
        F_fused   = F_refined + F_init
        return F_fused


class CMFM_single(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln    = nn.LayerNorm(hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                stride=1, padding=1, groups=hidden_dim)

    def forward(self, x: torch.Tensor):
        x_permuted = x.permute(0, 2, 3, 1)
        x_norm     = self.ln(x_permuted).permute(0, 3, 1, 2)
        x_out      = self.dwconv(x_norm)
        return x_out

class DGCFM(nn.Module):


    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.input_norm_over  = nn.GroupNorm(8, hidden_dim)
        self.input_norm_under = nn.GroupNorm(8, hidden_dim)
        self.diff_fusion      = DifferentialFusion(hidden_dim)
        self.cmfm             = CMFM_single(hidden_dim)

    def forward(self, feat_over: torch.Tensor, feat_under: torch.Tensor):
        feat_over  = self.input_norm_over(feat_over)
        feat_under = self.input_norm_under(feat_under)

        feat_fused = self.diff_fusion(feat_over, feat_under)

        feat_fused = self.cmfm(feat_fused)

        feat_fused = feat_fused + (feat_over + feat_under) * 0.5

        return feat_fused