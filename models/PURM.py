import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class BrightnessAdaptiveAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.local_bright_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attn = self.spatial(x)
        masked_x_sum = F.avg_pool2d(x * attn, kernel_size=31, stride=1, padding=15, divisor_override=1)
        attn_sum = F.avg_pool2d(attn, kernel_size=31, stride=1, padding=15, divisor_override=1)
        local_illum = masked_x_sum / (attn_sum + 1e-6)
        w_map = self.local_bright_net(local_illum)
        w_bright = w_map[:, 0:1, :, :]
        w_dark = w_map[:, 1:2, :, :]
        return w_dark * (x * attn) + w_bright * x


class PURM(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBnRelu(1, 32), ConvBnRelu(32, 32),
            ConvBnRelu(32, 32), ConvBnRelu(32, 32),
            nn.Conv2d(32, 4, 3, padding=1),
        )
        self.baa = BrightnessAdaptiveAttention()
        self.s2 = nn.Sequential(
            ConvBnRelu(1, 32), ConvBnRelu(32, 32),
            ConvBnRelu(32, 32), ConvBnRelu(32, 32),
            nn.Conv2d(32, 4, 3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        def step(y, r):
            y = y.clamp(1e-4, 1.0 - 1e-4)
            return (2.0 * y - y ** r).clamp(1e-4, 1.0 - 1e-4)


        r1_all = self.sigmoid(self.s1(x)) * 0.6 + 0.8
        r1, r2, r3, r4 = torch.split(r1_all, 1, dim=1)

        cur = x
        cur = step(cur, r1)
        cur = step(cur, r2)
        cur = step(cur, r3)
        enhance_stage1 = step(cur, r4)


        attn_out = self.baa(enhance_stage1)

        r2_all = self.sigmoid(self.s2(attn_out)) * 0.6 + 0.8
        r5, r6, r7, r8 = torch.split(r2_all, 1, dim=1)

        cur2 = enhance_stage1
        cur2 = step(cur2, r5)
        cur2 = step(cur2, r6)
        cur2 = step(cur2, r7)
        enhance_final = step(cur2, r8)

        return enhance_final, r1_all, r2_all, enhance_stage1