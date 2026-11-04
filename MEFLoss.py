import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class Sobelxy(nn.Module):

    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class fusion_loss_mef(nn.Module):


    def __init__(self, total_epochs=200):
        super(fusion_loss_mef, self).__init__()
        self.sobelconv = Sobelxy()
        self.total_epochs = total_epochs

    def get_dynamic_weights(self, epoch, raw_ssim_value):
        if epoch < 10:
            w_l1, w_ssim, w_grad = 12.0, 6.0, 12.0
            stage = "Early"
            adaptive = f"Standard(SSIM={raw_ssim_value:.3f})"
        elif epoch < 25:
            w_l1, w_ssim, w_grad = 10.0, 5.0, 10.0
            stage = "Middle"
            adaptive = f"Balanced(SSIM={raw_ssim_value:.3f})"
        else:
            stage = "Late"
            if raw_ssim_value >= 0.90:
                w_l1, w_ssim, w_grad = 12.0, 4.0, 10.0
                adaptive = f"SSIM_VeryHigh(={raw_ssim_value:.3f})"
            elif raw_ssim_value >= 0.85:
                w_l1, w_ssim, w_grad = 10.0, 4.0, 7.0
                adaptive = f"SSIM_High(={raw_ssim_value:.3f})"
            else:
                w_l1, w_ssim, w_grad = 10.0, 6.0, 7.0
                adaptive = f"SSIM_Normal(={raw_ssim_value:.3f})"

        return w_l1, w_ssim, w_grad, stage, adaptive

    def forward(self, epoch, image_fused, img_over, img_under, weights=None):

        gradient_over = self.sobelconv(img_over)
        gradient_under = self.sobelconv(img_under)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_over, gradient_under)
        raw_grad = F.l1_loss(gradient_fused, gradient_joint)

        ssim_over_base = ms_ssim(img_over, image_fused, data_range=1.0, size_average=False)
        ssim_under_base = ms_ssim(img_under, image_fused, data_range=1.0, size_average=False)

        raw_ssim = 0.5 * ssim_over_base.mean() + 0.5 * ssim_under_base.mean()

        w_l1, w_ssim, w_grad, stage, adaptive = self.get_dynamic_weights(epoch, raw_ssim.item())

        if weights is not None:
            w_over = weights[:, 0:1, :, :]
            w_under = weights[:, 1:2, :, :]

            intensity_joint = w_over * img_over + w_under * img_under
            raw_l1 = F.l1_loss(image_fused, intensity_joint)
            loss_l1 = w_l1 * raw_l1

            global_w_over = w_over.mean(dim=[1, 2, 3])   # [B]
            global_w_under = w_under.mean(dim=[1, 2, 3]) # [B]
            weighted_ssim = (global_w_over * ssim_over_base + global_w_under * ssim_under_base).mean()
            loss_SSIM = w_ssim * (1.0 - weighted_ssim)

        else:
            intensity_joint = 0.5 * img_over + 0.5 * img_under
            raw_l1 = F.l1_loss(image_fused, intensity_joint)
            loss_l1 = w_l1 * raw_l1
            loss_SSIM = w_ssim * (1.0 - raw_ssim)

        loss_grad = w_grad * raw_grad
        fusion_loss = loss_l1 + loss_SSIM + loss_grad

        return fusion_loss, loss_l1, loss_SSIM, loss_grad