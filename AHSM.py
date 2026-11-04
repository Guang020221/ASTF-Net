import torch
import torch.nn as nn

def kaiming_init(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class IlluminationPredictor(nn.Module):
    def __init__(self, channels=16):
        super(IlluminationPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.apply(kaiming_init)

    def forward(self, x):
        return self.net(x)

class AHSM(nn.Module):

    def __init__(self):
        super(AHSM, self).__init__()
        self.predictor = IlluminationPredictor(channels=16)
        self.beta = nn.Parameter(torch.tensor([3.0]))
        self.num_iterations = 8

    def forward(self, x):
        x_curr = x
        safe_beta = 1.0 + torch.abs(self.beta - 1.0)

        for i in range(self.num_iterations):
            A_i = self.predictor(x_curr)
            x_curr = x_curr - A_i * torch.pow(x_curr + 1e-6, safe_beta)
            x_curr = torch.clamp(x_curr, 0.0, 1.0)

        return x_curr