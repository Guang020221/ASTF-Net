import math
import torch
import torch.nn as nn
from option import args


class eca_block(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=kernel_size,
                              bias=False, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, 1, c])
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        return x * inputs


class GatherExcite(nn.Module):
    def __init__(self, channels, extent=0, extra_params=False):
        super(GatherExcite, self).__init__()
        self.channels = channels
        self.extent = extent
        self.extra_params = extra_params

        self.fc1 = nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=1, stride=1, padding=0, bias=extra_params)

        if self.extra_params:
            self.fc2 = nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=1, stride=1, padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gathered = nn.functional.adaptive_avg_pool2d(x, 1)

        excitation = self.fc1(gathered)
        if self.extra_params:
            excitation = nn.functional.relu(excitation)
            excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        return x * excitation


class DenseLayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(DenseLayer, self).__init__()
        self.norm = nn.GroupNorm(1, num_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(num_channels, growth, kernel_size=3,
                              padding=1, bias=False)

    def forward(self, x):
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = torch.cat((x, out), 1)
        return out


def _make_gn(num_channels):
    if num_channels % 8 == 0:
        return nn.GroupNorm(8, num_channels)
    elif num_channels % 4 == 0:
        return nn.GroupNorm(4, num_channels)
    else:
        return nn.GroupNorm(1, num_channels)


class ImprovedDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 3, 1, 1),
            _make_gn(96),  # BN → GN
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 64, 3, 1, 1),
            _make_gn(64),  # BN → GN
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            _make_gn(32),  # BN → GN
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Conv2d(32, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        out = self.conv1(x)


        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        skip = self.skip(x)
        out = self.alpha * out + (1 - self.alpha) * skip
        out = torch.tanh(out)
        return out


class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        num_features = args.num_features
        growth = args.growth

        self.conv_1 = nn.Sequential(
            nn.Conv2d(args.in_channels, num_features, kernel_size=3,
                      padding=1, bias=False),
            _make_gn(num_features),
            nn.LeakyReLU(0.2, inplace=True)
        )

        modules = []
        for i in range(args.num_layers):
            modules.append(DenseLayer(num_features, growth))
            num_features += growth
            if (i + 1) % 2 == 0 and i < args.num_layers - 1:
                compression_channels = num_features // 2
                modules.append(nn.Sequential(
                    eca_block(num_features),
                    _make_gn(num_features),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_features, compression_channels,
                              1, 1, 0, bias=False),
                    _make_gn(compression_channels)
                ))
                num_features = compression_channels

        self.dense_layers = nn.Sequential(*modules)
        self.eca_final = eca_block(num_features)
        self.encoder_norm = nn.Sequential(
            _make_gn(num_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, 96, 1, 1, 0, bias=False),
            _make_gn(96)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.eca_final(x)
        x = self.encoder_norm(x)
        return x


class FusionDecoder(nn.Module):
    def __init__(self):
        super(FusionDecoder, self).__init__()
        self.sub = ImprovedDecoder(96, args.in_channels)

    def forward(self, fused_feat):
        out = self.sub(fused_feat)
        out = (out + 1.0) / 2.0
        return out


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.num_channels = 2 * args.in_channels
        self.num_features = args.num_features
        self.growth = args.growth

        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_features, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.num_features),
            nn.LeakyReLU(0.2, inplace=True)
        )

        modules = []
        for i in range(args.num_layers):
            modules.append(DenseLayer(self.num_features, self.growth))
            self.num_features += self.growth
            if (i + 1) % 2 == 0 and i < args.num_layers - 1:
                compression_channels = self.num_features // 2
                modules.append(nn.Sequential(
                    nn.BatchNorm2d(self.num_features),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.num_features, compression_channels,
                              1, 1, 0, bias=False),
                    nn.BatchNorm2d(compression_channels)
                ))
                self.num_features = compression_channels

        self.dense_layers = nn.Sequential(*modules)
        self.encoder_norm = nn.Sequential(
            nn.BatchNorm2d(self.num_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.num_features, 96, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96),
            nn.Tanh()
        )
        self.sub = ImprovedDecoder(96, args.in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.encoder_norm(x)
        x = self.sub(x)
        x = (x + 1.0) / 2.0
        return x

    def get_encoder_features(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.encoder_norm(x)
        return x