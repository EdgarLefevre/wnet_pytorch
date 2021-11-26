""" Parts of the U-Net model """

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DoubleSepConv(nn.Module):
    """(Separable convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleSepConv, self).__init__()
        self.double_conv = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_Block(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5):
        super(Down_Block, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(drop))

    def forward(self, x):
        c = self.conv(x)
        return c, self.down(c)


class Down_Sep_Block(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5):
        super(Down_Sep_Block, self).__init__()
        self.conv = DoubleSepConv(in_channels, out_channels)
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(drop))

    def forward(self, x):
        c = self.conv(x)
        return c, self.down(c)


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels, drop):
        super(Bridge, self).__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels), nn.Dropout(drop)
        )

    def forward(self, x):
        return self.conv(x)


class SepBridge(nn.Module):
    def __init__(self, in_channels, out_channels, drop):
        super(SepBridge, self).__init__()
        self.conv = nn.Sequential(
            DoubleSepConv(in_channels, out_channels), nn.Dropout(drop)
        )

    def forward(self, x):
        return self.conv(x)


class Up_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5, attention=False):
        super(Up_Block, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels), nn.Dropout(p=drop)
        )
        self.attention = attention
        if attention:
            self.gating = GatingSignal(in_channels, out_channels)
            self.att_gat = Attention_Gate(out_channels)

    def forward(self, x, conc):
        x1 = self.up(x)
        if self.attention:
            gat = self.gating(x)
            map, att = self.att_gat(conc, gat)
            x = torch.cat([x1, att], dim=1)
            return map, self.conv(x)
        else:
            x = torch.cat([conc, x1], dim=1)
            return self.conv(x)


class Up_Sep_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5, attention=False):
        super(Up_Sep_Block, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = nn.Sequential(
            DoubleSepConv(in_channels, out_channels), nn.Dropout(p=drop)
        )
        self.attention = attention
        if attention:
            self.gating = GatingSignal(in_channels, out_channels)
            self.att_gat = Attention_Gate(out_channels)

    def forward(self, x, conc):
        x1 = self.up(x)
        if self.attention:
            gat = self.gating(x)
            map, att = self.att_gat(conc, gat)
            x = torch.cat([x1, att], dim=1)
            return map, self.conv(x)
        else:
            x = torch.cat([conc, x1], dim=1)
            return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, sig=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if sig:
            self.activ = nn.Sigmoid()
        else:
            self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.activ(x)


class NewOutConv(nn.Module):
    def __init__(self, in_channels, out_channels, sig=False):
        super(NewOutConv, self).__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        if sig:
            self.activ = nn.Sigmoid()
        else:
            self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.activ(x)

    def forward(self, x):
        return self.conv(x)


class GatingSignal(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(GatingSignal, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x)


class Attention_Gate(nn.Module):
    def __init__(self, in_channels):
        super(Attention_Gate, self).__init__()
        self.conv_theta_x = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=(2, 2)
        )
        self.conv_phi_g = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.att = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x, gat):
        theta_x = self.conv_theta_x(x)
        phi_g = self.conv_phi_g(gat)
        res = torch.add(phi_g, theta_x)
        res = self.att(res)
        # print(res.size(), x.size())
        return res, torch.mul(res, x)


class Res_Block_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_Block_down, self).__init__()
        self.conv_relu1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        x1 = self.conv_relu1(x)
        x2 = self.conv_block(x1)
        x3 = torch.add(x1, x2)
        x = self.conv_relu2(x3)
        return x, self.down(x)


class Res_Block_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_Block_up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_relu1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, conc):
        xup = self.up(x)
        xconc = torch.cat([xup, conc], dim=1)
        x1 = self.conv_relu1(xconc)
        x2 = self.conv_block(x1)
        x3 = torch.add(x1, x2)
        x = self.conv_relu2(x3)
        return x


class Res_conv_Block_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_conv_Block_down, self).__init__()
        self.conv_relu1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_relu1(x)
        x2 = self.conv_block(x1)
        x_short = self.shortcut(x)
        x3 = torch.add(x_short, x2)
        x = self.conv_relu2(x3)
        return x, self.down(x)


class Res_conv_Block_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_conv_Block_up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_relu1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, conc):
        xup = self.up(x)
        xconc = torch.cat([xup, conc], dim=1)
        x1 = self.conv_relu1(xconc)
        x2 = self.conv_block(x1)
        x_short = self.shortcut(xconc)
        x3 = torch.add(x_short, x2)
        x = self.conv_relu2(x3)
        return x


class Res_preactivation_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_preactivation_down, self).__init__()
        self.conv_relu1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_relu1(x)
        x2 = self.conv_block(x1)
        x_short = self.shortcut(x)
        x3 = torch.add(x_short, x2)
        x = self.conv_relu2(x3)
        return x, self.down(x)


class Res_preactivation_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_preactivation_up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, conc):
        xup = self.up(x)
        xconc = torch.cat([xup, conc], dim=1)
        x2 = self.conv_block(xconc)
        x_short = self.shortcut(xconc)
        x3 = torch.add(x_short, x2)
        x = self.conv_relu2(x3)
        return x


class Res_Sep_preactivation_down(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.3):
        super(Res_Sep_preactivation_down, self).__init__()
        self.conv_relu1 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3)
        )
        self.conv_relu2 = nn.Sequential(
            SeparableConv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU()
        )
        self.dp = nn.Dropout(p=drop)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_relu1(x)
        x2 = self.conv_block(x1)
        x2 = self.dp(x2)
        x_short = self.shortcut(x)
        x = torch.add(x_short, x2)
        # x = self.conv_relu2(x)
        return x, self.down(x)



class Res_Sep_preactivation_up(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.3):
        super(Res_Sep_preactivation_up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3),
        )
        self.conv_relu2 = nn.Sequential(
            SeparableConv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU()
        )
        self.dp = nn.Dropout(p=drop)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, conc):
        xup = self.up(x)
        xconc = torch.cat([xup, conc], dim=1)
        x2 = self.conv_block(xconc)
        x2 = self.dp(x2)
        x_short = self.shortcut(xconc)
        x = torch.add(x_short, x2)
        # x = self.conv_relu2(x)
        return x
