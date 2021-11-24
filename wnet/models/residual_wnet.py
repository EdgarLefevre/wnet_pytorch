# -*- coding: utf-8 -*-

import wnet.models.utils_model as um
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, filters, drop_r=0.5):
        super(Unet, self).__init__()

        self.down1 = um.Res_conv_Block_down(1, filters)

        self.down2 = um.Res_conv_Block_down(filters, filters * 2)
        self.down3 = um.Res_conv_Block_down(filters * 2, filters * 4)
        self.down4 = um.Res_conv_Block_down(filters * 4, filters * 8)

        self.bridge = um.Bridge(filters * 8, filters * 16, drop_r)

        self.up1 = um.Res_conv_Block_up(filters * 16, filters * 8)
        self.up2 = um.Res_conv_Block_up(filters * 8, filters * 4)
        self.up3 = um.Res_conv_Block_up(filters * 4, filters * 2)
        self.up4 = um.Res_conv_Block_up(filters * 2, filters)

        self.outc = um.OutConv(filters, 1)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x = self.up1(bridge, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        mask = self.outc(x)
        return mask


class Wnet(nn.Module):
    def __init__(self, filters, drop_r=0.3):
        super(Wnet, self).__init__()
        self.u_enc = nn.DataParallel(Unet(filters, drop_r))
        self.u_dec = nn.DataParallel(Unet(filters, drop_r))

    def forward(self, x):
        mask = self.u_enc(x)
        reconstruction = self.u_dec(mask)
        return reconstruction, mask


class Unet_preact(nn.Module):
    def __init__(self, filters, drop_r=0.5):
        super(Unet_preact, self).__init__()

        self.down1 = um.Res_preactivation_down(1, filters)

        self.down2 = um.Res_preactivation_down(filters, filters * 2)
        self.down3 = um.Res_preactivation_down(filters * 2, filters * 4)
        self.down4 = um.Res_preactivation_down(filters * 4, filters * 8)

        self.bridge = um.Bridge(filters * 8, filters * 16, drop_r)

        self.up1 = um.Res_preactivation_down(filters * 16, filters * 8)
        self.up2 = um.Res_preactivation_down(filters * 8, filters * 4)
        self.up3 = um.Res_preactivation_down(filters * 4, filters * 2)
        self.up4 = um.Res_preactivation_down(filters * 2, filters)

        self.outc = um.OutConv(filters, 1)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x = self.up1(bridge, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        mask = self.outc(x)
        return mask


class Wnet_preact(nn.Module):
    def __init__(self, filters, drop_r=0.3):
        super(Wnet_preact, self).__init__()
        self.u_enc = nn.DataParallel(Unet(filters, drop_r))
        self.u_dec = nn.DataParallel(Unet(filters, drop_r))

    def forward(self, x):
        mask = self.u_enc(x)
        reconstruction = self.u_dec(mask)
        return reconstruction, mask
