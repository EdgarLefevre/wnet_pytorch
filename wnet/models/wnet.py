# -*- coding: utf-8 -*-

import wnet.models.utils_model as um
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, filters, drop_r=0.5):
        super(Unet, self).__init__()

        self.down1 = um.Down_Block(1, filters)

        self.down2 = um.Down_Block(filters, filters * 2, drop_r)
        self.down3 = um.Down_Block(filters * 2, filters * 4, drop_r)
        self.down4 = um.Down_Block(filters * 4, filters * 8, drop_r)

        self.bridge = um.Bridge(filters * 8, filters * 16, drop_r)

        self.up1 = um.Up_Block(filters * 16, filters * 8, drop_r, False)
        self.up2 = um.Up_Block(filters * 8, filters * 4, drop_r, False)
        self.up3 = um.Up_Block(filters * 4, filters * 2, drop_r, False)
        self.up4 = um.Up_Block(filters * 2, filters, drop_r, False)

        self.outc = um.NewOutConv(filters, 1)

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


class Unet_Sep(nn.Module):
    def __init__(self, filters, drop_r=0.5, sig=False):
        super(Unet_Sep, self).__init__()

        self.down1 = um.Down_Block(1, filters)

        self.down2 = um.Down_Sep_Block(filters, filters * 2, drop_r)
        self.down3 = um.Down_Sep_Block(filters * 2, filters * 4, drop_r)
        self.down4 = um.Down_Sep_Block(filters * 4, filters * 8, drop_r)

        self.bridge = um.SepBridge(filters * 8, filters * 16, drop_r)

        self.up1 = um.Up_Sep_Block(filters * 16, filters * 8, drop_r, False)
        self.up2 = um.Up_Sep_Block(filters * 8, filters * 4, drop_r, False)
        self.up3 = um.Up_Sep_Block(filters * 4, filters * 2, drop_r, False)

        self.up4 = um.Up_Block(filters * 2, filters, drop_r, False)

        self.outc = um.NewOutConv(filters, 1, sig)

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

    def forward_enc(self, x):
        return self.u_enc(x)

    def forward(self, x):
        mask = self.u_enc(x)
        reconstruction = self.u_dec(mask)
        return reconstruction


class WnetSep(nn.Module):
    def __init__(self, filters, drop_r=0.3):
        super(WnetSep, self).__init__()
        self.u_enc = nn.DataParallel(Unet_Sep(filters, drop_r, sig=True))
        self.u_dec = nn.DataParallel(Unet_Sep(filters, drop_r))

    def forward(self, x):
        mask = self.u_enc(x)
        reconstruction = self.u_dec(mask)
        return reconstruction, mask