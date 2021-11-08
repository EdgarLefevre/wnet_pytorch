# -*- coding: utf-8 -*-

import wnet.models.utils_model as um
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, filters, drop_r=0.5, attention=True, out_conv_dim=1):
        super(Unet, self).__init__()
        if out_conv_dim == 2:
            self.down1 = um.Down_Block(1, filters)
        else:
            self.down1 = um.Down_Block(2, filters)

        self.down2 = um.Down_Block(filters, filters * 2, drop_r)
        self.down3 = um.Down_Block(filters * 2, filters * 4, drop_r)
        self.down4 = um.Down_Block(filters * 4, filters * 8, drop_r)

        self.bridge = um.Bridge(filters * 8, filters * 16, drop_r)

        self.up1 = um.Up_Block(filters * 16, filters * 8, drop_r, attention)
        self.up2 = um.Up_Block(filters * 8, filters * 4, drop_r, attention)
        self.up3 = um.Up_Block(filters * 4, filters * 2, drop_r, attention)
        self.up4 = um.Up_Block(filters * 2, filters, drop_r, attention)

        self.outc = um.OutConv(filters, out_conv_dim)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        _, x = self.up1(bridge, c4)
        _, x = self.up2(x, c3)
        _, x = self.up3(x, c2)
        att, x = self.up4(x, c1)
        mask = self.outc(x)
        return mask, att


class Wnet_attention(nn.Module):
    def __init__(self, filters, drop_r=0.3):
        super(Wnet_attention, self).__init__()
        self.u_enc = Unet(filters, drop_r, True, out_conv_dim=2)
        self.u_dec = Unet(filters, drop_r, True)

    def forward_enc(self, x):
        mask, att = self.u_enc(x)
        return mask, att

    def forward(self, x):
        mask, att = self.u_enc(x)
        reconstruction, _ = self.u_dec(mask)
        return reconstruction
