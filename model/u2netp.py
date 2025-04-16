# model/u2netp.py

import torch
import torch.nn as nn

# --- 基本構造ブロック定義 ---
class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(REBNCONV, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

def _upsample_like(src, tar):
    return nn.functional.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

# --- RSU Block (簡略版) ---
class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv = REBNCONV(out_ch, out_ch)

    def forward(self, x):
        hx = self.rebnconvin(x)
        h = self.rebnconv(hx)
        return h + hx

# --- U2NETP（軽量版U²-Net） ---
class U2NETP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()
        self.stage1 = RSU7(in_ch, 16, 64)
        self.outconv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        hx = self.stage1(x)
        d0 = self.outconv(hx)
        return [d0]
