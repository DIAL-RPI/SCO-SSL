import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# single convolutional layer (Conv + BN + ReLU) followed by shadow dropout (Shadow-DROP)
class conv_shadow(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super(conv_shadow, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, shadow):
        y = self.conv(x)
        if shadow is not None:
           # shadow dropout
           y = y * shadow.expand(-1, y.shape[1], -1, -1, -1)
        return y
# double convolutional layers with Shadow-DROP
class double_conv_shadow(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_shadow, self).__init__()
        self.conv1 = conv_shadow(in_ch, out_ch, 3, padding=1)
        self.conv2 = conv_shadow(out_ch, out_ch, 3, padding=1)
    def forward(self, x, shadow=None):
        y = self.conv1(x, shadow)
        y = self.conv2(y, shadow)
        return y
# convolutional block with Shadow-DROP for encoding path
class enc_block_shadow(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(enc_block_shadow, self).__init__()
        self.conv = double_conv_shadow(in_ch, out_ch)
        self.down = nn.MaxPool3d(2)
    def forward(self, x, shadow=None):
        y_conv = self.conv(x, shadow)
        y = self.down(y_conv)
        return y, y_conv
# convolutional block with Shadow-DROP for decoding path
class dec_block_shadow(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dec_block_shadow, self).__init__()
        self.conv = double_conv_shadow(in_ch, out_ch)
        self.up = nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)
    def forward(self, x, shadow=None):
        y_conv = self.conv(x, shadow)
        y = self.up(y_conv)
        return y, y_conv

# double convolutional layers (without Shadow-DROP)
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y
# convolutional block (without Shadow-DROP) for encoding path
class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.down = nn.MaxPool3d(2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)
        return y, y_conv
# convolutional block (without Shadow-DROP) for decoding path
class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.up = nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)
        return y, y_conv

def concatenate(x1, x2):
    diffZ = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    diffX = x2.size()[4] - x1.size()[4]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2,
                    diffZ // 2, diffZ - diffZ//2))        
    y = torch.cat([x2, x1], dim=1)
    return y

# Shadow augmentation (Shadow-AUG)
def shadow_aug(image, cfg, order):
    tmp_tensor = image.detach()
    t_min = (1.0 - cfg['rs_intensity'][0]) / (cfg['rs_intensity'][1] - cfg['rs_intensity'][0])
    t_max = (cfg['shadow_threshold'] - cfg['rs_intensity'][0]) / (cfg['rs_intensity'][1] - cfg['rs_intensity'][0])
    tmp_tensor = torch.clamp(tmp_tensor, min=t_min, max=t_max)
    tmp_tensor = 0.5 * (torch.cos(np.pi * (tmp_tensor-t_min)/(t_max-t_min) + np.pi) + 1)
    shadow_mask = torch.zeros_like(tmp_tensor)
    if order == 'ascending':
        shadow_mask[0:1,:] = tmp_tensor[-1:,:]
        shadow_mask[1:,:] = tmp_tensor[0:-1,:]
    else:
        shadow_mask[0:-1,:] = tmp_tensor[1:,:]
        shadow_mask[-1:,:] = tmp_tensor[0:1,:]
    image = image * shadow_mask

    return image, shadow_mask

# A 3D U-Net equipped with Shadow-DROP in encoding path
class ShadowUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super(ShadowUNet, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch

        self.enc1 = enc_block_shadow(in_ch, base_ch)
        self.enc2 = enc_block_shadow(base_ch, base_ch*2)
        self.enc3 = enc_block_shadow(base_ch*2, base_ch*4)
        self.enc4 = enc_block_shadow(base_ch*4, base_ch*8)

        self.dec1 = dec_block(base_ch*8, base_ch*8)
        self.dec2 = dec_block(base_ch*8+base_ch*8, base_ch*4)
        self.dec3 = dec_block(base_ch*4+base_ch*4, base_ch*2)
        self.dec4 = dec_block(base_ch*2+base_ch*2, base_ch)
        self.lastconv = double_conv(base_ch+base_ch, base_ch)

        self.outconv = nn.Conv3d(base_ch, 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, shadow=None):
        if shadow is not None:
            enc1, enc1_conv = self.enc1(x, shadow)
            shadow1 = F.interpolate(shadow, size=[enc1.shape[2], enc1.shape[3], enc1.shape[4]], mode='trilinear')
            enc2, enc2_conv = self.enc2(enc1, shadow1)
            shadow2 = F.interpolate(shadow, size=[enc2.shape[2], enc2.shape[3], enc2.shape[4]], mode='trilinear')
            enc3, enc3_conv = self.enc3(enc2, shadow2)
            shadow3 = F.interpolate(shadow, size=[enc3.shape[2], enc3.shape[3], enc3.shape[4]], mode='trilinear')
            enc4, enc4_conv = self.enc4(enc3, shadow3)
            dec1, _ = self.dec1(enc4)
            dec2, _ = self.dec2(concatenate(dec1, enc4_conv))
            dec3, _ = self.dec3(concatenate(dec2, enc3_conv))
            dec4, _ = self.dec4(concatenate(dec3, enc2_conv))
            lastconv = self.lastconv(concatenate(dec4, enc1_conv))
        else:
            enc1, enc1_conv = self.enc1(x)
            enc2, enc2_conv = self.enc2(enc1)
            enc3, enc3_conv = self.enc3(enc2)
            enc4, enc4_conv = self.enc4(enc3)
            dec1, _ = self.dec1(enc4)
            dec2, _ = self.dec2(concatenate(dec1, enc4_conv))
            dec3, _ = self.dec3(concatenate(dec2, enc3_conv))
            dec4, _ = self.dec4(concatenate(dec3, enc2_conv))
            lastconv = self.lastconv(concatenate(dec4, enc1_conv))
        output_conv = self.outconv(lastconv)
        output = self.softmax(output_conv)

        return output

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def description(self):
        return 'U-Net equipped with Shadow-DROP in encoding path (input channel = {0:d}, base channel = {1:d})'.format(self.in_ch, self.base_ch)