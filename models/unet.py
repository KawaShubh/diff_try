import torch
import torch.nn as nn
import torch.nn.functional as F

class WCESuperResUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=64):
        super().__init__()
        
        # Downsample blocks
        self.down1 = self._block(in_channels, base_channels)
        self.down2 = self._block(base_channels, base_channels*2)
        self.down3 = self._block(base_channels*2, base_channels*4)
        
        # Upsample blocks
        self.up1 = self._block(base_channels*6, base_channels*2)  # Skip connection
        self.up2 = self._block(base_channels*3, base_channels)
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Bilinear upsampling

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)          # [B, 64, H, W]
        d2 = self.down2(self.pool(d1)) # [B, 128, H/2, W/2]
        d3 = self.down3(self.pool(d2)) # [B, 256, H/4, W/4]
        
        # Upsample with skip connections
        u1 = self.upsample(d3)  # [B, 256, H/4, W/4] -> [B, 256, H/2, W/2]
        
        # Make sure sizes of u1 and d2 match before concatenating
        if u1.size(2) != d2.size(2) or u1.size(3) != d2.size(3):
            # Apply padding if necessary
            u1 = F.pad(u1, (0, d2.size(3) - u1.size(3), 0, d2.size(2) - u1.size(2)))
        
        u1 = self.up1(torch.cat([u1, d2], dim=1))  # [B, 128, H/2, W/2]
        
        u2 = self.upsample(u1)  # [B, 128, H/2, W/2] -> [B, 128, H, W]
        
        # Make sure sizes of u2 and d1 match before concatenating
        if u2.size(2) != d1.size(2) or u2.size(3) != d1.size(3):
            # Apply padding if necessary
            u2 = F.pad(u2, (0, d1.size(3) - u2.size(3), 0, d1.size(2) - u2.size(2)))
        
        u2 = self.up2(torch.cat([u2, d1], dim=1))  # [B, 64, H, W]
        
        return self.final(u2)  # [B, 4, H, W]
