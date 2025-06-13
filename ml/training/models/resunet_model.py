""" Full assembly of the parts to form the complete Residual U-Net network """

from shared.resunet.resunet_parts import *
import torch.utils.checkpoint as cp


class ResUNet(nn.Module):
    """Residual U-Net with skip connections and residual blocks"""
    
    def __init__(self, n_channels, n_classes, bilinear=False, use_attention=False):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        # Initial convolution
        self.inc = ResidualDoubleConv(n_channels, 64)
        
        # Encoder (downsampling path)
        self.down1 = ResDown(64, 128)
        self.down2 = ResDown(128, 256)
        self.down3 = ResDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = ResDown(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = ResUp(1024, 512 // factor, bilinear)
        self.up2 = ResUp(512, 256 // factor, bilinear)
        self.up3 = ResUp(256, 128 // factor, bilinear)
        self.up4 = ResUp(128, 64, bilinear)
        
        # Attention gates (optional)
        if use_attention:
            self.att1 = AttentionGate(in_channels_g=512 // factor, in_channels_x=512, int_channels=256)
            self.att2 = AttentionGate(in_channels_g=256 // factor, in_channels_x=256, int_channels=128)
            self.att3 = AttentionGate(in_channels_g=128 // factor, in_channels_x=128, int_channels=64)
            self.att4 = AttentionGate(in_channels_g=64, in_channels_x=64, int_channels=32)
        
        # Output convolution
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        if self.use_attention:
            x4_att = self.att1(x4, x5)
            x = self.up1(x5, x4_att)
            
            x3_att = self.att2(x3, x)
            x = self.up2(x, x3_att)
            
            x2_att = self.att3(x2, x)
            x = self.up3(x, x2_att)
            
            x1_att = self.att4(x1, x)
            x = self.up4(x, x1_att)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class DeepResUNet(nn.Module):
    """Deeper Residual U-Net with more layers"""
    
    def __init__(self, n_channels, n_classes, bilinear=False, use_attention=False):
        super(DeepResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        # Initial convolution
        self.inc = ResidualDoubleConv(n_channels, 32)
        
        # Encoder (downsampling path) - deeper network
        self.down1 = ResDown(32, 64)
        self.down2 = ResDown(64, 128)
        self.down3 = ResDown(128, 256)
        self.down4 = ResDown(256, 512)
        factor = 2 if bilinear else 1
        self.down5 = ResDown(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = ResUp(1024, 512 // factor, bilinear)
        self.up2 = ResUp(512, 256 // factor, bilinear)
        self.up3 = ResUp(256, 128 // factor, bilinear)
        self.up4 = ResUp(128, 64 // factor, bilinear)
        self.up5 = ResUp(64, 32, bilinear)
        
        # Attention gates (optional)
        if use_attention:
            self.att1 = AttentionGate(in_channels_g=512 // factor, in_channels_x=512, int_channels=256)
            self.att2 = AttentionGate(in_channels_g=256 // factor, in_channels_x=256, int_channels=128)
            self.att3 = AttentionGate(in_channels_g=128 // factor, in_channels_x=128, int_channels=64)
            self.att4 = AttentionGate(in_channels_g=64 // factor, in_channels_x=64, int_channels=32)
            self.att5 = AttentionGate(in_channels_g=32, in_channels_x=32, int_channels=16)
        
        # Output convolution
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        # Decoder path with skip connections
        if self.use_attention:
            x5_att = self.att1(x5, x6)
            x = self.up1(x6, x5_att)
            
            x4_att = self.att2(x4, x)
            x = self.up2(x, x4_att)
            
            x3_att = self.att3(x3, x)
            x = self.up3(x, x3_att)
            
            x2_att = self.att4(x2, x)
            x = self.up4(x, x2_att)
            
            x1_att = self.att5(x1, x)
            x = self.up5(x, x1_att)
        else:
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        
        logits = self.outc(x)
        return logits


# Convenience functions for easy model creation
def create_resunet(n_channels=1, n_classes=1, bilinear=False, use_attention=False):
    """Create a standard Residual U-Net model"""
    return ResUNet(n_channels, n_classes, bilinear, use_attention)


def create_deep_resunet(n_channels=1, n_classes=1, bilinear=False, use_attention=False):
    """Create a deeper Residual U-Net model"""
    return DeepResUNet(n_channels, n_classes, bilinear, use_attention)


def create_attention_resunet(n_channels=1, n_classes=1, bilinear=False):
    """Create a Residual U-Net with attention gates"""
    return ResUNet(n_channels, n_classes, bilinear, use_attention=True)
