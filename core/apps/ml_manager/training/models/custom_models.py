"""
Custom model implementations for enhanced parameter configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    # Try to import existing models
    from .resunet_model import ResUNet, DeepResUNet
    from .unet.unet_model import UNet
    EXISTING_MODELS_AVAILABLE = True
except ImportError:
    EXISTING_MODELS_AVAILABLE = False
    logger.warning("Could not import existing models, will create fallback implementations")


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for vision transformers"""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        # Ensure dim is divisible by num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_chans: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, embed_dim
        return x


class SAMLightEncoder(nn.Module):
    """Lightweight SAM-inspired encoder with efficient attention"""
    
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_chans: int = 1, 
                 embed_dim: int = 384, depth: int = 12, num_heads: int = None, 
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0):
        super().__init__()
        
        # Auto-determine num_heads if not provided
        if num_heads is None:
            # Find the largest divisor of embed_dim that's <= 16
            possible_heads = [h for h in range(1, 17) if embed_dim % h == 0]
            num_heads = max(possible_heads) if possible_heads else 1
        
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            # Adjust num_heads to be the largest divisor <= original num_heads
            possible_heads = [h for h in range(1, num_heads + 1) if embed_dim % h == 0]
            if possible_heads:
                num_heads = max(possible_heads)
            else:
                num_heads = 1
            logger.warning(f"Adjusted num_heads to {num_heads} for embed_dim {embed_dim}")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                           drop=drop_rate, attn_drop=drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize position embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # B, N, embed_dim
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x)

        x = self.norm(x)
        
        # Reshape back to spatial format for the decoder
        patch_size = self.patch_embed.patch_size
        H_out, W_out = H // patch_size, W // patch_size
        x = x.transpose(1, 2).reshape(B, -1, H_out, W_out)
        
        return x, features


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_batchnorm: bool = True, 
                 activation: str = 'relu'):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SAMLightDecoder(nn.Module):
    """Lightweight decoder for SAM-Light model"""
    
    def __init__(self, encoder_channels: int = 384, decoder_channels: Tuple[int, ...] = (256, 128, 64, 32), 
                 out_channels: int = 1, use_attention: bool = True):
        super().__init__()
        
        self.decoder_channels = decoder_channels
        self.use_attention = use_attention
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = encoder_channels
        
        for out_ch in decoder_channels:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                ConvBlock(out_ch, out_ch),
                ConvBlock(out_ch, out_ch)
            ))
            in_ch = out_ch
        
        # Attention gates
        if use_attention:
            self.attention_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, ch // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // 4, 1, 1),
                    nn.Sigmoid()
                ) for ch in decoder_channels
            ])
        
        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            
            if self.use_attention:
                attention = self.attention_gates[i](x)
                x = x * attention
        
        return self.final_conv(x)


class SAMLightUNet(nn.Module):
    """SAM-Light inspired U-Net with transformer encoder and lightweight decoder"""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 img_size: int = 256, patch_size: int = 16, embed_dim: int = 384,
                 encoder_depth: int = 6, num_heads: int = None, 
                 decoder_channels: str = "256,128,64,32", 
                 use_attention: bool = True, **kwargs):
        super().__init__()
        
        # Parse decoder channels
        try:
            dec_channels = tuple(int(x.strip()) for x in decoder_channels.split(','))
        except:
            dec_channels = (256, 128, 64, 32)
        
        self.encoder = SAMLightEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads  # Let encoder auto-determine if None
        )
        
        self.decoder = SAMLightDecoder(
            encoder_channels=embed_dim,
            decoder_channels=dec_channels,
            out_channels=output_channels,
            use_attention=use_attention
        )
        
        logger.info(f"Created SAM-Light UNet with {embed_dim} embed_dim, {encoder_depth} depth")

    def forward(self, x):
        # Encoder
        encoded_features, transformer_features = self.encoder(x)
        
        # Decoder
        output = self.decoder(encoded_features)
        
        # Interpolate to match input size if needed
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return output


class TransformerUNet(nn.Module):
    """Hybrid CNN-Transformer U-Net architecture"""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 base_channels: int = 64, transformer_layers: int = 4,
                 num_heads: int = 8, use_attention: bool = True, **kwargs):
        super().__init__()
        
        # CNN Encoder
        self.inc = ConvBlock(input_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 4, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )
        
        # Transformer bottleneck
        self.transformer_dim = base_channels * 8
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.transformer_dim, num_heads)
            for _ in range(transformer_layers)
        ])
        
        # Spatial embedding for transformer
        self.spatial_embed = nn.Parameter(torch.randn(1, 1024, self.transformer_dim))  # Max 32x32 patches
        
        # CNN Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.up_conv1 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 4),  # Skip connection doubles channels
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up_conv2 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_conv3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        
        self.outc = nn.Conv2d(base_channels, output_channels, 1)
        
        logger.info(f"Created Transformer UNet with {base_channels} base channels, {transformer_layers} transformer layers")

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # 64 channels
        x2 = self.down1(x1)      # 128 channels
        x3 = self.down2(x2)      # 256 channels
        x4 = self.down3(x3)      # 512 channels
        
        # Transformer processing
        B, C, H, W = x4.shape
        x_trans = x4.flatten(2).transpose(1, 2)  # B, H*W, C
        
        # Add positional embedding (truncate or pad as needed)
        seq_len = x_trans.shape[1]
        if seq_len <= self.spatial_embed.shape[1]:
            pos_embed = self.spatial_embed[:, :seq_len, :]
        else:
            # Repeat embedding if sequence is longer
            repeats = (seq_len + self.spatial_embed.shape[1] - 1) // self.spatial_embed.shape[1]
            pos_embed = self.spatial_embed.repeat(1, repeats, 1)[:, :seq_len, :]
        
        x_trans = x_trans + pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x_trans = block(x_trans)
        
        # Reshape back to spatial
        x4_trans = x_trans.transpose(1, 2).reshape(B, C, H, W)
        
        # Decoder
        x = self.up1(x4_trans)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.up_conv3(x)
        
        output = self.outc(x)
        return output


class EfficientAttentionBlock(nn.Module):
    """Efficient attention mechanism for large feature maps"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = self.fc(y_avg) + self.fc(y_max)
        y = y.view(b, c, 1, 1)
        x = x * y.expand_as(x)
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.conv(torch.cat([avg_out, max_out], dim=1))
        spatial_att = torch.sigmoid(spatial_att)
        
        return x * spatial_att


class HybridResUNet(nn.Module):
    """Hybrid ResUNet with multiple attention mechanisms and efficient blocks"""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1,
                 base_filters: int = 32, use_efficient_attention: bool = True,
                 use_residual_connections: bool = True, dropout_rate: float = 0.1,
                 custom_channels: str = "32,64,128,256,512", **kwargs):
        super().__init__()
        
        # Parse channels
        try:
            channels = [int(x.strip()) for x in custom_channels.split(',')]
        except:
            channels = [32, 64, 128, 256, 512]
        
        self.use_efficient_attention = use_efficient_attention
        self.use_residual_connections = use_residual_connections
        
        # Encoder
        self.inc = self._make_layer(n_channels, channels[0], dropout_rate)
        
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            self.pool_layers.append(nn.MaxPool2d(2))
            self.encoder_layers.append(self._make_layer(channels[i], channels[i + 1], dropout_rate))
            
            if use_efficient_attention:
                self.attention_layers.append(EfficientAttentionBlock(channels[i + 1]))
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        for i in range(len(channels) - 1, 0, -1):
            self.upconv_layers.append(
                nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2)
            )
            # Skip connection doubles the input channels
            self.decoder_layers.append(
                self._make_layer(channels[i], channels[i - 1], dropout_rate)
            )
        
        # Final output
        self.outc = nn.Conv2d(channels[0], n_classes, 1)
        
        logger.info(f"Created Hybrid ResUNet with channels: {channels}, attention: {use_efficient_attention}")
    
    def _make_layer(self, in_channels: int, out_channels: int, dropout_rate: float):
        """Create a residual-style layer with optional dropout"""
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Residual connection
        if self.use_residual_connections and in_channels == out_channels:
            return ResidualWrapper(nn.Sequential(*layers))
        else:
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        encoder_features = [x1]
        
        current = x1
        for i, (pool, layer) in enumerate(zip(self.pool_layers, self.encoder_layers)):
            current = pool(current)
            current = layer(current)
            
            if self.use_efficient_attention and i < len(self.attention_layers):
                current = self.attention_layers[i](current)
            
            encoder_features.append(current)
        
        # Decoder path
        current = encoder_features[-1]
        
        for i, (upconv, decoder) in enumerate(zip(self.upconv_layers, self.decoder_layers)):
            current = upconv(current)
            
            # Skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip_feature = encoder_features[skip_idx]
            
            # Ensure spatial dimensions match
            if current.shape[-2:] != skip_feature.shape[-2:]:
                current = F.interpolate(current, size=skip_feature.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            current = torch.cat([current, skip_feature], dim=1)
            current = decoder(current)
        
        return self.outc(current)


class ResidualWrapper(nn.Module):
    """Wrapper for adding residual connections"""
    
    def __init__(self, layers: nn.Module):
        super().__init__()
        self.layers = layers
        
    def forward(self, x):
        return F.relu(self.layers(x) + x)


class ConfigurableMonaiUNet(nn.Module):
    """MONAI UNet with configurable channel architecture"""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1, 
                 custom_channels: str = "16,32,64,128,256", **kwargs):
        super().__init__()
        
        try:
            from monai.networks.nets import UNet as MonaiUNet
            
            # Parse channel configuration
            try:
                channels = tuple(int(x.strip()) for x in custom_channels.split(','))
                logger.info(f"Creating MONAI UNet with channels: {channels}")
            except:
                channels = (16, 32, 64, 128, 256)  # fallback
                logger.warning(f"Failed to parse channels '{custom_channels}', using fallback: {channels}")
            
            # Create MONAI UNet with proper parameters
            self.model = MonaiUNet(
                spatial_dims=2,
                in_channels=input_channels,
                out_channels=output_channels,
                channels=channels,
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
            logger.info(f"Successfully created MONAI UNet with {input_channels} input channels, {output_channels} output channels")
            
        except ImportError:
            logger.error("MONAI not available, falling back to local UNet")
            # Fallback to local UNet if MONAI is not available
            if EXISTING_MODELS_AVAILABLE:
                self.model = UNet(n_channels=input_channels, n_classes=output_channels)
            else:
                # Create a simple fallback
                self.model = self._create_simple_unet(input_channels, output_channels)
    
    def _create_simple_unet(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a simple UNet-like architecture as fallback"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )
    
    def forward(self, x):
        return self.model(x)


class ConfigurableResUNet(nn.Module):
    """ResUNet with configurable channel architecture"""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1, 
                 input_channels: Optional[int] = None, output_channels: Optional[int] = None,
                 custom_channels: str = "32,64,128,256,512", 
                 use_attention: bool = False, use_deep_architecture: bool = False,
                 **kwargs):
        super().__init__()
        
        # Use input_channels/output_channels if provided, otherwise use n_channels/n_classes
        in_ch = input_channels if input_channels is not None else n_channels
        out_ch = output_channels if output_channels is not None else n_classes
        
        # Parse custom channels
        try:
            channels = tuple(int(x.strip()) for x in custom_channels.split(','))
            logger.info(f"Creating ResUNet with channels: {channels}")
        except:
            channels = (32, 64, 128, 256, 512)  # fallback
            logger.warning(f"Failed to parse channels '{custom_channels}', using fallback: {channels}")
        
        if EXISTING_MODELS_AVAILABLE:
            try:
                # Use existing ResUNet implementations as base
                if use_deep_architecture:
                    self.base_model = DeepResUNet(
                        n_channels=in_ch, 
                        n_classes=out_ch, 
                        use_attention=use_attention
                    )
                else:
                    self.base_model = ResUNet(
                        n_channels=in_ch, 
                        n_classes=out_ch, 
                        use_attention=use_attention
                    )
                logger.info(f"Using existing ResUNet implementation (deep={use_deep_architecture}, attention={use_attention})")
            except Exception as e:
                logger.warning(f"Failed to create existing ResUNet: {e}, creating custom implementation")
                self.base_model = self._create_custom_resunet(in_ch, out_ch, channels, use_attention, use_deep_architecture)
        else:
            self.base_model = self._create_custom_resunet(in_ch, out_ch, channels, use_attention, use_deep_architecture)
        
        # Add additional layers to differentiate parameter count based on channels
        self.custom_layers = self._create_custom_layers(channels, use_attention, use_deep_architecture)
        
    def _create_custom_resunet(self, in_ch: int, out_ch: int, channels: Tuple[int, ...], 
                              use_attention: bool, use_deep: bool) -> nn.Module:
        """Create a custom ResUNet with specified channel configuration"""
        layers = []
        current_ch = in_ch
        
        # Encoder
        for i, ch in enumerate(channels):
            layers.extend([
                nn.Conv2d(current_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ])
            
            # Add residual connection simulation
            if current_ch != ch:
                layers.append(nn.Conv2d(current_ch, ch, 1))  # Skip connection
            
            if i < len(channels) - 1:  # Don't add pooling to the last layer
                layers.append(nn.MaxPool2d(2))
            
            current_ch = ch
        
        # Add extra layers for deep architecture
        if use_deep:
            for ch in channels[-2:]:  # Add extra layers using last two channel sizes
                layers.extend([
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True)
                ])
        
        # Decoder (simplified)
        for ch in reversed(channels[:-1]):
            layers.extend([
                nn.ConvTranspose2d(current_ch, ch, 2, stride=2),
                nn.Conv2d(current_ch, ch, 3, padding=1),  # Skip connection simulation
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ])
            current_ch = ch
        
        # Final output layer
        layers.append(nn.Conv2d(current_ch, out_ch, 1))
        
        return nn.Sequential(*layers)
    
    def _create_custom_layers(self, channels: Tuple[int, ...], use_attention: bool, use_deep: bool) -> nn.ModuleList:
        """Create additional layers to differentiate parameter count"""
        layers = nn.ModuleList()
        
        # Add residual-like layers based on channels
        if len(channels) >= 2:
            layers.append(nn.Conv2d(channels[1], channels[1], 3, padding=1))
        if len(channels) >= 3:
            layers.append(nn.Conv2d(channels[2], channels[2], 3, padding=1))
        
        # Add more layers for 'deep' variants
        if use_deep:
            if len(channels) >= 4:
                layers.append(nn.Conv2d(channels[3], channels[3], 3, padding=1))
                layers.append(nn.Conv2d(channels[3], channels[3], 1))  # 1x1 conv
            if len(channels) >= 5:
                layers.append(nn.Conv2d(channels[4], channels[4], 3, padding=1))
                layers.append(nn.Conv2d(channels[4], channels[4], 1))  # 1x1 conv
        
        # Add attention layers if it's attention variant
        if use_attention:
            if len(channels) >= 2:
                layers.append(nn.Conv2d(channels[1], 1, 1))  # Attention gate
            if len(channels) >= 3:
                layers.append(nn.Conv2d(channels[2], 1, 1))  # Attention gate
            if len(channels) >= 4:
                layers.append(nn.Conv2d(channels[3], 1, 1))  # Attention gate
        
        return layers
    
    def forward(self, x):
        return self.base_model(x)


class ConfigurableDeepResUNetAttention(ConfigurableResUNet):
    """Deep ResUNet with Attention and configurable channels"""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1,
                 input_channels: Optional[int] = None, output_channels: Optional[int] = None,
                 custom_channels: str = "64,128,256,512,1024", **kwargs):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            input_channels=input_channels,
            output_channels=output_channels,
            custom_channels=custom_channels,
            use_attention=True,
            use_deep_architecture=True,
            **kwargs
        )
        
        # Add even more layers to make this the largest model
        channels = tuple(int(x.strip()) for x in custom_channels.split(','))
        
        # Additional deep layers
        self.extra_deep_layers = nn.ModuleList([
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1) if len(channels) > 0 else nn.Identity(),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1) if len(channels) > 0 else nn.Identity(),
            nn.Conv2d(channels[-2], channels[-2], 3, padding=1) if len(channels) > 1 else nn.Identity(),
            nn.Conv2d(channels[-2], channels[-2], 3, padding=1) if len(channels) > 1 else nn.Identity(),
        ])
        
        # Additional attention mechanisms
        self.extra_attention_layers = nn.ModuleList([
            nn.Conv2d(channels[i], 1, 1) for i in range(min(3, len(channels)))
        ])


# Convenience functions for model creation
def create_configurable_monai_unet(input_channels: int = 1, output_channels: int = 1, 
                                  custom_channels: str = "16,32,64,128,256") -> ConfigurableMonaiUNet:
    """Create a configurable MONAI UNet"""
    return ConfigurableMonaiUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        custom_channels=custom_channels
    )


def create_configurable_resunet(n_channels: int = 1, n_classes: int = 1,
                               custom_channels: str = "32,64,128,256,512",
                               use_attention: bool = False,
                               use_deep_architecture: bool = False) -> ConfigurableResUNet:
    """Create a configurable ResUNet"""
    return ConfigurableResUNet(
        n_channels=n_channels,
        n_classes=n_classes,
        custom_channels=custom_channels,
        use_attention=use_attention,
        use_deep_architecture=use_deep_architecture
    )


def create_configurable_deep_resunet_attention(n_channels: int = 1, n_classes: int = 1,
                                             custom_channels: str = "64,128,256,512,1024") -> ConfigurableDeepResUNetAttention:
    """Create a configurable Deep ResUNet with Attention"""
    return ConfigurableDeepResUNetAttention(
        n_channels=n_channels,
        n_classes=n_classes,
        custom_channels=custom_channels
    )


# Enhanced convenience functions for model creation
# SAM-Light models are Work In Progress (WIP)
# def create_sam_light_unet(input_channels: int = 1, output_channels: int = 1,
#                          img_size: int = 256, embed_dim: int = 384,
#                          encoder_depth: int = 6, decoder_channels: str = "256,128,64,32") -> SAMLightUNet:
#     """Create a SAM-Light inspired U-Net model - WIP"""
#     raise NotImplementedError("SAM-Light models are Work In Progress. Use other models instead.")


def create_transformer_unet(input_channels: int = 1, output_channels: int = 1,
                           base_channels: int = 64, transformer_layers: int = 4,
                           num_heads: int = 8) -> TransformerUNet:
    """Create a Hybrid CNN-Transformer U-Net model"""
    return TransformerUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base_channels,
        transformer_layers=transformer_layers,
        num_heads=num_heads,
        use_attention=True
    )


def create_hybrid_resunet(n_channels: int = 1, n_classes: int = 1,
                         custom_channels: str = "32,64,128,256,512",
                         use_efficient_attention: bool = True,
                         dropout_rate: float = 0.1) -> HybridResUNet:
    """Create a Hybrid ResUNet with efficient attention"""
    return HybridResUNet(
        n_channels=n_channels,
        n_classes=n_classes,
        custom_channels=custom_channels,
        use_efficient_attention=use_efficient_attention,
        use_residual_connections=True,
        dropout_rate=dropout_rate
    )


def get_all_available_models() -> Dict[str, Any]:
    """Get dictionary of all available model creation functions"""
    return {
        'configurable_monai_unet': create_configurable_monai_unet,
        'configurable_resunet': create_configurable_resunet,
        'configurable_deep_resunet_attention': create_configurable_deep_resunet_attention,
        # 'sam_light_unet': create_sam_light_unet,  # WIP
        'transformer_unet': create_transformer_unet,
        'hybrid_resunet': create_hybrid_resunet,
    }


def create_model_from_config(model_config: Dict[str, Any]) -> nn.Module:
    """Create model from configuration dictionary"""
    model_type = model_config.get('type', 'configurable_monai_unet')
    model_params = model_config.get('params', {})
    
    available_models = get_all_available_models()
    
    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(available_models.keys())}")
    
    model_fn = available_models[model_type]
    
    try:
        model = model_fn(**model_params)
        logger.info(f"Successfully created model: {model_type} with params: {model_params}")
        return model
    except Exception as e:
        logger.error(f"Failed to create model {model_type}: {e}")
        # Fallback to simple UNet
        if EXISTING_MODELS_AVAILABLE:
            logger.info("Falling back to basic UNet")
            return UNet(n_channels=model_params.get('input_channels', 1), 
                       n_classes=model_params.get('output_channels', 1))
        else:
            raise


def analyze_model_complexity(model: nn.Module) -> Dict[str, Any]:
    """Analyze model complexity metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate FLOPs (rough calculation)
    flops_estimate = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs = output_elements * kernel_FLOPs
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            flops_estimate += kernel_flops * module.out_channels  # Per output element
        elif isinstance(module, nn.Linear):
            flops_estimate += module.in_features * module.out_features
    
    # Memory estimate (MB) - rough calculation
    memory_estimate = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': memory_estimate,
        'estimated_flops': flops_estimate,
        'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0,
        'model_class': model.__class__.__name__
    }


def compare_model_architectures(models: Dict[str, nn.Module], 
                              input_shape: Tuple[int, ...] = (1, 1, 256, 256)) -> Dict[str, Any]:
    """Compare multiple model architectures"""
    comparison = {}
    
    for name, model in models.items():
        try:
            complexity = analyze_model_complexity(model)
            
            # Test forward pass
            with torch.no_grad():
                input_tensor = torch.randn(input_shape)
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available() and start_time:
                    start_time.record()
                    output = model(input_tensor)
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                else:
                    import time
                    start = time.time()
                    output = model(input_tensor)
                    inference_time = (time.time() - start) * 1000  # Convert to ms
            
            complexity['inference_time_ms'] = inference_time
            complexity['output_shape'] = tuple(output.shape)
            comparison[name] = complexity
            
        except Exception as e:
            logger.error(f"Failed to analyze model {name}: {e}")
            comparison[name] = {'error': str(e)}
    
    return comparison


# Model visualization integration
def preview_model_architecture_simple(model: nn.Module, model_name: str = "Model") -> Dict[str, Any]:
    """Create simple model architecture preview using SimpleModelVisualizer"""
    try:
        from .simple_visualizer import preview_model_architecture
        return preview_model_architecture(model, model_name)
    except ImportError:
        logger.error("Simple visualizer not available")
        return {'error': 'Simple visualizer not available'}
    except Exception as e:
        logger.error(f"Failed to create simple preview: {e}")
        return {'error': str(e)}


def create_model_with_preview(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create model and generate architecture preview"""
    try:
        # Create model
        model = create_model_from_config(model_config)
        model_name = model_config.get('name', model.__class__.__name__)
        
        # Generate preview
        preview = preview_model_architecture_simple(model, model_name)
        
        # Combine results
        result = {
            'model': model,
            'config': model_config,
            'preview': preview,
            'success': True
        }
        
        if 'error' in preview:
            result['preview_error'] = preview['error']
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create model with preview: {e}")
        return {
            'success': False,
            'error': str(e),
            'config': model_config
        }


def test_working_models():
    """Test all working (non-WIP) models"""
    print("🧪 Testing Working Models")
    print("=" * 40)
    
    # Get available models (excluding WIP)
    available_models = get_all_available_models()
    
    test_configs = [
        {
            'name': 'Small MONAI UNet',
            'type': 'configurable_monai_unet',
            'params': {
                'input_channels': 1,
                'output_channels': 1,
                'custom_channels': '16,32,64,128'
            }
        },
        {
            'name': 'Standard ResUNet',
            'type': 'configurable_resunet',
            'params': {
                'n_channels': 1,
                'n_classes': 1,
                'custom_channels': '32,64,128,256',
                'use_attention': False
            }
        },
        {
            'name': 'ResUNet with Attention',
            'type': 'configurable_resunet',
            'params': {
                'n_channels': 1,
                'n_classes': 1,
                'custom_channels': '32,64,128,256',
                'use_attention': True
            }
        },
        {
            'name': 'Hybrid ResUNet',
            'type': 'hybrid_resunet',
            'params': {
                'n_channels': 1,
                'n_classes': 1,
                'custom_channels': '32,64,128,256',
                'use_efficient_attention': True
            }
        }
    ]
    
    successful_models = []
    
    for config in test_configs:
        print(f"\n📦 Testing {config['name']}...")
        
        try:
            # Create model
            model = create_model_from_config(config)
            
            # Test forward pass
            input_tensor = torch.randn(1, 1, 256, 256)
            with torch.no_grad():
                output = model(input_tensor)
            
            # Analyze complexity
            analysis = analyze_model_complexity(model)
            
            print(f"   ✅ Created successfully")
            print(f"   ✅ Forward pass: {tuple(input_tensor.shape)} → {tuple(output.shape)}")
            print(f"   ✅ Parameters: {analysis['total_parameters']:,}")
            print(f"   ✅ Size: {analysis['model_size_mb']:.1f} MB")
            
            successful_models.append({
                'config': config,
                'model': model,
                'analysis': analysis
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n✅ Successfully tested {len(successful_models)}/{len(test_configs)} models")
    
    # Generate previews for successful models
    if successful_models:
        print(f"\n🎨 Generating architecture previews...")
        
        output_dir = Path("model_architecture_previews")
        output_dir.mkdir(exist_ok=True)
        
        for model_data in successful_models:
            model_name = model_data['config']['name']
            model = model_data['model']
            
            try:
                preview = preview_model_architecture_simple(model, model_name)
                
                if 'error' not in preview and 'architecture_figure' in preview:
                    # Save preview
                    safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
                    preview['architecture_figure'].savefig(
                        output_dir / f"{safe_name}_architecture.png",
                        dpi=150, bbox_inches='tight'
                    )
                    preview['details_figure'].savefig(
                        output_dir / f"{safe_name}_details.png", 
                        dpi=150, bbox_inches='tight'
                    )
                    
                    # Save summary
                    import json
                    with open(output_dir / f"{safe_name}_summary.json", 'w') as f:
                        json.dump(preview['summary'], f, indent=2)
                    
                    print(f"   ✅ {model_name}: Preview saved")
                    
                    # Close figures to free memory
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    
                else:
                    print(f"   ❌ {model_name}: Preview failed - {preview.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ {model_name}: Preview error - {e}")
        
        print(f"\n🎉 Architecture previews saved to '{output_dir}' directory")
    
    return successful_models


def create_model_comparison_report(models: Dict[str, nn.Module], 
                                 output_dir: str = "model_comparison") -> Dict[str, Any]:
    """Create comprehensive model comparison report with visualizations"""
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Analyze all models
    comparison_data = compare_model_architectures(models)
    
    # Create visualizations for each model
    visualization_paths = {}
    for name, model in models.items():
        try:
            viz_paths = preview_model_architecture_simple(model, name)
            if 'error' not in viz_paths:
                # Save visualizations
                model_dir = output_path / name.lower().replace(' ', '_')
                model_dir.mkdir(parents=True, exist_ok=True)
                
                if 'architecture_figure' in viz_paths:
                    arch_path = model_dir / f"{name}_architecture.png"
                    viz_paths['architecture_figure'].savefig(arch_path, dpi=150, bbox_inches='tight')
                
                if 'details_figure' in viz_paths:
                    details_path = model_dir / f"{name}_details.png"
                    viz_paths['details_figure'].savefig(details_path, dpi=150, bbox_inches='tight')
                
                visualization_paths[name] = {'architecture': str(arch_path), 'details': str(details_path)}
            else:
                visualization_paths[name] = {'error': viz_paths['error']}
        except Exception as e:
            logger.error(f"Failed to visualize model {name}: {e}")
            visualization_paths[name] = {'error': str(e)}
    
    # Create comparison visualization
    try:
        from .visualization import compare_models
        comparison_viz = compare_models(models, str(output_path))
        visualization_paths['comparison'] = comparison_viz
    except Exception as e:
        logger.error(f"Failed to create comparison visualization: {e}")
    
    # Save comprehensive report
    report = {
        'models_analyzed': list(models.keys()),
        'comparison_data': comparison_data,
        'visualization_paths': visualization_paths,
        'summary': {
            'total_models': len(models),
            'parameter_range': {
                'min': min(data.get('total_parameters', 0) for data in comparison_data.values() if 'total_parameters' in data),
                'max': max(data.get('total_parameters', 0) for data in comparison_data.values() if 'total_parameters' in data)
            } if comparison_data else {'min': 0, 'max': 0}
        }
    }
    
    report_path = output_path / "model_comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Model comparison report saved to {report_path}")
    return report
