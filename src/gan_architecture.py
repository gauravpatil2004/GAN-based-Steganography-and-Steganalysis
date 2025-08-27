"""
Day 4: GAN Architecture Design for Steganography

Goal: Design and implement Generator and Discriminator networks that can:
1. Hide secret images with high quality (>30 dB PSNR)
2. Achieve high capacity (>50% hiding)
3. Be undetectable by steganalysis tools

This will overcome the limitations we found in traditional LSB methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SteganoGenerator(nn.Module):
    """
    Generator network for steganography.
    Takes cover image + secret image and produces stego image.
    """
    
    def __init__(self, input_channels=6, output_channels=3):
        """
        Args:
            input_channels (int): 6 (3 for cover + 3 for secret)
            output_channels (int): 3 (RGB stego image)
        """
        super(SteganoGenerator, self).__init__()
        
        # Encoder: Extract features from cover+secret
        self.encoder = nn.Sequential(
            # Input: 6 x 32 x 32
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 16 x 16
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 8 x 8
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 4 x 4
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 x 1 x 1 (bottleneck)
        )
        
        # Decoder: Generate stego image
        self.decoder = nn.Sequential(
            # 512 x 1 x 1
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 4 x 4
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 8 x 8
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 16 x 16
            
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
            # 3 x 32 x 32
        )
        
        # Skip connections for better quality preservation
        self.skip_conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(256, 256, kernel_size=1)
    
    def forward(self, cover, secret):
        """
        Args:
            cover (torch.Tensor): Cover image [B, 3, H, W]
            secret (torch.Tensor): Secret image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Stego image [B, 3, H, W]
        """
        # Concatenate cover and secret
        x = torch.cat([cover, secret], dim=1)  # [B, 6, H, W]
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode to generate stego image
        stego = self.decoder(encoded)
        
        return stego


class SteganoDiscriminator(nn.Module):
    """
    Discriminator network for steganography detection.
    Determines if an image contains hidden content (steganalysis).
    """
    
    def __init__(self, input_channels=3):
        """
        Args:
            input_channels (int): 3 (RGB image)
        """
        super(SteganoDiscriminator, self).__init__()
        
        self.features = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 16 x 16
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 2 x 2
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 512 x 1 x 1
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Probability of being stego [B, 1]
        """
        features = self.features(x)
        output = self.classifier(features)
        return output


class SecretExtractor(nn.Module):
    """
    Network to extract hidden secret from stego image.
    This ensures the secret can be recovered accurately.
    """
    
    def __init__(self, input_channels=3, output_channels=3):
        """
        Args:
            input_channels (int): 3 (RGB stego image)
            output_channels (int): 3 (RGB secret image)
        """
        super(SecretExtractor, self).__init__()
        
        self.extractor = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Bottleneck
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Reconstruction
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, stego):
        """
        Args:
            stego (torch.Tensor): Stego image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Extracted secret [B, 3, H, W]
        """
        return self.extractor(stego)


def test_networks():
    """Test the network architectures."""
    print("🧪 Testing GAN Architecture...")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create networks
    generator = SteganoGenerator().to(device)
    discriminator = SteganoDiscriminator().to(device)
    extractor = SecretExtractor().to(device)
    
    # Test input
    batch_size = 4
    cover = torch.randn(batch_size, 3, 32, 32).to(device)
    secret = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print(f"Input shapes:")
    print(f"  Cover: {cover.shape}")
    print(f"  Secret: {secret.shape}")
    
    # Test Generator
    print(f"\n📱 Testing Generator...")
    stego = generator(cover, secret)
    print(f"  Stego output shape: {stego.shape}")
    print(f"  ✅ Generator working!")
    
    # Test Discriminator
    print(f"\n🔍 Testing Discriminator...")
    disc_real = discriminator(cover)
    disc_fake = discriminator(stego)
    print(f"  Real image prediction: {disc_real.shape}")
    print(f"  Stego image prediction: {disc_fake.shape}")
    print(f"  ✅ Discriminator working!")
    
    # Test Extractor
    print(f"\n🔓 Testing Secret Extractor...")
    extracted = extractor(stego)
    print(f"  Extracted secret shape: {extracted.shape}")
    print(f"  ✅ Extractor working!")
    
    # Parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Network Statistics:")
    print(f"  Generator parameters: {count_parameters(generator):,}")
    print(f"  Discriminator parameters: {count_parameters(discriminator):,}")
    print(f"  Extractor parameters: {count_parameters(extractor):,}")
    print(f"  Total parameters: {count_parameters(generator) + count_parameters(discriminator) + count_parameters(extractor):,}")
    
    print(f"\n🎉 All networks tested successfully!")
    return generator, discriminator, extractor


if __name__ == "__main__":
    print("🚀 DAY 4: GAN ARCHITECTURE DESIGN")
    print("=" * 60)
    print("Goal: Design networks for high-quality steganography")
    print("=" * 60)
    
    # Test the architectures
    generator, discriminator, extractor = test_networks()
    
    print(f"\n💡 Architecture Summary:")
    print(f"1. 📱 Generator: Combines cover + secret → high-quality stego")
    print(f"2. 🔍 Discriminator: Detects if image contains hidden content")
    print(f"3. 🔓 Extractor: Recovers secret from stego image")
    
    print(f"\n🎯 Key Features:")
    print(f"  • Encoder-decoder architecture for quality preservation")
    print(f"  • Adversarial training for undetectable hiding")
    print(f"  • Dedicated extractor for accurate secret recovery")
    print(f"  • Optimized for CIFAR-10 (32x32) images")
    
    print(f"\n🚀 Next Steps:")
    print(f"  • Implement loss functions")
    print(f"  • Create training loop")
    print(f"  • Test on CIFAR-10 dataset")
    print(f"  • Compare with LSB baselines")
