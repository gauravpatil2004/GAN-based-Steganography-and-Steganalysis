"""
GAN Architecture for Text-in-Image Steganography

Modified GAN networks to hide text messages in images instead of image-in-image.
Much more practical and faster to train.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TextSteganoGenerator(nn.Module):
    """Generator that embeds text into cover images."""
    
    def __init__(self, text_embed_dim: int = 128, image_channels: int = 3):
        super().__init__()
        
        self.text_embed_dim = text_embed_dim
        self.image_channels = image_channels
        
        # Text processing branch
        self.text_projector = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True)
        )
        
        # Image encoder (extract features from cover image)
        self.image_encoder = nn.Sequential(
            # 32x32x3 -> 16x16x64
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 4x4x256 -> 2x2x512
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Fusion layer (combine text and image features)
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2 * 2 + 1024, 512 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (generate stego image)
        self.decoder = nn.Sequential(
            # 2x2x512 -> 4x4x256
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16x64 -> 32x32x3
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        print("🎨 TextSteganoGenerator initialized")
        print(f"   Text embed dim: {text_embed_dim}")
        print(f"   Image channels: {image_channels}")
    
    def forward(self, cover_image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cover_image: [B, 3, 32, 32] Cover image
            text_embedding: [B, text_embed_dim] Text embedding
        Returns:
            stego_image: [B, 3, 32, 32] Image with hidden text
        """
        batch_size = cover_image.size(0)
        
        # Process text
        text_features = self.text_projector(text_embedding)  # [B, 1024]
        
        # Process cover image
        image_features = self.image_encoder(cover_image)  # [B, 512, 2, 2]
        image_features_flat = image_features.view(batch_size, -1)  # [B, 512*2*2]
        
        # Fuse text and image features
        combined_features = torch.cat([image_features_flat, text_features], dim=1)
        fused_features = self.fusion(combined_features)  # [B, 512*2*2]
        fused_features = fused_features.view(batch_size, 512, 2, 2)  # [B, 512, 2, 2]
        
        # Generate stego image
        stego_image = self.decoder(fused_features)  # [B, 3, 32, 32]
        
        return stego_image


class TextSteganoDiscriminator(nn.Module):
    """Discriminator to detect presence of hidden text in images."""
    
    def __init__(self, image_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # 32x32x3 -> 16x16x64
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4x256 -> 2x2x512
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # Probability of containing hidden text
        )
        
        print("🔍 TextSteganoDiscriminator initialized")
        print(f"   Image channels: {image_channels}")
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, 32, 32] Input image
        Returns:
            probability: [B, 1] Probability of containing hidden text
        """
        features = self.features(image)
        features_flat = features.view(features.size(0), -1)
        probability = self.classifier(features_flat)
        return probability


class TextExtractor(nn.Module):
    """Extract hidden text from stego images."""
    
    def __init__(self, vocab_size: int, max_text_length: int, image_channels: int = 3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length
        
        # Image feature extractor
        self.feature_extractor = nn.Sequential(
            # 32x32x3 -> 16x16x64
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 4x4x256 -> 2x2x512
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Text decoder
        self.text_decoder = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, max_text_length * (vocab_size + 3))  # +3 for special tokens
        )
        
        print("🔓 TextExtractor initialized")
        print(f"   Vocab size: {vocab_size + 3}")
        print(f"   Max text length: {max_text_length}")
    
    def forward(self, stego_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stego_image: [B, 3, 32, 32] Stego image
        Returns:
            text_logits: [B, max_length, vocab_size+3] Text predictions
        """
        batch_size = stego_image.size(0)
        
        # Extract features
        features = self.feature_extractor(stego_image)
        features_flat = features.view(batch_size, -1)
        
        # Decode text
        text_output = self.text_decoder(features_flat)
        text_logits = text_output.view(batch_size, self.max_text_length, self.vocab_size + 3)
        
        return text_logits


def test_text_stegano_networks():
    """Test the text steganography networks."""
    print("🧪 Testing Text Steganography Networks")
    print("=" * 50)
    
    # Parameters
    batch_size = 4
    text_embed_dim = 128
    vocab_size = 95  # Printable ASCII
    max_text_length = 64
    
    # Create test data
    cover_images = torch.randn(batch_size, 3, 32, 32)
    text_embeddings = torch.randn(batch_size, text_embed_dim)
    
    print(f"📊 Test setup:")
    print(f"   Batch size: {batch_size}")
    print(f"   Image shape: {cover_images.shape}")
    print(f"   Text embedding shape: {text_embeddings.shape}")
    
    # Test Generator
    print(f"\n🎨 Testing Generator...")
    generator = TextSteganoGenerator(text_embed_dim)
    stego_images = generator(cover_images, text_embeddings)
    print(f"   Input: cover {cover_images.shape} + text {text_embeddings.shape}")
    print(f"   Output: stego {stego_images.shape}")
    
    # Test Discriminator
    print(f"\n🔍 Testing Discriminator...")
    discriminator = TextSteganoDiscriminator()
    disc_real = discriminator(cover_images)
    disc_fake = discriminator(stego_images.detach())
    print(f"   Real output: {disc_real.shape}, mean: {disc_real.mean():.3f}")
    print(f"   Fake output: {disc_fake.shape}, mean: {disc_fake.mean():.3f}")
    
    # Test Extractor
    print(f"\n🔓 Testing Extractor...")
    extractor = TextExtractor(vocab_size, max_text_length)
    text_logits = extractor(stego_images)
    print(f"   Input: stego {stego_images.shape}")
    print(f"   Output: text logits {text_logits.shape}")
    
    # Test text prediction
    text_predictions = torch.argmax(text_logits, dim=-1)
    print(f"   Text predictions: {text_predictions.shape}")
    print(f"   Sample prediction: {text_predictions[0][:10].tolist()}")
    
    print(f"\n✅ All networks working correctly!")
    
    # Calculate model sizes
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Sizes:")
    print(f"   Generator: {count_parameters(generator):,} parameters")
    print(f"   Discriminator: {count_parameters(discriminator):,} parameters")
    print(f"   Extractor: {count_parameters(extractor):,} parameters")
    
    total_params = (count_parameters(generator) + 
                   count_parameters(discriminator) + 
                   count_parameters(extractor))
    print(f"   Total: {total_params:,} parameters")


if __name__ == "__main__":
    test_text_stegano_networks()
