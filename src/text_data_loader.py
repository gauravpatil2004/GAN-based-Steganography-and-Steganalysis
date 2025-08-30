"""
Data Loader for Text-in-Image Steganography

Creates paired datasets of images and text messages for training
text steganography models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
from typing import List, Tuple, Optional
import string
import os

from text_processor import TextProcessor, TextEmbedding


class TextImageDataset(Dataset):
    """Dataset pairing images with text messages for steganography."""
    
    def __init__(self, 
                 image_dataset,
                 text_processor: TextProcessor,
                 text_corpus: List[str] = None,
                 transform=None):
        self.image_dataset = image_dataset
        self.text_processor = text_processor
        self.transform = transform
        
        # Default text corpus if none provided
        self.text_corpus = text_corpus or self._generate_default_corpus()
        
        print(f"📊 TextImageDataset initialized:")
        print(f"   Images: {len(self.image_dataset)}")
        print(f"   Text corpus: {len(self.text_corpus)} messages")
        print(f"   Max text length: {self.text_processor.max_length}")
    
    def _generate_default_corpus(self) -> List[str]:
        """Generate diverse text corpus for training."""
        
        # Common message types for steganography
        corpus = []
        
        # 1. Passwords and keys
        passwords = [
            "mySecretPassword123!",
            "admin@2024#secure",
            "P@ssw0rd!Strong",
            "SecureKey2024$",
            "HiddenMessage123",
            "StealthMode!2024",
            "CovertOps@123",
            "SilentComm#456"
        ]
        corpus.extend(passwords)
        
        # 2. URLs and addresses
        urls = [
            "https://secret.example.com/path",
            "ftp://hidden.server.org/files",
            "http://covert.communication.net",
            "https://secure.channel.io/api",
            "mailto:secret@hidden.domain",
            "https://bit.ly/secretlink123",
            "sftp://backup.secure.com/data",
            "https://encrypted.vault.org"
        ]
        corpus.extend(urls)
        
        # 3. Coordinates and locations
        coordinates = [
            "GPS: 40.7128N, 74.0060W",
            "Location: Safe House Alpha",
            "Coordinates: 51.5074N, 0.1278W",
            "Meeting Point: Central Park",
            "Rendezvous: Pier 39, SF",
            "Drop Zone: Grid 123-456",
            "Pickup: Terminal 5, Gate B12",
            "Base: Underground Level 3"
        ]
        corpus.extend(coordinates)
        
        # 4. Encrypted-like strings
        encrypted = [
            "AES256:k3yH4sh!nG$ecur3",
            "RSA:Publ1cK3y#Encr7pt",
            "SHA256:H4sh!nGAlg0r1thm",
            "Base64:VGVzdE1lc3NhZ2U=",
            "HEX:48656C6C6F576F726C64",
            "MD5:5d41402abc4b2a76b9719",
            "JWT:eyJhbGciOiJIUzI1NiIs",
            "UUID:550e8400-e29b-41d4"
        ]
        corpus.extend(encrypted)
        
        # 5. Code names and operations
        codenames = [
            "Operation Blackbird active",
            "Asset Delta compromised",
            "Mission Phoenix successful",
            "Code Red initiated at 0300",
            "Package delivered to Eagle",
            "Abort sequence Zulu-7",
            "Extraction point Charlie",
            "Protocol Omega in effect"
        ]
        corpus.extend(codenames)
        
        # 6. Technical information
        technical = [
            "Server IP: 192.168.1.100",
            "Port: 8080, Protocol: HTTPS",
            "Database: users@localhost",
            "API Key: sk-1234567890abcdef",
            "Token: Bearer xyz789abc",
            "Version: 2.1.4-beta",
            "Build: 20240815-1430",
            "Checksum: CRC32-4A7B9C2D"
        ]
        corpus.extend(technical)
        
        # 7. Short messages
        short_messages = [
            "Message received",
            "All clear",
            "Stand by",
            "Mission abort",
            "Package secure",
            "Ready for pickup",
            "Target acquired",
            "Extraction needed"
        ]
        corpus.extend(short_messages)
        
        # 8. Longer narrative messages
        narratives = [
            "The package has been delivered to the specified location. Await further instructions via secure channel.",
            "Intelligence suggests the meeting has been compromised. Recommend switching to backup protocol immediately.",
            "Financial records indicate unusual activity in account 4571. Investigate source of transactions.",
            "Surveillance equipment installed successfully. Begin monitoring phase at 0600 hours tomorrow.",
            "Communication blackout in sector 7. All units switch to emergency frequency until further notice."
        ]
        corpus.extend(narratives)
        
        # 9. JSON-like structured data
        structured = [
            '{"user":"admin","pass":"secret"}',
            '{"lat":40.7128,"lng":-74.0060}',
            '{"status":"active","code":200}',
            '{"key":"value","encrypted":true}',
            '{"id":12345,"token":"abc123"}',
            '{"alert":"high","level":3}',
            '{"time":"23:59","zone":"UTC"}',
            '{"auth":"bearer","valid":true}'
        ]
        corpus.extend(structured)
        
        # 10. Alphanumeric codes
        codes = [
            "ID-7734-ALPHA-9982",
            "REF:2024-08-15-001",
            "CTRL-ALT-DEL-7734",
            "X1Y2Z3-A4B5C6-D7E8F9",
            "BATCH-4571-SECURE",
            "SERIAL:ABC123XYZ789",
            "MODEL:ST-2024-PRO",
            "LICENSE:GPL-3.0-2024"
        ]
        corpus.extend(codes)
        
        # Shuffle the corpus
        random.shuffle(corpus)
        return corpus
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Cover image tensor
            text_encoded: Encoded text tensor
        """
        # Get image
        image, _ = self.image_dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Get random text from corpus
        text = random.choice(self.text_corpus)
        
        # Encode text
        text_encoded = self.text_processor.encode_text(text)
        
        return image, text_encoded


def create_text_data_loader(batch_size: int = 16, 
                           data_path: str = './data',
                           max_text_length: int = 128,
                           num_workers: int = 0) -> Tuple[DataLoader, TextProcessor, TextEmbedding]:
    """
    Create data loader for text steganography training.
    
    Returns:
        DataLoader, TextProcessor, TextEmbedding
    """
    
    # Initialize text processor
    text_processor = TextProcessor(max_length=max_text_length)
    
    # Initialize text embedding network
    text_embedding = TextEmbedding(
        vocab_size=text_processor.vocab_size,
        embed_dim=128,
        hidden_dim=256
    )
    
    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Note: Normalization to [-1, 1] will be done in training loop
    ])
    
    # Load CIFAR-10 dataset
    cifar_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    # Create text-image dataset
    text_image_dataset = TextImageDataset(
        image_dataset=cifar_dataset,
        text_processor=text_processor,
        transform=None  # Already applied in CIFAR-10
    )
    
    # Custom collate function for variable text lengths
    def collate_fn(batch):
        images, texts = zip(*batch)
        
        # Stack images
        images = torch.stack(images)
        
        # Stack text tensors (already padded by TextProcessor)
        texts = torch.stack(texts)
        
        return images, texts
    
    # Create data loader
    dataloader = DataLoader(
        text_image_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ Text data loader created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Dataset size: {len(text_image_dataset)}")
    print(f"   Batches per epoch: {len(dataloader)}")
    
    return dataloader, text_processor, text_embedding


def create_custom_text_corpus(file_path: str) -> List[str]:
    """Load custom text corpus from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"📝 Loaded {len(texts)} texts from {file_path}")
        return texts
    
    except FileNotFoundError:
        print(f"⚠️ File {file_path} not found. Using default corpus.")
        return None


def test_text_data_loader():
    """Test the text data loader."""
    print("🧪 Testing Text Data Loader")
    print("=" * 40)
    
    # Create data loader
    dataloader, text_processor, text_embedding = create_text_data_loader(
        batch_size=4,
        max_text_length=64
    )
    
    # Test one batch
    print(f"\n📊 Testing batch processing...")
    for batch_idx, (images, texts) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"   Images shape: {images.shape}")
        print(f"   Texts shape: {texts.shape}")
        
        # Test text embedding
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        text_embedding = text_embedding.to(device)
        texts = texts.to(device)
        
        text_embeddings = text_embedding(texts)
        print(f"   Text embeddings shape: {text_embeddings.shape}")
        
        # Decode some texts
        for i in range(min(2, len(texts))):
            decoded_text = text_processor.decode_text(texts[i])
            print(f"   Sample text {i+1}: '{decoded_text}'")
        
        if batch_idx >= 2:  # Test only first 3 batches
            break
    
    print(f"\n✅ Text data loader working correctly!")


if __name__ == "__main__":
    test_text_data_loader()
