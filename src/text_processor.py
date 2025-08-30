"""
Text Processing Pipeline for Text-in-Image Steganography

Handles text encoding, decoding, padding, and conversion to embeddings
for GAN-based steganography.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import string
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib


class TextProcessor:
    """Complete text processing for steganography."""
    
    def __init__(self, max_length: int = 256, encoding: str = 'utf-8'):
        self.max_length = max_length
        self.encoding = encoding
        
        # Character vocabulary (printable ASCII + common symbols)
        self.vocab = string.printable[:95]  # Exclude non-printable
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        
        print(f"📝 TextProcessor initialized:")
        print(f"   Max length: {max_length} characters")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Encoding: {encoding}")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices."""
        # Add start token
        sequence = [self.START_TOKEN]
        
        # Convert characters to indices
        for char in text:
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char] + 3)  # +3 for special tokens
            else:
                # Handle unknown characters as space
                sequence.append(self.char_to_idx[' '] + 3)
        
        # Add end token
        sequence.append(self.END_TOKEN)
        
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text."""
        text = ""
        for idx in sequence:
            if idx == self.START_TOKEN:
                continue
            elif idx == self.END_TOKEN:
                break
            elif idx == self.PAD_TOKEN:
                continue
            else:
                char_idx = idx - 3  # Adjust for special tokens
                if 0 <= char_idx < self.vocab_size:
                    text += self.vocab[char_idx]
        
        return text
    
    def pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad or truncate sequence to max_length."""
        if len(sequence) > self.max_length:
            # Truncate but keep end token
            sequence = sequence[:self.max_length-1] + [self.END_TOKEN]
        else:
            # Pad with PAD_TOKEN
            sequence = sequence + [self.PAD_TOKEN] * (self.max_length - len(sequence))
        
        return sequence
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        sequence = self.text_to_sequence(text)
        padded_sequence = self.pad_sequence(sequence)
        return torch.tensor(padded_sequence, dtype=torch.long)
    
    def decode_text(self, tensor: torch.Tensor) -> str:
        """Decode tensor back to text."""
        if tensor.dim() > 1:
            tensor = tensor.squeeze()
        sequence = tensor.cpu().tolist()
        return self.sequence_to_text(sequence)
    
    def text_to_binary(self, text: str) -> str:
        """Convert text to binary string."""
        return ''.join(format(ord(char), '08b') for char in text)
    
    def binary_to_text(self, binary: str) -> str:
        """Convert binary string back to text."""
        chars = []
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        return ''.join(chars)
    
    def get_capacity(self, image_size: Tuple[int, int, int]) -> int:
        """Calculate text capacity for given image size."""
        h, w, c = image_size
        total_pixels = h * w * c
        # Use 1 bit per pixel for text (conservative estimate)
        bits_available = total_pixels
        chars_available = bits_available // 8  # 8 bits per character
        return min(chars_available, self.max_length)


class TextEncryption:
    """Text encryption for secure steganography."""
    
    def __init__(self):
        self.key = None
    
    def generate_key_from_password(self, password: str, salt: bytes = None) -> bytes:
        """Generate encryption key from password."""
        if salt is None:
            salt = b"steganography_salt_2024"  # Fixed salt for reproducibility
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_text(self, text: str, password: str) -> bytes:
        """Encrypt text with password."""
        key = self.generate_key_from_password(password)
        f = Fernet(key)
        encrypted = f.encrypt(text.encode())
        return encrypted
    
    def decrypt_text(self, encrypted_data: bytes, password: str) -> str:
        """Decrypt text with password."""
        try:
            key = self.generate_key_from_password(password)
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_data)
            return decrypted.decode()
        except:
            return ""  # Return empty string if decryption fails


class TextEmbedding(nn.Module):
    """Neural text embedding for GAN integration."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size + 3, embed_dim)  # +3 for special tokens
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, embed_dim)
        
        print(f"🧠 TextEmbedding initialized:")
        print(f"   Vocab size: {vocab_size + 3}")
        print(f"   Embed dim: {embed_dim}")
        print(f"   Hidden dim: {hidden_dim}")
    
    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_indices: [batch_size, seq_len]
        Returns:
            text_embedding: [batch_size, embed_dim]
        """
        # Embed characters
        embedded = self.embedding(text_indices)  # [B, L, E]
        
        # Process with LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # [B, L, H*2]
        
        # Use final hidden state (concatenated bidirectional)
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, H*2]
        
        # Project to final embedding
        text_embed = self.output_proj(final_hidden)  # [B, E]
        
        return text_embed


# Utility functions
def calculate_text_metrics(original_text: str, recovered_text: str) -> dict:
    """Calculate text recovery metrics."""
    
    # Character Error Rate
    char_errors = sum(c1 != c2 for c1, c2 in zip(original_text, recovered_text))
    char_errors += abs(len(original_text) - len(recovered_text))
    cer = char_errors / max(len(original_text), 1)
    
    # Word Error Rate
    orig_words = original_text.split()
    rec_words = recovered_text.split()
    word_errors = sum(w1 != w2 for w1, w2 in zip(orig_words, rec_words))
    word_errors += abs(len(orig_words) - len(rec_words))
    wer = word_errors / max(len(orig_words), 1)
    
    # Exact match
    exact_match = original_text == recovered_text
    
    # Length accuracy
    length_accuracy = 1.0 - abs(len(original_text) - len(recovered_text)) / max(len(original_text), 1)
    
    return {
        'character_error_rate': cer,
        'word_error_rate': wer,
        'character_accuracy': 1.0 - cer,
        'word_accuracy': 1.0 - wer,
        'exact_match': exact_match,
        'length_accuracy': length_accuracy
    }


def test_text_processor():
    """Test the text processing pipeline."""
    print("🧪 Testing Text Processing Pipeline")
    print("=" * 50)
    
    # Initialize processor
    processor = TextProcessor(max_length=128)
    
    # Test texts
    test_texts = [
        "Hello World!",
        "This is a secret message for steganography.",
        "Password: mySecret123!",
        "https://example.com/secret-url",
        "123456789 numeric test",
        "Special chars: @#$%^&*()"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Original: '{text}'")
        
        # Encode and decode
        encoded = processor.encode_text(text)
        decoded = processor.decode_text(encoded)
        
        print(f"   Encoded shape: {encoded.shape}")
        print(f"   Decoded: '{decoded}'")
        
        # Calculate metrics
        metrics = calculate_text_metrics(text, decoded)
        print(f"   Character accuracy: {metrics['character_accuracy']:.3f}")
        print(f"   Exact match: {metrics['exact_match']}")
    
    # Test capacity calculation
    capacity = processor.get_capacity((32, 32, 3))  # CIFAR-10 size
    print(f"\n📊 Text capacity for CIFAR-10 (32x32x3): {capacity} characters")
    
    # Test encryption
    print(f"\n🔒 Testing Encryption:")
    encryption = TextEncryption()
    test_text = "Secret message for encryption test"
    password = "mypassword123"
    
    encrypted = encryption.encrypt_text(test_text, password)
    decrypted = encryption.decrypt_text(encrypted, password)
    
    print(f"   Original: '{test_text}'")
    print(f"   Encrypted length: {len(encrypted)} bytes")
    print(f"   Decrypted: '{decrypted}'")
    print(f"   Match: {test_text == decrypted}")


if __name__ == "__main__":
    test_text_processor()
