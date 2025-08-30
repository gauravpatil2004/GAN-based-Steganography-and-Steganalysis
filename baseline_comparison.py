#!/usr/bin/env python3
"""
LSB (Least Significant Bit) Steganography Baseline
For comparison with GAN-based text steganography
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

# Add src directory to path
sys.path.append('./src')
from text_processor import TextProcessor

class LSBSteganography:
    """Simple LSB steganography for baseline comparison."""
    
    def __init__(self):
        """Initialize LSB steganography."""
        self.text_processor = TextProcessor(max_length=128)
        print("📝 LSB Steganography initialized")
        
    def text_to_binary(self, text: str) -> str:
        """Convert text to binary string."""
        # Encode text using the same processor as GAN
        encoded = self.text_processor.encode_text(text)
        
        # Convert to binary string
        binary_str = ""
        for idx in encoded:
            # Convert each character index to 8-bit binary
            binary_str += format(idx.item(), '08b')
        
        # Add delimiter to mark end of message
        binary_str += "1111111111111110"  # End marker
        return binary_str
    
    def binary_to_text(self, binary_str: str) -> str:
        """Convert binary string back to text."""
        # Find end marker
        end_marker = "1111111111111110"
        end_pos = binary_str.find(end_marker)
        if end_pos != -1:
            binary_str = binary_str[:end_pos]
        
        # Convert binary to character indices
        indices = []
        for i in range(0, len(binary_str), 8):
            if i + 8 <= len(binary_str):
                byte = binary_str[i:i+8]
                indices.append(int(byte, 2))
        
        # Convert indices back to text
        if indices:
            tensor_indices = torch.tensor(indices)
            return self.text_processor.decode_text(tensor_indices)
        return ""
    
    def hide_text_in_image(self, image: np.ndarray, text: str) -> Tuple[np.ndarray, bool]:
        """Hide text in image using LSB method."""
        # Convert text to binary
        binary_message = self.text_to_binary(text)
        
        # Check if image can hold the message
        image_flat = image.flatten()
        if len(binary_message) > len(image_flat):
            print(f"❌ Message too long for image capacity")
            return image, False
        
        # Create copy of image
        stego_image = image.copy().flatten()
        
        # Hide message in LSBs
        for i, bit in enumerate(binary_message):
            # Modify LSB of pixel value
            stego_image[i] = (stego_image[i] & 0xFE) | int(bit)
        
        # Reshape back to original dimensions
        stego_image = stego_image.reshape(image.shape)
        
        return stego_image, True
    
    def extract_text_from_image(self, stego_image: np.ndarray, max_bits: int = 8192) -> str:
        """Extract hidden text from stego image."""
        # Extract LSBs
        stego_flat = stego_image.flatten()
        binary_message = ""
        
        # Extract bits until we hit max_bits or find end marker
        for i in range(min(max_bits, len(stego_flat))):
            binary_message += str(stego_flat[i] & 1)
            
            # Check for end marker every 16 bits
            if len(binary_message) >= 16 and binary_message[-16:] == "1111111111111110":
                break
        
        # Convert binary back to text
        return self.binary_to_text(binary_message)
    
    def calculate_psnr(self, original: np.ndarray, stego: np.ndarray) -> float:
        """Calculate PSNR between original and stego images."""
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr


class BaselineComparison:
    """Compare GAN steganography with LSB baseline."""
    
    def __init__(self):
        """Initialize comparison framework."""
        self.lsb = LSBSteganography()
        self.test_sentences = [
            "password123",
            "https://secret-site.com/login",
            "GPS: 40.7128, -74.0060",
            "API_KEY=abc123xyz789",
            "Transfer $1000 to account 456789",
            "Meeting at 3PM Room 205",
            "admin:StrongPass2024!",
            "wallet:1A1zP1eP5QGefi2D",
            "Emergency: +1-555-0123",
            "mysql://user:pass@localhost"
        ]
        print("🔍 Baseline Comparison initialized")
        
    def evaluate_lsb_performance(self) -> dict:
        """Evaluate LSB steganography performance."""
        print("📊 Evaluating LSB Steganography Performance...")
        
        results = []
        total_char_acc = 0
        total_word_acc = 0
        total_psnr = 0
        
        for i, text in enumerate(self.test_sentences):
            print(f"\n🧪 LSB Test {i+1}: '{text}'")
            
            # Create random cover image (32x32 RGB like CIFAR-10)
            cover_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            
            # Hide text
            stego_image, success = self.lsb.hide_text_in_image(cover_image, text)
            
            if not success:
                print(f"   ❌ Failed to hide text")
                continue
                
            # Extract text
            extracted_text = self.lsb.extract_text_from_image(stego_image)
            
            # Calculate metrics
            char_acc = self._calculate_character_accuracy(text, extracted_text)
            word_acc = 1.0 if text.strip() == extracted_text.strip() else 0.0
            psnr = self.lsb.calculate_psnr(cover_image, stego_image)
            
            result = {
                'original_text': text,
                'extracted_text': extracted_text,
                'character_accuracy': char_acc,
                'word_accuracy': word_acc,
                'psnr': psnr,
                'success': success
            }
            
            results.append(result)
            total_char_acc += char_acc
            total_word_acc += word_acc
            total_psnr += psnr
            
            print(f"   Original:  '{text}'")
            print(f"   Extracted: '{extracted_text}'")
            print(f"   Char Acc:  {char_acc:.3f} ({char_acc*100:.1f}%)")
            print(f"   Word Acc:  {word_acc:.3f}")
            print(f"   PSNR:      {psnr:.2f} dB")
        
        # Calculate averages
        num_successful = len(results)
        avg_results = {
            'average_character_accuracy': total_char_acc / num_successful if num_successful > 0 else 0,
            'average_word_accuracy': total_word_acc / num_successful if num_successful > 0 else 0,
            'average_psnr': total_psnr / num_successful if num_successful > 0 else 0,
            'success_rate': num_successful / len(self.test_sentences),
            'total_tests': len(self.test_sentences),
            'successful_tests': num_successful,
            'results': results
        }
        
        print(f"\n📊 LSB Baseline Results:")
        print(f"   Success Rate: {avg_results['success_rate']*100:.1f}%")
        print(f"   Character Accuracy: {avg_results['average_character_accuracy']*100:.1f}%")
        print(f"   Word Accuracy: {avg_results['average_word_accuracy']*100:.1f}%")
        print(f"   Average PSNR: {avg_results['average_psnr']:.2f} dB")
        
        return avg_results
    
    def _calculate_character_accuracy(self, original: str, extracted: str) -> float:
        """Calculate character-level accuracy."""
        if not original:
            return 1.0 if not extracted else 0.0
            
        # Pad shorter string
        max_len = max(len(original), len(extracted))
        orig_padded = original.ljust(max_len)
        extr_padded = extracted.ljust(max_len)
        
        # Calculate accuracy
        correct = sum(1 for o, e in zip(orig_padded, extr_padded) if o == e)
        return correct / max_len
    
    def compare_with_gan_results(self, gan_results: dict, lsb_results: dict):
        """Compare GAN and LSB steganography results."""
        print("\n🏆 GAN vs LSB COMPARISON")
        print("=" * 50)
        
        # Extract metrics
        gan_char_acc = gan_results.get('average_character_accuracy', 0) * 100
        lsb_char_acc = lsb_results.get('average_character_accuracy', 0) * 100
        
        gan_word_acc = gan_results.get('average_word_accuracy', 0) * 100
        lsb_word_acc = lsb_results.get('average_word_accuracy', 0) * 100
        
        gan_psnr = gan_results.get('average_psnr', 0)
        lsb_psnr = lsb_results.get('average_psnr', 0)
        
        print(f"📊 Character Accuracy:")
        print(f"   GAN Model: {gan_char_acc:.1f}%")
        print(f"   LSB Baseline: {lsb_char_acc:.1f}%")
        print(f"   Winner: {'🏆 GAN' if gan_char_acc > lsb_char_acc else '🏆 LSB' if lsb_char_acc > gan_char_acc else '🤝 TIE'}")
        
        print(f"\n📊 Word Accuracy:")
        print(f"   GAN Model: {gan_word_acc:.1f}%")
        print(f"   LSB Baseline: {lsb_word_acc:.1f}%")
        print(f"   Winner: {'🏆 GAN' if gan_word_acc > lsb_word_acc else '🏆 LSB' if lsb_word_acc > gan_word_acc else '🤝 TIE'}")
        
        print(f"\n📊 Image Quality (PSNR):")
        print(f"   GAN Model: {gan_psnr:.2f} dB")
        print(f"   LSB Baseline: {lsb_psnr:.2f} dB")
        print(f"   Winner: {'🏆 GAN' if gan_psnr > lsb_psnr else '🏆 LSB' if lsb_psnr > gan_psnr else '🤝 TIE'}")
        
        # Overall assessment
        gan_wins = sum([
            gan_char_acc > lsb_char_acc,
            gan_word_acc > lsb_word_acc,
            gan_psnr > lsb_psnr
        ])
        
        print(f"\n🎯 Overall Winner: ", end="")
        if gan_wins >= 2:
            print("🏆 GAN MODEL")
            print("   The GAN-based approach outperforms LSB baseline!")
        elif gan_wins == 1:
            print("🤝 MIXED RESULTS")
            print("   Both methods have strengths in different areas.")
        else:
            print("🏆 LSB BASELINE")
            print("   The traditional LSB method performed better.")
        
        return {
            'gan_wins': gan_wins,
            'comparison_summary': {
                'character_accuracy': {'gan': gan_char_acc, 'lsb': lsb_char_acc},
                'word_accuracy': {'gan': gan_word_acc, 'lsb': lsb_word_acc},
                'psnr': {'gan': gan_psnr, 'lsb': lsb_psnr}
            }
        }


def run_baseline_comparison():
    """Run complete baseline comparison."""
    print("🔍 LSB BASELINE COMPARISON")
    print("=" * 50)
    
    # Initialize comparison
    comparison = BaselineComparison()
    
    # Evaluate LSB performance
    lsb_results = comparison.evaluate_lsb_performance()
    
    # Mock GAN results (would come from actual evaluation)
    # Using the known results from training: 88.3% character accuracy
    gan_results = {
        'average_character_accuracy': 0.883,  # 88.3% from training
        'average_word_accuracy': 0.0,        # 0% word accuracy from training
        'average_psnr': 11.92,               # 11.92 dB from training
        'total_tests': 10
    }
    
    print(f"\n📋 Using GAN Results from Training:")
    print(f"   Character Accuracy: {gan_results['average_character_accuracy']*100:.1f}%")
    print(f"   Word Accuracy: {gan_results['average_word_accuracy']*100:.1f}%")
    print(f"   PSNR: {gan_results['average_psnr']:.2f} dB")
    
    # Compare results
    comparison_results = comparison.compare_with_gan_results(gan_results, lsb_results)
    
    # Save results
    import json
    os.makedirs('evaluation_results', exist_ok=True)
    
    all_results = {
        'lsb_results': lsb_results,
        'gan_results': gan_results,
        'comparison': comparison_results
    }
    
    with open('evaluation_results/baseline_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: evaluation_results/baseline_comparison.json")
    
    return all_results


if __name__ == "__main__":
    run_baseline_comparison()
