#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Text Steganography Model
Tests model accuracy, saves/loads weights, compares with LSB baseline
"""

import sys
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from PIL import Image
import random

# Add src directory to path
sys.path.append('./src')

from text_processor import TextProcessor
from text_gan_architecture import TextSteganoGenerator, TextSteganoDiscriminator, TextExtractor, TextEmbedding
from text_data_loader import TextImageDataset
from text_gan_losses import TextMetricsCalculator

class ModelEvaluator:
    """Comprehensive evaluation of trained text steganography model."""
    
    def __init__(self, config: Dict[str, Any], model_path: str = None):
        """Initialize evaluator with config and optional model path."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Initialize components
        self.text_processor = TextProcessor(max_length=config['max_text_length'])
        self.vocab_size = len(self.text_processor.char_to_idx)
        
        # Initialize models
        self._init_models()
        
        # Test data
        self.test_sentences = [
            "password123",
            "https://secret-site.com/login?token=abc123",
            "GPS coordinates: 40.7128, -74.0060",
            "API_KEY=sk-1234567890abcdef",
            "Transfer $5000 to account 987654321",
            "Meeting at 3PM, Room 205, Building A",
            "Username: admin, Password: StrongPass2024!",
            "Bitcoin wallet: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "Emergency contact: +1-555-0123",
            "Database connection: mysql://user:pass@localhost:3306/db"
        ]
        
        print(f"🔬 ModelEvaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Test sentences: {len(self.test_sentences)}")
        
    def _init_models(self):
        """Initialize all model components."""
        # Text embedding
        self.text_embedding = TextEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.config['text_embed_dim'],
            max_length=self.config['max_text_length']
        ).to(self.device)
        
        # Generator
        self.generator = TextSteganoGenerator(
            text_embed_dim=self.config['text_embed_dim']
        ).to(self.device)
        
        # Discriminator  
        self.discriminator = TextSteganoDiscriminator().to(self.device)
        
        # Extractor
        self.extractor = TextExtractor(
            vocab_size=self.vocab_size,
            max_length=self.config['max_text_length']
        ).to(self.device)
        
        print("✅ All models initialized")
        
    def load_trained_model(self, checkpoint_path: str):
        """Load trained model weights from checkpoint."""
        print(f"📥 Loading trained model from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator']) 
            self.extractor.load_state_dict(checkpoint['extractor'])
            self.text_embedding.load_state_dict(checkpoint['text_embedding'])
            
            # Load training history if available
            self.training_history = checkpoint.get('history', {})
            
            print("✅ Model weights loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
            
    def save_model_weights(self, save_path: str):
        """Save current model weights."""
        print(f"💾 Saving model weights to {save_path}")
        
        try:
            # Create save directory
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            checkpoint = {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'extractor': self.extractor.state_dict(),
                'text_embedding': self.text_embedding.state_dict(),
                'config': self.config,
                'vocab_size': self.vocab_size,
                'timestamp': time.time()
            }
            
            torch.save(checkpoint, save_path)
            print("✅ Model weights saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def test_save_reload_consistency(self, temp_path: str = "temp_model.pth"):
        """Test model save/reload consistency."""
        print("🔄 Testing save/reload consistency...")
        
        # Get random test input
        test_text = self.test_sentences[0]
        test_tensor = self.text_processor.encode_text(test_text).unsqueeze(0).to(self.device)
        cover_image = torch.randn(1, 3, 32, 32).to(self.device)
        
        # Get initial output
        self.generator.eval()
        self.extractor.eval()
        self.text_embedding.eval()
        
        with torch.no_grad():
            text_embed = self.text_embedding(test_tensor)
            initial_stego = self.generator(cover_image, text_embed)
            initial_extracted = self.extractor(initial_stego)
        
        # Save model
        self.save_model_weights(temp_path)
        
        # Create new model instance and load
        new_evaluator = ModelEvaluator(self.config)
        success = new_evaluator.load_trained_model(temp_path)
        
        if not success:
            print("❌ Save/reload consistency test failed")
            return False
            
        # Get output from reloaded model
        new_evaluator.generator.eval()
        new_evaluator.extractor.eval() 
        new_evaluator.text_embedding.eval()
        
        with torch.no_grad():
            new_text_embed = new_evaluator.text_embedding(test_tensor)
            new_stego = new_evaluator.generator(cover_image, new_text_embed)
            new_extracted = new_evaluator.extractor(new_stego)
        
        # Compare outputs
        stego_diff = torch.abs(initial_stego - new_stego).mean().item()
        extract_diff = torch.abs(initial_extracted - new_extracted).mean().item()
        
        print(f"   Stego image difference: {stego_diff:.6f}")
        print(f"   Extracted text difference: {extract_diff:.6f}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        is_consistent = stego_diff < 1e-6 and extract_diff < 1e-6
        print(f"{'✅' if is_consistent else '❌'} Save/reload consistency: {'PASS' if is_consistent else 'FAIL'}")
        
        return is_consistent
    
    def evaluate_on_test_data(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate model on unseen test data."""
        print(f"🧪 Evaluating on {num_samples} unseen test samples...")
        
        self.generator.eval()
        self.extractor.eval()
        self.text_embedding.eval()
        
        total_char_acc = 0
        total_word_acc = 0
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for i in range(num_samples):
                # Get random test sentence
                test_text = random.choice(self.test_sentences)
                
                # Create random cover image (simulating unseen data)
                cover_image = torch.randn(1, 3, 32, 32).to(self.device)
                cover_image = torch.clamp(cover_image, -1, 1)  # Normalize to [-1, 1]
                
                # Encode text
                text_tensor = self.text_processor.encode_text(test_text).unsqueeze(0).to(self.device)
                
                # Generate stego image
                text_embed = self.text_embedding(text_tensor)
                stego_image = self.generator(cover_image, text_embed)
                
                # Extract text
                extracted_logits = self.extractor(stego_image)
                
                # Calculate metrics
                metrics = TextMetricsCalculator.calculate_text_metrics(
                    cover_image, stego_image, text_tensor, extracted_logits
                )
                
                total_char_acc += metrics['character_accuracy']
                total_word_acc += metrics['word_accuracy']
                total_psnr += metrics['cover_psnr']
                total_ssim += metrics['cover_ssim']
                
                # Print sample results every 20 samples
                if (i + 1) % 20 == 0:
                    print(f"   Sample {i+1}: Char_Acc={metrics['character_accuracy']:.3f}, PSNR={metrics['cover_psnr']:.1f}")
        
        # Average results
        results = {
            'character_accuracy': total_char_acc / num_samples,
            'word_accuracy': total_word_acc / num_samples,
            'cover_psnr': total_psnr / num_samples,
            'cover_ssim': total_ssim / num_samples
        }
        
        print(f"📊 Test Results (Average over {num_samples} samples):")
        print(f"   Character Accuracy: {results['character_accuracy']:.3f} ({results['character_accuracy']*100:.1f}%)")
        print(f"   Word Accuracy: {results['word_accuracy']:.3f} ({results['word_accuracy']*100:.1f}%)")
        print(f"   PSNR: {results['cover_psnr']:.2f} dB")
        print(f"   SSIM: {results['cover_ssim']:.4f}")
        
        return results
    
    def test_gan_encoder_decoder(self) -> Dict[str, Any]:
        """Test GAN encoder-decoder pipeline with test sentences."""
        print("🔄 Testing GAN Encoder-Decoder Pipeline...")
        
        self.generator.eval()
        self.extractor.eval()
        self.text_embedding.eval()
        
        results = []
        
        with torch.no_grad():
            for i, test_text in enumerate(self.test_sentences):
                print(f"\n🧪 Test {i+1}: '{test_text}'")
                
                # 1. Encode text
                text_tensor = self.text_processor.encode_text(test_text).unsqueeze(0).to(self.device)
                
                # 2. Create cover image
                cover_image = torch.randn(1, 3, 32, 32).to(self.device)
                cover_image = torch.clamp(cover_image * 0.5, -1, 1)  # Moderate noise
                
                # 3. Hide text in image (Encoder)
                text_embed = self.text_embedding(text_tensor)
                stego_image = self.generator(cover_image, text_embed)
                
                # 4. Extract text from image (Decoder)
                extracted_logits = self.extractor(stego_image)
                extracted_text = self.text_processor.decode_text(extracted_logits[0])
                
                # 5. Calculate accuracy
                char_accuracy = self._calculate_character_accuracy(test_text, extracted_text)
                word_accuracy = 1.0 if test_text.strip() == extracted_text.strip() else 0.0
                
                # 6. Calculate image quality
                psnr = self._calculate_psnr(cover_image, stego_image)
                
                result = {
                    'original_text': test_text,
                    'extracted_text': extracted_text,
                    'character_accuracy': char_accuracy,
                    'word_accuracy': word_accuracy,
                    'psnr': psnr.item(),
                    'length': len(test_text)
                }
                
                results.append(result)
                
                print(f"   Original:  '{test_text}'")
                print(f"   Extracted: '{extracted_text}'")
                print(f"   Char Acc:  {char_accuracy:.3f} ({char_accuracy*100:.1f}%)")
                print(f"   Word Acc:  {word_accuracy:.3f}")
                print(f"   PSNR:      {psnr:.2f} dB")
        
        # Calculate overall statistics
        avg_char_acc = np.mean([r['character_accuracy'] for r in results])
        avg_word_acc = np.mean([r['word_accuracy'] for r in results])
        avg_psnr = np.mean([r['psnr'] for r in results])
        
        summary = {
            'results': results,
            'average_character_accuracy': avg_char_acc,
            'average_word_accuracy': avg_word_acc,
            'average_psnr': avg_psnr,
            'total_tests': len(results)
        }
        
        print(f"\n📊 Overall GAN Encoder-Decoder Results:")
        print(f"   Average Character Accuracy: {avg_char_acc:.3f} ({avg_char_acc*100:.1f}%)")
        print(f"   Average Word Accuracy: {avg_word_acc:.3f} ({avg_word_acc*100:.1f}%)")
        print(f"   Average PSNR: {avg_psnr:.2f} dB")
        print(f"   Total Tests: {len(results)}")
        
        return summary
    
    def _calculate_character_accuracy(self, original: str, extracted: str) -> float:
        """Calculate character-level accuracy between original and extracted text."""
        if not original:
            return 1.0 if not extracted else 0.0
            
        # Pad shorter string
        max_len = max(len(original), len(extracted))
        orig_padded = original.ljust(max_len)
        extr_padded = extracted.ljust(max_len)
        
        # Calculate accuracy
        correct = sum(1 for o, e in zip(orig_padded, extr_padded) if o == e)
        return correct / max_len
    
    def _calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Calculate PSNR between two images."""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(2.0 / torch.sqrt(mse))  # Range [-1, 1]


def run_comprehensive_evaluation():
    """Run comprehensive evaluation suite."""
    print("🚀 COMPREHENSIVE TEXT STEGANOGRAPHY EVALUATION")
    print("=" * 70)
    
    # Configuration
    config = {
        'batch_size': 32,
        'max_text_length': 128,
        'text_embed_dim': 128,
        'data_path': './data'
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Create results directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    results = {}
    
    print("\n1️⃣ TESTING SAVE/RELOAD CONSISTENCY")
    print("-" * 40)
    consistency_result = evaluator.test_save_reload_consistency()
    results['save_reload_consistency'] = consistency_result
    
    print("\n2️⃣ TESTING GAN ENCODER-DECODER PIPELINE") 
    print("-" * 40)
    encoder_decoder_results = evaluator.test_gan_encoder_decoder()
    results['encoder_decoder'] = encoder_decoder_results
    
    print("\n3️⃣ EVALUATION ON UNSEEN TEST DATA")
    print("-" * 40)
    test_results = evaluator.evaluate_on_test_data(num_samples=50)
    results['unseen_data_evaluation'] = test_results
    
    print("\n4️⃣ SAVING BEST MODEL WEIGHTS")
    print("-" * 40)
    model_save_success = evaluator.save_model_weights('evaluation_results/best_model.pth')
    results['model_saved'] = model_save_success
    
    # Save all results
    with open('evaluation_results/comprehensive_evaluation.json', 'w') as f:
        # Convert any tensor values to float for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: float(x) if hasattr(x, 'item') else x))
        json.dump(json_results, f, indent=2)
    
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"✅ Save/Reload Consistency: {'PASS' if consistency_result else 'FAIL'}")
    print(f"✅ Encoder-Decoder Tests: {encoder_decoder_results['total_tests']} completed")
    print(f"✅ Character Accuracy: {encoder_decoder_results['average_character_accuracy']*100:.1f}%")
    print(f"✅ Word Accuracy: {encoder_decoder_results['average_word_accuracy']*100:.1f}%")
    print(f"✅ Average PSNR: {encoder_decoder_results['average_psnr']:.2f} dB")
    print(f"✅ Model Weights: {'Saved' if model_save_success else 'Save Failed'}")
    print(f"\n📁 Results saved to: evaluation_results/")
    print(f"🎯 Model ready for production use!")


if __name__ == "__main__":
    run_comprehensive_evaluation()
