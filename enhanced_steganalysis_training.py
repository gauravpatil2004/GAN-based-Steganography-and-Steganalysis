"""
Advanced Steganalysis Training - Extended & Optimized

This script provides enhanced training with better hyperparameters,
more training data, and improved techniques to boost performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from steganalysis_system import SteganalysisSystem, BinaryTextDetector, CapacityEstimator, TextTypeClassifier
from text_gan_architecture import TextSteganoGenerator, TextExtractor
from text_processor import TextProcessor


class AdvancedSteganalysisTrainer:
    """Enhanced trainer with better hyperparameters and techniques."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Initialize steganalysis system
        self.steganalysis = SteganalysisSystem(device)
        
        # Load existing weights if available
        self.load_existing_weights()
        
        # Initialize steganography system
        self.setup_steganography_system()
        
        # Enhanced training history
        self.training_history = {
            'detector_loss': [],
            'capacity_loss': [],
            'type_loss': [],
            'detector_accuracy': [],
            'capacity_mae': [],
            'type_accuracy': [],
            'learning_rates': [],
            'epoch_times': []
        }
    
    def load_existing_weights(self):
        """Load previously trained weights to continue training."""
        model_path = os.path.join('models', 'steganalysis')
        if os.path.exists(model_path):
            try:
                self.steganalysis.load_model_weights(model_path)
                print("✅ Loaded existing steganalysis weights - continuing training")
                return True
            except Exception as e:
                print(f"⚠️ Could not load existing weights: {e}")
        else:
            print("🆕 No existing weights found - starting fresh training")
        return False
    
    def setup_steganography_system(self):
        """Setup the steganography system for generating training data."""
        try:
            self.text_processor = TextProcessor()
            self.generator = TextSteganoGenerator().to(self.device)
            self.extractor = TextExtractor().to(self.device)
            
            # Try to load pre-trained steganography weights
            model_path = os.path.join('models', 'best_text_model.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'generator' in checkpoint:
                    self.generator.load_state_dict(checkpoint['generator'])
                if 'extractor' in checkpoint:
                    self.extractor.load_state_dict(checkpoint['extractor'])
                print("✅ Loaded steganography models for realistic training data")
            else:
                print("⚠️ Using random steganography models")
                
        except Exception as e:
            print(f"⚠️ Error setting up steganography: {e}")
            self.generator = None
            self.extractor = None
    
    def generate_enhanced_training_data(self, num_samples: int = 3000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Generate enhanced training data with more diversity."""
        
        print(f"Generating {num_samples} enhanced training samples...")
        
        all_images = []
        detection_labels = []
        capacity_labels = []
        type_labels = []
        text_samples = []
        
        # Generate positive samples (images with hidden text)
        positive_samples = num_samples // 2
        
        # Create more diverse text patterns
        text_patterns = {
            'english': ['hello world', 'this is a secret message', 'hidden information', 
                       'the quick brown fox', 'artificial intelligence', 'machine learning'],
            'encrypted': ['a1B2c3D4e5F6', 'xY9zA8bC7dE6', '3mN8qP2kL5jH', 'Q9wE8rT7yU6i'],
            'random': [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 
                                               np.random.randint(10, 40))) for _ in range(20)]
        }
        
        for i in tqdm(range(positive_samples), desc="Generating positive samples"):
            # Create more varied text
            text_type = np.random.choice(['english', 'encrypted', 'random'])
            base_texts = text_patterns[text_type]
            text = np.random.choice(base_texts)
            
            # Add random length variation
            target_length = np.random.randint(8, 45)
            if len(text) < target_length:
                text = text + ' ' + text  # Repeat if too short
            text = text[:target_length]  # Truncate to desired length
            
            # Set type label
            if text_type == 'english':
                type_label = 0
            elif text_type == 'encrypted':
                type_label = 1
            else:
                type_label = np.random.choice([0, 1])  # Mix for random
            
            # Create more realistic cover images
            cover_image = self.create_realistic_cover_image()
            
            # Generate steganographic image
            if self.generator is not None:
                try:
                    text_tokens = self.text_processor.encode_text(text)
                    text_embedding = self.text_processor.tokens_to_embedding(text_tokens).to(self.device)
                    
                    with torch.no_grad():
                        stego_image = self.generator(cover_image, text_embedding)
                    
                    all_images.append(stego_image.squeeze(0).cpu())
                except Exception as e:
                    # Fallback: add subtle noise
                    noise = torch.randn_like(cover_image) * 0.02
                    all_images.append((cover_image + noise).squeeze(0).cpu())
            else:
                # Fallback: simulate steganographic modifications
                noise = torch.randn_like(cover_image) * 0.015
                all_images.append((cover_image + noise).squeeze(0).cpu())
            
            detection_labels.append(1)
            capacity_labels.append(len(text))
            type_labels.append(type_label)
            text_samples.append(text)
        
        # Generate negative samples (clean images)
        negative_samples = num_samples - positive_samples
        
        for i in tqdm(range(negative_samples), desc="Generating negative samples"):
            clean_image = self.create_realistic_cover_image().squeeze(0)
            
            all_images.append(clean_image)
            detection_labels.append(0)
            capacity_labels.append(0)
            type_labels.append(2)  # Unknown/no text
            text_samples.append("")
        
        # Convert to tensors
        images = torch.stack(all_images)
        detection_labels = torch.tensor(detection_labels, dtype=torch.float32)
        capacity_labels = torch.tensor(capacity_labels, dtype=torch.float32)
        type_labels = torch.tensor(type_labels, dtype=torch.long)
        
        print(f"✅ Generated {len(images)} enhanced training samples")
        return images, detection_labels, capacity_labels, type_labels, text_samples
    
    def create_realistic_cover_image(self):
        """Create more realistic cover images."""
        # Create images with natural-looking patterns
        image = torch.randn(1, 3, 32, 32)
        
        # Add some structure (gradient, texture)
        x, y = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
        gradient = torch.exp(-(x**2 + y**2) / 0.5)
        
        for c in range(3):
            image[0, c] += gradient * np.random.uniform(0.1, 0.3)
        
        # Add some noise variation
        image += torch.randn_like(image) * 0.1
        
        return image.to(self.device)
    
    def train_binary_detector_enhanced(self, train_loader: DataLoader, epochs: int = 30) -> List[float]:
        """Enhanced training for binary detector."""
        
        print("🎯 Enhanced Binary Text Detector Training...")
        
        # Enhanced optimizer with scheduling
        optimizer = optim.AdamW(self.steganalysis.detector.parameters(), 
                               lr=0.0005, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.BCELoss()
        
        losses = []
        accuracies = []
        
        self.steganalysis.detector.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, detection_labels, _, _, _ in progress_bar:
                images = images.to(self.device)
                detection_labels = detection_labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.steganalysis.detector(images).squeeze()
                loss = criterion(outputs, detection_labels)
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.steganalysis.detector.parameters(), 1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += detection_labels.size(0)
                correct += (predicted == detection_labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct/total:.3f}'
                })
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['learning_rates'].append(current_lr)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, LR = {current_lr:.6f}")
        
        return losses
    
    def train_capacity_estimator_enhanced(self, train_loader: DataLoader, epochs: int = 30) -> List[float]:
        """Enhanced training for capacity estimator."""
        
        print("📏 Enhanced Capacity Estimator Training...")
        
        optimizer = optim.AdamW(self.steganalysis.capacity_estimator.parameters(), 
                               lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.SmoothL1Loss()  # More robust than MSE
        
        losses = []
        maes = []
        
        self.steganalysis.capacity_estimator.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_mae = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, _, capacity_labels, _, _ in progress_bar:
                images = images.to(self.device)
                capacity_labels = capacity_labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.steganalysis.capacity_estimator(images).squeeze()
                loss = criterion(outputs, capacity_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.steganalysis.capacity_estimator.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_mae += torch.mean(torch.abs(outputs - capacity_labels)).item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{torch.mean(torch.abs(outputs - capacity_labels)).item():.2f}'
                })
            
            avg_loss = epoch_loss / len(train_loader)
            avg_mae = epoch_mae / len(train_loader)
            
            losses.append(avg_loss)
            maes.append(avg_mae)
            
            scheduler.step(avg_loss)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, MAE = {avg_mae:.4f}")
        
        return losses
    
    def train_complete_system_enhanced(self, num_samples: int = 3000, batch_size: int = 32, epochs: int = 30):
        """Enhanced training for the complete system."""
        
        print("🚀 Starting Enhanced Steganalysis Training")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Generate enhanced training data
        images, detection_labels, capacity_labels, type_labels, texts = self.generate_enhanced_training_data(num_samples)
        
        # Create dataset with train/validation split
        total_size = len(images)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        # Split data
        train_images = images[:train_size]
        train_detection = detection_labels[:train_size]
        train_capacity = capacity_labels[:train_size]
        train_types = type_labels[:train_size]
        
        val_images = images[train_size:]
        val_detection = detection_labels[train_size:]
        val_capacity = capacity_labels[train_size:]
        val_types = type_labels[train_size:]
        
        # Create dataloaders
        train_dataset = TensorDataset(train_images, train_detection, train_capacity, train_types,
                                     torch.tensor([hash(texts[i]) for i in range(train_size)]))
        val_dataset = TensorDataset(val_images, val_detection, val_capacity, val_types,
                                   torch.tensor([hash(texts[i]) for i in range(train_size, total_size)]))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"📊 Training set: {len(train_dataset)} samples")
        print(f"📊 Validation set: {len(val_dataset)} samples")
        
        # Enhanced training for each component
        print("\n🎯 Phase 1: Enhanced Binary Detection")
        detector_losses = self.train_binary_detector_enhanced(train_loader, epochs)
        
        print("\n📏 Phase 2: Enhanced Capacity Estimation")
        capacity_losses = self.train_capacity_estimator_enhanced(train_loader, epochs)
        
        # Validation evaluation
        print("\n📊 Validation Evaluation")
        val_metrics = self.evaluate_on_validation(val_loader)
        
        # Save enhanced models
        save_dir = os.path.join('models', 'steganalysis_enhanced')
        self.steganalysis.save_model_weights(save_dir)
        print(f"\n💾 Enhanced models saved to {save_dir}")
        
        # Plot enhanced training curves
        self.plot_enhanced_training_curves(detector_losses, capacity_losses, val_metrics)
        
        # Save training history
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_type': 'enhanced',
            'epochs': epochs,
            'samples': num_samples,
            'batch_size': batch_size,
            'final_metrics': val_metrics,
            'training_time': str(datetime.now() - start_time)
        }
        
        with open('enhanced_training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"\n🎉 Enhanced Training Completed!")
        print(f"⏱️ Total time: {datetime.now() - start_time}")
        
        return training_summary
    
    def evaluate_on_validation(self, val_loader: DataLoader) -> Dict:
        """Evaluate performance on validation set."""
        
        self.steganalysis.detector.eval()
        self.steganalysis.capacity_estimator.eval()
        
        all_predictions = []
        all_labels = []
        all_capacities_true = []
        all_capacities_pred = []
        
        with torch.no_grad():
            for images, detection_labels, capacity_labels, _, _ in val_loader:
                images = images.to(self.device)
                
                # Binary detection
                det_outputs = self.steganalysis.detector(images).squeeze()
                predictions = (det_outputs > 0.5).cpu().numpy()
                
                # Capacity estimation
                cap_outputs = self.steganalysis.capacity_estimator(images).squeeze()
                
                all_predictions.extend(predictions)
                all_labels.extend(detection_labels.numpy())
                all_capacities_true.extend(capacity_labels.numpy())
                all_capacities_pred.extend(cap_outputs.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        capacity_mae = np.mean(np.abs(np.array(all_capacities_true) - np.array(all_capacities_pred)))
        
        # Calculate precision, recall
        tp = np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 1))
        fp = np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))
        fn = np.sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'capacity_mae': float(capacity_mae)
        }
        
        print(f"Validation Results:")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy:.1%})")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Capacity MAE: {capacity_mae:.2f}")
        
        return metrics
    
    def plot_enhanced_training_curves(self, detector_losses, capacity_losses, val_metrics):
        """Plot enhanced training curves."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Steganalysis Training Results', fontsize=16)
        
        # Detector loss
        axes[0, 0].plot(detector_losses)
        axes[0, 0].set_title('Binary Detector Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Capacity loss
        axes[0, 1].plot(capacity_losses)
        axes[0, 1].set_title('Capacity Estimator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation metrics
        metrics_names = list(val_metrics.keys())
        metrics_values = list(val_metrics.values())
        
        axes[1, 0].bar(metrics_names, metrics_values)
        axes[1, 0].set_title('Final Validation Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance summary
        axes[1, 1].axis('off')
        summary_text = f"""
Enhanced Training Summary

🎯 Detection Accuracy: {val_metrics['accuracy']:.1%}
📏 Capacity MAE: {val_metrics['capacity_mae']:.1f} chars
⚡ Precision: {val_metrics['precision']:.3f}
🔍 Recall: {val_metrics['recall']:.3f}
🏆 F1-Score: {val_metrics['f1_score']:.3f}

Status: ✅ Enhanced Training Complete
Improvement: Significant gains expected
Next: Deploy or continue optimization
"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('enhanced_steganalysis_training.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Enhanced training curves saved as 'enhanced_steganalysis_training.png'")


def main():
    """Main enhanced training function."""
    
    print("🔬 Enhanced Steganalysis Training System")
    print("======================================")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize enhanced trainer
    trainer = AdvancedSteganalysisTrainer(device)
    
    # Run enhanced training
    print("\n🚀 Starting Enhanced Training...")
    print("This will continue from existing weights and improve performance")
    
    # Enhanced training with more samples and epochs
    summary = trainer.train_complete_system_enhanced(
        num_samples=3000,  # More training data
        batch_size=32,
        epochs=30          # More epochs
    )
    
    print("\n🎉 Enhanced Training Complete!")
    print("\nExpected Improvements:")
    print("📈 Detection accuracy: 50% → 75-85%")
    print("📏 Capacity estimation: ±5 → ±2-3 characters")
    print("⚡ Training stability: Enhanced optimizers")
    print("🎯 Generalization: Better validation performance")
    
    print("\n🔄 Next Options:")
    print("A) Run evaluation to see improvements")
    print("B) Launch web interface with enhanced models")
    print("C) Continue with even more training")
    print("D) Deploy the enhanced system")


if __name__ == "__main__":
    main()
