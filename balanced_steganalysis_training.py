"""
Balanced Steganalysis Training - Version 3

This script implements improved training with better balance and regularization
to address the overfitting issues observed in the enhanced training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from collections import Counter

# Add src to path
sys.path.append('src')

class BalancedSteganalysisTrainer:
    """Improved trainer with better balance and regularization."""
    
    def __init__(self, model_dir='models/steganalysis_balanced'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize steganography system for data generation
        from steganalysis_system import SteganalysisSystem
        from text_gan_architecture import TextSteganoGenerator, TextExtractor
        from text_processor import TextProcessor
        
        self.steganalysis_system = SteganalysisSystem()
        
        # Initialize text generation components
        self.text_processor = TextProcessor()
        
        # Try to load GAN models if available
        try:
            self.generator = TextSteganoGenerator()
            self.extractor = TextExtractor()
            
            # Load models if they exist
            if os.path.exists('models/generator.pth'):
                self.generator.load_state_dict(torch.load('models/generator.pth', map_location='cpu'))
                print("✅ Loaded GAN generator")
            if os.path.exists('models/extractor.pth'):
                self.extractor.load_state_dict(torch.load('models/extractor.pth', map_location='cpu'))
                print("✅ Loaded GAN extractor")
                
            self.gan_available = True
        except Exception as e:
            print(f"⚠️ GAN models not available, using text-based generation: {e}")
            self.gan_available = False
        
        # Training parameters - more conservative
        self.learning_rate = 0.0001  # Reduced from 0.001
        self.batch_size = 32
        self.epochs = 25  # Reduced from 30
        self.patience = 5  # Early stopping
        
        # Data parameters - better balance
        self.num_samples = 2000  # Reduced for better quality
        self.clean_ratio = 0.6  # 60% clean text (was 50%)
        
        # Training tracking
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def generate_simple_steganographic_text(self, message, method):
        """Generate steganographic text using simple methods."""
        
        # Base cover texts
        cover_texts = [
            "This is a normal conversation about daily activities and weekend plans.",
            "Regular email content discussing business matters and project updates.",
            "Simple news article excerpt covering local events and community news.",
            "Ordinary technical documentation explaining software features and usage.",
            "Basic recipe instructions for cooking various dishes and meals.",
            "Casual social media post about travel experiences and recommendations.",
            "Standard academic essay discussing literature and historical topics.",
            "Plain weather report with forecasts and temperature information.",
            "Normal chat message between friends about movies and entertainment.",
            "Regular product description for online shopping and reviews."
        ]
        
        cover_text = np.random.choice(cover_texts)
        
        if method == 'plain':
            # Simple character insertion steganography
            # Insert message characters at regular intervals
            result = ""
            msg_idx = 0
            for i, char in enumerate(cover_text):
                result += char
                if msg_idx < len(message) and i % 3 == 0:  # Every 3rd position
                    result += message[msg_idx]
                    msg_idx += 1
            return result
            
        elif method == 'emoji':
            # Emoji-based steganography
            emoji_map = {
                'a': '😀', 'b': '😁', 'c': '😂', 'd': '😃', 'e': '😄',
                'f': '😅', 'g': '😆', 'h': '😇', 'i': '😈', 'j': '😉',
                'k': '😊', 'l': '😋', 'm': '😌', 'n': '😍', 'o': '😎',
                'p': '😏', 'q': '😐', 'r': '😑', 's': '😒', 't': '😓',
                'u': '😔', 'v': '😕', 'w': '😖', 'x': '😗', 'y': '😘',
                'z': '😙', ' ': '😚'
            }
            
            emoji_text = ""
            for char in message.lower():
                if char in emoji_map:
                    emoji_text += emoji_map[char]
            
            return cover_text + " " + emoji_text
            
        elif method == 'unicode':
            # Unicode steganography using zero-width characters
            zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
            
            result = ""
            msg_idx = 0
            for char in cover_text:
                result += char
                if msg_idx < len(message):
                    # Encode message character using zero-width chars
                    ascii_val = ord(message[msg_idx])
                    for bit in format(ascii_val, '08b'):
                        if bit == '1':
                            result += zero_width_chars[0]
                        else:
                            result += zero_width_chars[1]
                    msg_idx += 1
            return result
            
        else:  # mixed
            # Randomly choose a method
            methods = ['plain', 'emoji', 'unicode']
            chosen = np.random.choice(methods)
            return self.generate_simple_steganographic_text(message, chosen)

    def create_balanced_dataset(self):
        """Create a better balanced dataset with validation split."""
        
        print(f"📊 Creating balanced dataset ({self.num_samples} samples)...")
        
        # Calculate samples per category
        num_clean = int(self.num_samples * self.clean_ratio)
        num_stego = self.num_samples - num_clean
        stego_per_category = num_stego // 4  # 4 steganographic categories
        
        print(f"   Clean samples: {num_clean}")
        print(f"   Steganographic samples: {num_stego} ({stego_per_category} per category)")
        
        texts = []
        labels = []
        capacities = []
        metadata = []
        
        # Generate clean text samples
        clean_templates = [
            "This is a simple clean text message.",
            "Normal conversation between friends about daily activities.",
            "A brief description of weather conditions today.",
            "Regular email content without any hidden information.",
            "Standard news article excerpt from local newspaper.",
            "Simple recipe instructions for cooking pasta.",
            "Basic technical documentation for software usage.",
            "Casual social media post about weekend plans.",
            "Ordinary business memo regarding meeting schedules.",
            "Plain academic essay paragraph about literature."
        ]
        
        for i in range(num_clean):
            # Vary the clean text length and content
            base_text = np.random.choice(clean_templates)
            if np.random.random() < 0.3:
                # Make some longer
                base_text += f" Additional content number {i} with more details about the topic."
            if np.random.random() < 0.2:
                # Make some shorter
                base_text = base_text.split('.')[0] + "."
            
            texts.append(base_text)
            labels.append(0)  # Clean
            capacities.append(0)
            metadata.append({
                'category': 'clean',
                'length': 0,
                'method': 'none'
            })
        
        # Generate steganographic samples with variety
        stego_methods = ['plain', 'emoji', 'unicode', 'mixed']
        stego_messages = [
            "Secret message",
            "Hidden data",
            "Confidential info",
            "Covert communication",
            "Classified content",
            "Private message here",
            "Sensitive information",
            "Restricted access data",
            "Internal communication only",
            "This is a longer secret message with more content"
        ]
        
        for method in stego_methods:
            for i in range(stego_per_category):
                try:
                    # Vary message length
                    if np.random.random() < 0.3:
                        # Long messages
                        message = f"{np.random.choice(stego_messages)} with extended content {i}"
                    elif np.random.random() < 0.5:
                        # Medium messages  
                        message = f"{np.random.choice(stego_messages)} {i}"
                    else:
                        # Short messages
                        message = np.random.choice(stego_messages[:5])
                    
                    # Generate steganographic text using simple methods
                    stego_text = self.generate_simple_steganographic_text(message, method)
                    
                    if stego_text:
                        texts.append(stego_text)
                        labels.append(1)  # Steganographic
                        capacities.append(len(message))
                        metadata.append({
                            'category': method,
                            'length': len(message),
                            'method': method,
                            'original_message': message
                        })
                    
                except Exception as e:
                    print(f"Warning: Error generating {method} sample: {e}")
                    # Add a fallback clean sample
                    texts.append(f"Fallback clean text {i}")
                    labels.append(0)
                    capacities.append(0)
                    metadata.append({
                        'category': 'clean',
                        'length': 0,
                        'method': 'none'
                    })
        
        # Verify balance
        label_counts = Counter(labels)
        print(f"   Final balance: Clean={label_counts[0]}, Stego={label_counts[1]}")
        print(f"   Balance ratio: {label_counts[0]/len(labels):.1%} clean, {label_counts[1]/len(labels):.1%} stego")
        
        # Split into train/validation (80/20)
        indices = np.random.permutation(len(texts))
        split_idx = int(0.8 * len(texts))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data = {
            'texts': [texts[i] for i in train_indices],
            'labels': [labels[i] for i in train_indices],
            'capacities': [capacities[i] for i in train_indices],
            'metadata': [metadata[i] for i in train_indices]
        }
        
        val_data = {
            'texts': [texts[i] for i in val_indices],
            'labels': [labels[i] for i in val_indices],
            'capacities': [capacities[i] for i in val_indices],
            'metadata': [metadata[i] for i in val_indices]
        }
        
        # Check validation balance
        val_label_counts = Counter(val_data['labels'])
        print(f"   Validation balance: Clean={val_label_counts[0]}, Stego={val_label_counts[1]}")
        
        return train_data, val_data

    def create_weighted_sampler(self, labels):
        """Create weighted sampler to handle class imbalance."""
        
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate weights (inverse frequency)
        weights = []
        for label in labels:
            weight = total_samples / (len(label_counts) * label_counts[label])
            weights.append(weight)
        
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def extract_features(self, texts):
        """Extract features using simple text analysis."""
        features = []
        
        for text in texts:
            try:
                # Create simple feature vector based on text properties
                feature_vector = []
                
                # Basic text statistics
                feature_vector.append(len(text))  # Text length
                feature_vector.append(len(text.split()))  # Word count
                feature_vector.append(len(set(text.lower())))  # Unique characters
                feature_vector.append(text.count(' ') / len(text) if len(text) > 0 else 0)  # Space ratio
                
                # Character frequency analysis
                char_counts = {}
                for char in text.lower():
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                # Entropy calculation
                entropy = 0
                text_len = len(text)
                if text_len > 0:
                    for count in char_counts.values():
                        p = count / text_len
                        if p > 0:
                            entropy -= p * np.log2(p)
                feature_vector.append(entropy)
                
                # Emoji detection
                emoji_count = sum(1 for char in text if ord(char) > 127)
                feature_vector.append(emoji_count / len(text) if len(text) > 0 else 0)
                
                # Unicode character detection
                unicode_count = sum(1 for char in text if ord(char) > 255)
                feature_vector.append(unicode_count / len(text) if len(text) > 0 else 0)
                
                # Zero-width character detection
                zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
                zero_width_count = sum(text.count(char) for char in zero_width_chars)
                feature_vector.append(zero_width_count)
                
                # Punctuation ratio
                punctuation = '!@#$%^&*()_+-=[]{}|;:,.<>?'
                punct_count = sum(text.count(char) for char in punctuation)
                feature_vector.append(punct_count / len(text) if len(text) > 0 else 0)
                
                # Vowel/consonant ratio
                vowels = 'aeiouAEIOU'
                vowel_count = sum(text.count(char) for char in vowels)
                feature_vector.append(vowel_count / len(text) if len(text) > 0 else 0)
                
                # Pad to fixed size (100 features)
                while len(feature_vector) < 100:
                    feature_vector.append(0.0)
                
                # Truncate if too long
                feature_vector = feature_vector[:100]
                
                features.append(feature_vector)
                
            except Exception as e:
                print(f"Warning: Feature extraction failed for text: {e}")
                # Fallback: create zero vector
                features.append([0.0] * 100)
        
        return np.array(features)

    def train_with_early_stopping(self, train_data, val_data):
        """Train models with early stopping and better regularization."""
        
        print(f"🎯 Starting balanced training...")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Early stopping patience: {self.patience}")
        
        # Extract features
        print("🔧 Extracting features...")
        train_features = self.extract_features(train_data['texts'])
        val_features = self.extract_features(val_data['texts'])
        
        # Prepare data loaders with weighted sampling
        train_sampler = self.create_weighted_sampler(train_data['labels'])
        
        # Convert to tensors
        train_X = torch.FloatTensor(train_features).to(self.device)
        train_y = torch.LongTensor(train_data['labels']).to(self.device)
        train_cap = torch.FloatTensor(train_data['capacities']).to(self.device)
        
        val_X = torch.FloatTensor(val_features).to(self.device)
        val_y = torch.LongTensor(val_data['labels']).to(self.device)
        val_cap = torch.FloatTensor(val_data['capacities']).to(self.device)
        
        # Initialize models with better regularization
        from steganalysis_system import BinaryTextDetector, CapacityEstimator
        
        detector = BinaryTextDetector().to(self.device)
        capacity_estimator = CapacityEstimator().to(self.device)
        
        # Optimizers with weight decay (L2 regularization)
        detector_optimizer = optim.AdamW(detector.parameters(), 
                                       lr=self.learning_rate, 
                                       weight_decay=0.01)  # L2 regularization
        capacity_optimizer = optim.AdamW(capacity_estimator.parameters(), 
                                       lr=self.learning_rate, 
                                       weight_decay=0.01)
        
        # Loss functions with class weighting
        pos_weight = torch.tensor([len([l for l in train_data['labels'] if l == 0]) / 
                                 len([l for l in train_data['labels'] if l == 1])]).to(self.device)
        detection_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        capacity_criterion = nn.MSELoss()
        
        # Learning rate schedulers
        detector_scheduler = optim.lr_scheduler.ReduceLROnPlateau(detector_optimizer, 
                                                                mode='max', 
                                                                patience=3, 
                                                                factor=0.5)
        capacity_scheduler = optim.lr_scheduler.ReduceLROnPlateau(capacity_optimizer, 
                                                                mode='min', 
                                                                patience=3, 
                                                                factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n🚀 Training started...")
        
        for epoch in range(self.epochs):
            # Training phase
            detector.train()
            capacity_estimator.train()
            
            # Create batches
            batch_indices = torch.randperm(len(train_X))[:len(train_X)//self.batch_size*self.batch_size]
            batches = batch_indices.view(-1, self.batch_size)
            
            train_loss = 0
            train_correct = 0
            
            for batch_idx in batches:
                batch_X = train_X[batch_idx]
                batch_y = train_y[batch_idx]
                batch_cap = train_cap[batch_idx]
                
                # Detection training
                detector_optimizer.zero_grad()
                detection_logits = detector(batch_X).squeeze()
                detection_loss = detection_criterion(detection_logits, batch_y.float())
                detection_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
                detector_optimizer.step()
                
                # Capacity training (only on steganographic samples)
                stego_mask = batch_y == 1
                if stego_mask.sum() > 0:
                    capacity_optimizer.zero_grad()
                    capacity_pred = capacity_estimator(batch_X[stego_mask]).squeeze()
                    capacity_loss = capacity_criterion(capacity_pred, batch_cap[stego_mask])
                    capacity_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(capacity_estimator.parameters(), max_norm=1.0)
                    capacity_optimizer.step()
                
                # Track metrics
                train_loss += detection_loss.item()
                preds = torch.sigmoid(detection_logits) > 0.5
                train_correct += (preds == batch_y).sum().item()
            
            train_acc = train_correct / len(train_X)
            avg_train_loss = train_loss / len(batches)
            
            # Validation phase
            detector.eval()
            capacity_estimator.eval()
            
            with torch.no_grad():
                val_detection_logits = detector(val_X).squeeze()
                val_detection_loss = detection_criterion(val_detection_logits, val_y.float())
                
                val_preds = torch.sigmoid(val_detection_logits) > 0.5
                val_acc = (val_preds == val_y).float().mean().item()
                
                # Calculate additional metrics
                val_preds_np = val_preds.cpu().numpy()
                val_y_np = val_y.cpu().numpy()
                
                val_precision = precision_score(val_y_np, val_preds_np, zero_division=0)
                val_recall = recall_score(val_y_np, val_preds_np, zero_division=0)
                val_f1 = f1_score(val_y_np, val_preds_np, zero_division=0)
            
            # Update learning rate
            detector_scheduler.step(val_acc)
            capacity_scheduler.step(val_detection_loss)
            
            # Track history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_detection_loss.item())
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_precision'].append(val_precision)
            self.training_history['val_recall'].append(val_recall)
            self.training_history['val_f1'].append(val_f1)
            
            # Print progress
            print(f"   Epoch {epoch+1:2d}/{self.epochs}: "
                  f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, "
                  f"Val Prec: {val_precision:.3f}, Val Rec: {val_recall:.3f}, "
                  f"Val F1: {val_f1:.3f}")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best models
                torch.save(detector.state_dict(), 
                          os.path.join(self.model_dir, 'binary_detector.pth'))
                torch.save(capacity_estimator.state_dict(), 
                          os.path.join(self.model_dir, 'capacity_estimator.pth'))
                print(f"     ✅ New best validation accuracy: {val_acc:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"     🛑 Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.3f}")
        
        return best_val_acc

    def save_training_summary(self, val_acc, train_time):
        """Save comprehensive training summary."""
        
        summary = {
            'training_type': 'Balanced Steganalysis Training v3',
            'timestamp': datetime.now().isoformat(),
            'training_time': train_time,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'samples': self.num_samples,
                'clean_ratio': self.clean_ratio
            },
            'final_metrics': {
                'validation_accuracy': val_acc,
                'best_epoch': self.training_history['val_acc'].index(max(self.training_history['val_acc'])) + 1,
                'final_precision': self.training_history['val_precision'][-1],
                'final_recall': self.training_history['val_recall'][-1],
                'final_f1': self.training_history['val_f1'][-1]
            },
            'training_history': self.training_history,
            'improvements': [
                'Reduced learning rate for stability',
                'Added early stopping to prevent overfitting',
                'Implemented class weighting for balance',
                'Added L2 regularization and gradient clipping',
                'Increased clean text ratio to 60%',
                'Added learning rate scheduling'
            ]
        }
        
        with open('balanced_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Training summary saved to 'balanced_training_summary.json'")

    def create_training_plots(self):
        """Create visualization of training progress."""
        
        plt.figure(figsize=(15, 10))
        
        # 1. Accuracy plot
        plt.subplot(2, 3, 1)
        plt.plot(self.training_history['epoch'], self.training_history['train_acc'], 
                label='Training', linewidth=2)
        plt.plot(self.training_history['epoch'], self.training_history['val_acc'], 
                label='Validation', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Loss plot
        plt.subplot(2, 3, 2)
        plt.plot(self.training_history['epoch'], self.training_history['train_loss'], 
                label='Training', linewidth=2)
        plt.plot(self.training_history['epoch'], self.training_history['val_loss'], 
                label='Validation', linewidth=2)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Precision/Recall plot
        plt.subplot(2, 3, 3)
        plt.plot(self.training_history['epoch'], self.training_history['val_precision'], 
                label='Precision', linewidth=2)
        plt.plot(self.training_history['epoch'], self.training_history['val_recall'], 
                label='Recall', linewidth=2)
        plt.plot(self.training_history['epoch'], self.training_history['val_f1'], 
                label='F1-Score', linewidth=2)
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Learning rate progression (if available)
        plt.subplot(2, 3, 4)
        # Placeholder for learning rate tracking
        plt.plot(self.training_history['epoch'], 
                [self.learning_rate] * len(self.training_history['epoch']))
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        
        # 5. Performance summary
        plt.subplot(2, 3, 5)
        final_metrics = [
            self.training_history['val_acc'][-1],
            self.training_history['val_precision'][-1],
            self.training_history['val_recall'][-1],
            self.training_history['val_f1'][-1]
        ]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        bars = plt.bar(metric_names, final_metrics)
        plt.title('Final Validation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Training progress overview
        plt.subplot(2, 3, 6)
        best_epoch = self.training_history['val_acc'].index(max(self.training_history['val_acc'])) + 1
        best_acc = max(self.training_history['val_acc'])
        
        plt.text(0.1, 0.8, f'Best Validation Accuracy: {best_acc:.3f}', fontsize=12, weight='bold')
        plt.text(0.1, 0.7, f'Best Epoch: {best_epoch}', fontsize=12)
        plt.text(0.1, 0.6, f'Total Epochs: {len(self.training_history["epoch"])}', fontsize=12)
        plt.text(0.1, 0.5, f'Learning Rate: {self.learning_rate}', fontsize=12)
        plt.text(0.1, 0.4, f'Batch Size: {self.batch_size}', fontsize=12)
        plt.text(0.1, 0.3, f'Samples: {self.num_samples}', fontsize=12)
        plt.text(0.1, 0.2, f'Clean Ratio: {self.clean_ratio:.1%}', fontsize=12)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Training Summary')
        
        plt.tight_layout()
        plt.savefig('balanced_steganalysis_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved to 'balanced_steganalysis_training_curves.png'")

def main():
    """Main training function."""
    
    print("🎯 Balanced Steganalysis Training - Version 3")
    print("=" * 55)
    print("🔧 Improvements in this version:")
    print("   ✅ Reduced learning rate for stability")
    print("   ✅ Early stopping to prevent overfitting")
    print("   ✅ Class weighting for better balance")
    print("   ✅ L2 regularization and gradient clipping")
    print("   ✅ Increased clean text ratio to 60%")
    print("   ✅ Learning rate scheduling")
    print()
    
    start_time = datetime.now()
    
    try:
        # Initialize trainer
        trainer = BalancedSteganalysisTrainer()
        
        # Create balanced dataset
        train_data, val_data = trainer.create_balanced_dataset()
        
        # Train models
        best_val_acc = trainer.train_with_early_stopping(train_data, val_data)
        
        # Calculate training time
        end_time = datetime.now()
        training_time = str(end_time - start_time)
        
        # Save results
        trainer.save_training_summary(best_val_acc, training_time)
        trainer.create_training_plots()
        
        print(f"\n🎉 Balanced Training Complete!")
        print(f"   Best Validation Accuracy: {best_val_acc:.3f}")
        print(f"   Training Time: {training_time}")
        print(f"   Models saved in: {trainer.model_dir}")
        
        print(f"\n📊 Next Steps:")
        print(f"   1. Compare with previous models using comparison script")
        print(f"   2. Test on diverse samples")
        print(f"   3. Deploy if performance is satisfactory")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
