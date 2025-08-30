"""
Text-Based Steganalysis Training - Simplified Version

This script trains text-based steganalysis models specifically for text steganography detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter

class TextSteganalysisDetector(nn.Module):
    """Simple neural network for text-based steganalysis."""
    
    def __init__(self, input_dim=100, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class TextCapacityEstimator(nn.Module):
    """Neural network for estimating hidden message capacity."""
    
    def __init__(self, input_dim=100, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class TextSteganalysisTrainer:
    """Trainer for text-based steganalysis."""
    
    def __init__(self, model_dir='models/text_steganalysis_balanced'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Training parameters - conservative for stability
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.epochs = 25
        self.patience = 5
        
        # Data parameters
        self.num_samples = 2000
        self.clean_ratio = 0.6
        
        # Feature extraction parameters
        self.feature_dim = 100
        
        # Training history
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def extract_text_features(self, texts):
        """Extract features from text for analysis."""
        features = []
        
        for text in texts:
            # Statistical features
            text_len = len(text)
            char_counts = Counter(text)
            
            # Basic statistics
            avg_char_freq = np.mean(list(char_counts.values())) if char_counts else 0
            char_entropy = self.calculate_entropy(text)
            
            # Character distribution features
            letter_count = sum(1 for c in text if c.isalpha())
            digit_count = sum(1 for c in text if c.isdigit())
            space_count = text.count(' ')
            punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
            
            # Ratios
            letter_ratio = letter_count / max(text_len, 1)
            digit_ratio = digit_count / max(text_len, 1)
            space_ratio = space_count / max(text_len, 1)
            punct_ratio = punct_count / max(text_len, 1)
            
            # N-gram features (bigrams, trigrams)
            bigram_entropy = self.calculate_ngram_entropy(text, 2)
            trigram_entropy = self.calculate_ngram_entropy(text, 3)
            
            # Suspicious patterns
            repeated_chars = self.count_repeated_patterns(text)
            unicode_chars = sum(1 for c in text if ord(c) > 127)
            zero_width_chars = sum(1 for c in text if c in ['\u200b', '\u200c', '\u200d', '\ufeff'])
            
            # Pattern-based features
            emoji_count = sum(1 for c in text if ord(c) >= 0x1F600 and ord(c) <= 0x1F64F)
            unusual_spacing = self.detect_unusual_spacing(text)
            
            # Compile feature vector (ensure exactly 100 features)
            feature_vector = [
                text_len,
                avg_char_freq,
                char_entropy,
                letter_count, digit_count, space_count, punct_count,
                letter_ratio, digit_ratio, space_ratio, punct_ratio,
                bigram_entropy, trigram_entropy,
                repeated_chars,
                unicode_chars,
                zero_width_chars,
                emoji_count,
                unusual_spacing
            ]
            
            # Add character frequency features (for common ASCII chars)
            common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 .,!?'
            for char in common_chars:
                feature_vector.append(text.lower().count(char))
            
            # Pad or truncate to exactly 100 features
            while len(feature_vector) < self.feature_dim:
                feature_vector.append(0.0)
            feature_vector = feature_vector[:self.feature_dim]
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)

    def calculate_entropy(self, text):
        """Calculate character entropy of text."""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        text_len = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            p = count / text_len
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

    def calculate_ngram_entropy(self, text, n):
        """Calculate n-gram entropy."""
        if len(text) < n:
            return 0.0
        
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        entropy = 0.0
        for count in ngram_counts.values():
            p = count / total_ngrams
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

    def count_repeated_patterns(self, text):
        """Count repeated character patterns."""
        repeated = 0
        i = 0
        while i < len(text) - 1:
            if text[i] == text[i + 1]:
                repeated += 1
                # Skip consecutive repeated chars
                while i < len(text) - 1 and text[i] == text[i + 1]:
                    i += 1
            i += 1
        return repeated

    def detect_unusual_spacing(self, text):
        """Detect unusual spacing patterns."""
        spaces = [i for i, c in enumerate(text) if c == ' ']
        if len(spaces) < 2:
            return 0
        
        # Calculate spacing intervals
        intervals = [spaces[i+1] - spaces[i] for i in range(len(spaces)-1)]
        if not intervals:
            return 0
        
        # Check for very regular or very irregular spacing
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Unusual if very regular (low std) or very irregular (high std)
        regularity_score = std_interval / max(avg_interval, 1)
        return min(regularity_score, 10.0)  # Cap the score

    def generate_simple_steganographic_text(self, message, method):
        """Generate steganographic text using simple methods."""
        
        # Base cover texts
        cover_texts = [
            "This is a normal conversation about daily activities and weekend plans that people typically have.",
            "Regular email content discussing business matters and project updates for the upcoming quarter.",
            "Simple news article excerpt covering local events and community news from around the neighborhood.",
            "Ordinary technical documentation explaining software features and usage guidelines for new users.",
            "Basic recipe instructions for cooking various dishes and meals that families enjoy together.",
            "Casual social media post about travel experiences and recommendations for vacation destinations.",
            "Standard academic essay discussing literature and historical topics from different time periods.",
            "Plain weather report with forecasts and temperature information for the next few days.",
            "Normal chat message between friends about movies and entertainment shows they recently watched.",
            "Regular product description for online shopping and customer reviews of popular items."
        ]
        
        cover_text = np.random.choice(cover_texts)
        
        if method == 'plain':
            # Simple character insertion
            result = ""
            msg_idx = 0
            for i, char in enumerate(cover_text):
                result += char
                if msg_idx < len(message) and i % 5 == 0:  # Every 5th position
                    result += message[msg_idx]
                    msg_idx += 1
            return result
            
        elif method == 'emoji':
            # Emoji encoding
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
            # Zero-width character encoding
            zero_width_chars = ['\u200b', '\u200c', '\u200d']
            
            result = cover_text
            for char in message:
                # Simple encoding: use different zero-width chars for different characters
                ascii_val = ord(char) % len(zero_width_chars)
                result += zero_width_chars[ascii_val]
            
            return result
            
        else:  # mixed
            methods = ['plain', 'emoji', 'unicode']
            chosen = np.random.choice(methods)
            return self.generate_simple_steganographic_text(message, chosen)

    def create_balanced_dataset(self):
        """Create balanced training dataset."""
        
        print(f"📊 Creating balanced dataset ({self.num_samples} samples)...")
        
        num_clean = int(self.num_samples * self.clean_ratio)
        num_stego = self.num_samples - num_clean
        stego_per_category = num_stego // 4
        
        print(f"   Clean samples: {num_clean}")
        print(f"   Steganographic samples: {num_stego} ({stego_per_category} per category)")
        
        texts = []
        labels = []
        capacities = []
        metadata = []
        
        # Generate clean texts
        clean_templates = [
            "This is a simple clean text message about everyday topics.",
            "Normal conversation between colleagues about work projects.",
            "A brief description of weather conditions and outdoor activities.",
            "Regular email content without any hidden information embedded.",
            "Standard news article excerpt from reputable news sources.",
            "Simple recipe instructions for cooking healthy meals at home.",
            "Basic technical documentation for software and applications.",
            "Casual social media post about hobbies and interests.",
            "Ordinary business memo regarding scheduled meetings.",
            "Plain academic text about various educational subjects."
        ]
        
        for i in range(num_clean):
            base_text = np.random.choice(clean_templates)
            # Add variation
            if np.random.random() < 0.3:
                base_text += f" Additional details about topic {i} with more specific information."
            if np.random.random() < 0.2:
                base_text = base_text.split('.')[0] + "."
            
            texts.append(base_text)
            labels.append(0)
            capacities.append(0)
            metadata.append({'category': 'clean', 'length': 0})
        
        # Generate steganographic samples
        methods = ['plain', 'emoji', 'unicode', 'mixed']
        messages = [
            "Secret message",
            "Hidden data",
            "Confidential info",
            "Covert communication",
            "Classified content",
            "Private message",
            "Sensitive information",
            "Restricted data",
            "Internal communication",
            "This is a longer hidden message with more content"
        ]
        
        for method in methods:
            for i in range(stego_per_category):
                # Vary message length
                if np.random.random() < 0.3:
                    message = f"{np.random.choice(messages)} extended content {i}"
                elif np.random.random() < 0.5:
                    message = f"{np.random.choice(messages)} {i}"
                else:
                    message = np.random.choice(messages[:5])
                
                stego_text = self.generate_simple_steganographic_text(message, method)
                
                texts.append(stego_text)
                labels.append(1)
                capacities.append(len(message))
                metadata.append({
                    'category': method,
                    'length': len(message),
                    'method': method
                })
        
        # Verify balance
        label_counts = Counter(labels)
        print(f"   Final balance: Clean={label_counts[0]}, Stego={label_counts[1]}")
        print(f"   Balance ratio: {label_counts[0]/len(labels):.1%} clean, {label_counts[1]/len(labels):.1%} stego")
        
        # Split into train/validation
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
        
        val_label_counts = Counter(val_data['labels'])
        print(f"   Validation balance: Clean={val_label_counts[0]}, Stego={val_label_counts[1]}")
        
        return train_data, val_data

    def train_models(self, train_data, val_data):
        """Train the steganalysis models."""
        
        print(f"🎯 Starting training...")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Epochs: {self.epochs}")
        
        # Extract features
        print("🔧 Extracting features...")
        train_features = self.extract_text_features(train_data['texts'])
        val_features = self.extract_text_features(val_data['texts'])
        
        print(f"   Feature shape: {train_features.shape}")
        
        # Create weighted sampler for balanced training
        label_counts = Counter(train_data['labels'])
        total_samples = len(train_data['labels'])
        weights = []
        for label in train_data['labels']:
            weight = total_samples / (len(label_counts) * label_counts[label])
            weights.append(weight)
        
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        # Convert to tensors
        train_X = torch.FloatTensor(train_features).to(self.device)
        train_y = torch.LongTensor(train_data['labels']).to(self.device)
        train_cap = torch.FloatTensor(train_data['capacities']).to(self.device)
        
        val_X = torch.FloatTensor(val_features).to(self.device)
        val_y = torch.LongTensor(val_data['labels']).to(self.device)
        val_cap = torch.FloatTensor(val_data['capacities']).to(self.device)
        
        # Initialize models
        detector = TextSteganalysisDetector(input_dim=self.feature_dim).to(self.device)
        capacity_estimator = TextCapacityEstimator(input_dim=self.feature_dim).to(self.device)
        
        # Optimizers with weight decay
        detector_optimizer = optim.AdamW(detector.parameters(), 
                                       lr=self.learning_rate, 
                                       weight_decay=0.01)
        capacity_optimizer = optim.AdamW(capacity_estimator.parameters(), 
                                       lr=self.learning_rate, 
                                       weight_decay=0.01)
        
        # Loss functions
        pos_weight = torch.tensor([len([l for l in train_data['labels'] if l == 0]) / 
                                 len([l for l in train_data['labels'] if l == 1])]).to(self.device)
        detection_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        capacity_criterion = nn.MSELoss()
        
        # Learning rate schedulers
        detector_scheduler = optim.lr_scheduler.ReduceLROnPlateau(detector_optimizer, 
                                                                mode='max', 
                                                                patience=3, 
                                                                factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n🚀 Training started...")
        
        for epoch in range(self.epochs):
            # Training phase
            detector.train()
            capacity_estimator.train()
            
            # Create random batches (weighted sampling handled by sampler)
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
                
                # Additional metrics
                val_preds_np = val_preds.cpu().numpy()
                val_y_np = val_y.cpu().numpy()
                
                val_precision = precision_score(val_y_np, val_preds_np, zero_division=0)
                val_recall = recall_score(val_y_np, val_preds_np, zero_division=0)
                val_f1 = f1_score(val_y_np, val_preds_np, zero_division=0)
            
            # Update learning rate
            detector_scheduler.step(val_acc)
            
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
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best models
                torch.save(detector.state_dict(), 
                          os.path.join(self.model_dir, 'text_detector.pth'))
                torch.save(capacity_estimator.state_dict(), 
                          os.path.join(self.model_dir, 'text_capacity_estimator.pth'))
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
        """Save training summary."""
        
        summary = {
            'training_type': 'Text-Based Steganalysis Training',
            'timestamp': datetime.now().isoformat(),
            'training_time': train_time,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'samples': self.num_samples,
                'clean_ratio': self.clean_ratio,
                'feature_dim': self.feature_dim
            },
            'final_metrics': {
                'validation_accuracy': val_acc,
                'best_epoch': self.training_history['val_acc'].index(max(self.training_history['val_acc'])) + 1,
                'final_precision': self.training_history['val_precision'][-1],
                'final_recall': self.training_history['val_recall'][-1],
                'final_f1': self.training_history['val_f1'][-1]
            },
            'training_history': self.training_history
        }
        
        with open('text_steganalysis_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Training summary saved to 'text_steganalysis_training_summary.json'")

    def create_training_plots(self):
        """Create training visualization plots."""
        
        plt.figure(figsize=(15, 10))
        
        # Accuracy plot
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
        
        # Loss plot
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
        
        # Metrics plot
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
        
        # Final metrics bar chart
        plt.subplot(2, 3, 4)
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
        
        # Add value labels
        for bar, value in zip(bars, final_metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Training summary
        plt.subplot(2, 3, 5)
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
        
        # Training progress
        plt.subplot(2, 3, 6)
        improvement = self.training_history['val_acc'][-1] - self.training_history['val_acc'][0]
        plt.text(0.1, 0.8, f'Initial Val Acc: {self.training_history["val_acc"][0]:.3f}', fontsize=12)
        plt.text(0.1, 0.7, f'Final Val Acc: {self.training_history["val_acc"][-1]:.3f}', fontsize=12)
        plt.text(0.1, 0.6, f'Improvement: {improvement:+.3f}', fontsize=12, 
                color='green' if improvement > 0 else 'red')
        plt.text(0.1, 0.4, f'Feature Dimension: {self.feature_dim}', fontsize=12)
        plt.text(0.1, 0.3, f'Architecture: Text-based Neural Network', fontsize=12)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Training Progress')
        
        plt.tight_layout()
        plt.savefig('text_steganalysis_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved to 'text_steganalysis_training_curves.png'")

def main():
    """Main training function."""
    
    print("🎯 Text-Based Steganalysis Training")
    print("=" * 45)
    print("🔧 Features:")
    print("   ✅ Text-specific feature extraction")
    print("   ✅ Balanced training with class weighting")
    print("   ✅ Early stopping and regularization")
    print("   ✅ Comprehensive text pattern analysis")
    print("   ✅ Multiple steganographic methods")
    print()
    
    start_time = datetime.now()
    
    try:
        # Initialize trainer
        trainer = TextSteganalysisTrainer()
        
        # Create dataset
        train_data, val_data = trainer.create_balanced_dataset()
        
        # Train models
        best_val_acc = trainer.train_models(train_data, val_data)
        
        # Calculate training time
        end_time = datetime.now()
        training_time = str(end_time - start_time)
        
        # Save results
        trainer.save_training_summary(best_val_acc, training_time)
        trainer.create_training_plots()
        
        print(f"\n🎉 Text-Based Steganalysis Training Complete!")
        print(f"   Best Validation Accuracy: {best_val_acc:.3f}")
        print(f"   Training Time: {training_time}")
        print(f"   Models saved in: {trainer.model_dir}")
        
        print(f"\n📊 Next Steps:")
        print(f"   1. Test the models on diverse text samples")
        print(f"   2. Compare with previous image-based models")
        print(f"   3. Create text-based web interface")
        print(f"   4. Analyze feature importance")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
