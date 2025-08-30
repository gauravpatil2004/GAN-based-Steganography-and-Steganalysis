"""
Simple Balanced Training - Quick Implementation

This script provides a simple, self-contained training approach that addresses
the overfitting issues without complex dependencies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append('src')

class SimpleTextDetector(nn.Module):
    """Simple neural network for text detection."""
    
    def __init__(self, input_size=50):
        super(SimpleTextDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleCapacityEstimator(nn.Module):
    """Simple neural network for capacity estimation."""
    
    def __init__(self, input_size=50):
        super(SimpleCapacityEstimator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def extract_simple_features(text):
    """Extract simple features from text."""
    
    if not text or len(text) == 0:
        return [0.0] * 50
    
    features = []
    
    # Basic statistics
    features.append(len(text))  # Text length
    features.append(len(text.split()))  # Word count
    features.append(len(set(text.lower())))  # Unique chars
    features.append(text.count(' ') / len(text))  # Space ratio
    features.append(text.count('\n') / len(text))  # Newline ratio
    
    # Character frequency analysis
    char_counts = {}
    for char in text.lower():
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Entropy
    entropy = 0
    for count in char_counts.values():
        p = count / len(text)
        if p > 0:
            entropy -= p * np.log2(p)
    features.append(entropy)
    
    # Special character counts
    features.append(sum(1 for c in text if ord(c) > 127) / len(text))  # Non-ASCII ratio
    features.append(sum(1 for c in text if ord(c) > 255) / len(text))  # Unicode ratio
    
    # Zero-width characters
    zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff']
    features.append(sum(text.count(zw) for zw in zero_width))  # Zero-width count
    
    # Emoji detection (rough)
    emoji_count = sum(1 for c in text if ord(c) > 0x1F000)
    features.append(emoji_count / len(text))  # Emoji ratio
    
    # Punctuation
    punct = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    features.append(sum(text.count(p) for p in punct) / len(text))  # Punct ratio
    
    # Vowel/consonant
    vowels = 'aeiouAEIOU'
    features.append(sum(text.count(v) for v in vowels) / len(text))  # Vowel ratio
    
    # Repetition patterns
    features.append(len(text) / len(set(text)) if len(set(text)) > 0 else 0)  # Repetition
    
    # Average word length
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    features.append(avg_word_len)
    
    # Digit ratio
    features.append(sum(c.isdigit() for c in text) / len(text))  # Digit ratio
    
    # Uppercase ratio
    features.append(sum(c.isupper() for c in text) / len(text))  # Upper ratio
    
    # Pad to 50 features
    while len(features) < 50:
        features.append(0.0)
    
    return features[:50]

def generate_steganographic_text(message, method='simple'):
    """Generate simple steganographic text."""
    
    cover_texts = [
        "This is a normal business email discussing project timelines and deliverables.",
        "Regular conversation about weekend plans and family activities.",
        "Standard news article covering local events and community updates.",
        "Ordinary technical documentation explaining software features.",
        "Basic recipe instructions for cooking traditional dishes.",
        "Casual social media post about travel experiences.",
        "Simple weather report with forecasts and temperatures.",
        "Normal chat between friends about movies and books.",
        "Regular product review for online shopping platform.",
        "Standard academic essay about historical events."
    ]
    
    cover = np.random.choice(cover_texts)
    
    if method == 'emoji':
        # Add emoji encoding
        emoji_map = '😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟😠😡😢😣😤😥😦😧😨😩😪😫😬😭😮😯😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿🙀'
        encoded = ""
        for char in message:
            if char.isalnum():
                encoded += emoji_map[ord(char.lower()) % len(emoji_map)]
        return cover + " " + encoded
        
    elif method == 'unicode':
        # Add zero-width characters
        result = ""
        msg_idx = 0
        for i, char in enumerate(cover):
            result += char
            if msg_idx < len(message) and i % 5 == 0:
                # Encode using zero-width chars
                result += '\u200b' if ord(message[msg_idx]) % 2 == 0 else '\u200c'
                msg_idx += 1
        return result
        
    else:  # simple insertion
        result = ""
        msg_idx = 0
        for i, char in enumerate(cover):
            result += char
            if msg_idx < len(message) and i % 4 == 0:
                result += message[msg_idx]
                msg_idx += 1
        return result

def create_training_data(num_samples=1000):
    """Create balanced training dataset."""
    
    print(f"📊 Creating balanced dataset ({num_samples} samples)...")
    
    texts = []
    labels = []
    capacities = []
    
    # 60% clean text
    num_clean = int(num_samples * 0.6)
    num_stego = num_samples - num_clean
    
    print(f"   Clean: {num_clean}, Steganographic: {num_stego}")
    
    # Generate clean texts
    clean_templates = [
        "Regular business communication about quarterly reports.",
        "Normal conversation between colleagues about project status.",
        "Standard email regarding meeting schedules and agendas.",
        "Ordinary social media post about daily activities.",
        "Basic news article covering local community events.",
        "Simple technical documentation for software usage.",
        "Regular academic essay about literature and history.",
        "Normal chat message about entertainment and hobbies.",
        "Standard product description for online marketplace.",
        "Ordinary weather forecast with temperature updates."
    ]
    
    for i in range(num_clean):
        base = np.random.choice(clean_templates)
        # Add variation
        if np.random.random() < 0.3:
            base += f" Additional details about topic {i} with more information."
        texts.append(base)
        labels.append(0)  # Clean
        capacities.append(0)
    
    # Generate steganographic texts
    messages = [
        "secret", "hidden", "private", "confidential", "classified",
        "covert message", "hidden data", "secret info", "private communication",
        "confidential document", "classified information", "hidden message here"
    ]
    
    methods = ['simple', 'emoji', 'unicode']
    
    for i in range(num_stego):
        message = np.random.choice(messages)
        method = np.random.choice(methods)
        
        # Add variation to message length
        if np.random.random() < 0.3:
            message += f" {i}"
        
        stego_text = generate_steganographic_text(message, method)
        texts.append(stego_text)
        labels.append(1)  # Steganographic
        capacities.append(len(message))
    
    return texts, labels, capacities

def train_simple_models():
    """Train simple models with balanced approach."""
    
    print("🎯 Simple Balanced Training")
    print("=" * 40)
    
    # Parameters
    learning_rate = 0.0001  # Conservative
    batch_size = 32
    epochs = 20
    
    # Create data
    texts, labels, capacities = create_training_data(1000)
    
    # Extract features
    print("🔧 Extracting features...")
    features = []
    for text in texts:
        features.append(extract_simple_features(text))
    
    features = np.array(features)
    labels = np.array(labels)
    capacities = np.array(capacities)
    
    # Split train/validation
    split_idx = int(0.8 * len(features))
    train_X, val_X = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]
    train_cap, val_cap = capacities[:split_idx], capacities[split_idx:]
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_X = torch.FloatTensor(train_X).to(device)
    train_y = torch.FloatTensor(train_y).to(device)
    train_cap = torch.FloatTensor(train_cap).to(device)
    
    val_X = torch.FloatTensor(val_X).to(device)
    val_y = torch.FloatTensor(val_y).to(device)
    val_cap = torch.FloatTensor(val_cap).to(device)
    
    # Initialize models
    detector = SimpleTextDetector().to(device)
    capacity_estimator = SimpleCapacityEstimator().to(device)
    
    # Optimizers
    det_optimizer = optim.AdamW(detector.parameters(), lr=learning_rate, weight_decay=0.01)
    cap_optimizer = optim.AdamW(capacity_estimator.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Loss functions
    detection_criterion = nn.BCELoss()
    capacity_criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_acc': [], 'val_acc': [], 'val_precision': [], 
        'val_recall': [], 'val_f1': []
    }
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    print(f"\n🚀 Training started...")
    
    for epoch in range(epochs):
        # Training
        detector.train()
        capacity_estimator.train()
        
        # Create batches
        num_batches = len(train_X) // batch_size
        indices = torch.randperm(len(train_X))
        
        train_correct = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = train_X[batch_indices]
            batch_y = train_y[batch_indices]
            batch_cap = train_cap[batch_indices]
            
            # Detection training
            det_optimizer.zero_grad()
            det_pred = detector(batch_X).squeeze()
            det_loss = detection_criterion(det_pred, batch_y)
            det_loss.backward()
            det_optimizer.step()
            
            # Capacity training (only on stego samples)
            stego_mask = batch_y == 1
            if stego_mask.sum() > 0:
                cap_optimizer.zero_grad()
                cap_pred = capacity_estimator(batch_X[stego_mask]).squeeze()
                cap_loss = capacity_criterion(cap_pred, batch_cap[stego_mask])
                cap_loss.backward()
                cap_optimizer.step()
            
            # Track accuracy
            train_correct += ((det_pred > 0.5) == batch_y).sum().item()
        
        train_acc = train_correct / len(train_X)
        
        # Validation
        detector.eval()
        capacity_estimator.eval()
        
        with torch.no_grad():
            val_pred = detector(val_X).squeeze()
            val_binary = (val_pred > 0.5).cpu().numpy()
            val_y_np = val_y.cpu().numpy()
            
            val_acc = accuracy_score(val_y_np, val_binary)
            val_precision = precision_score(val_y_np, val_binary, zero_division=0)
            val_recall = recall_score(val_y_np, val_binary, zero_division=0)
            val_f1 = f1_score(val_y_np, val_binary, zero_division=0)
        
        # Track history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"   Epoch {epoch+1:2d}/{epochs}: Train Acc: {train_acc:.3f}, "
              f"Val Acc: {val_acc:.3f}, Prec: {val_precision:.3f}, "
              f"Rec: {val_recall:.3f}, F1: {val_f1:.3f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save models
            os.makedirs('models/simple_balanced', exist_ok=True)
            torch.save(detector.state_dict(), 'models/simple_balanced/detector.pth')
            torch.save(capacity_estimator.state_dict(), 'models/simple_balanced/capacity_estimator.pth')
            print(f"     ✅ New best: {val_acc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"     🛑 Early stopping")
                break
    
    # Save results
    results = {
        'training_type': 'Simple Balanced Training',
        'timestamp': datetime.now().isoformat(),
        'best_validation_accuracy': best_val_acc,
        'final_metrics': {
            'accuracy': history['val_acc'][-1],
            'precision': history['val_precision'][-1],
            'recall': history['val_recall'][-1],
            'f1': history['val_f1'][-1]
        },
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epoch + 1,
            'samples': len(texts)
        },
        'history': history
    }
    
    with open('simple_balanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.3f}")
    print(f"   Final metrics: Acc={history['val_acc'][-1]:.3f}, "
          f"Prec={history['val_precision'][-1]:.3f}, "
          f"Rec={history['val_recall'][-1]:.3f}, F1={history['val_f1'][-1]:.3f}")
    
    return best_val_acc, results

def quick_test():
    """Quick test of the trained models."""
    
    print(f"\n🧪 Quick Test of Trained Models")
    print("=" * 35)
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = SimpleTextDetector().to(device)
    
    try:
        detector.load_state_dict(torch.load('models/simple_balanced/detector.pth', map_location=device))
        print("✅ Loaded trained detector")
    except:
        print("❌ Could not load trained model")
        return
    
    detector.eval()
    
    # Test samples
    test_texts = [
        "This is a normal business email about project updates.",  # Clean
        "Regular conversation about weekend plans and activities.",  # Clean
        generate_steganographic_text("secret message", "simple"),  # Stego
        generate_steganographic_text("hidden data", "emoji"),      # Stego
        generate_steganographic_text("confidential", "unicode")    # Stego
    ]
    
    labels = ["Clean", "Clean", "Stego", "Stego", "Stego"]
    
    print(f"\nTesting {len(test_texts)} samples:")
    
    correct = 0
    for i, (text, true_label) in enumerate(zip(test_texts, labels)):
        features = extract_simple_features(text)
        features_tensor = torch.FloatTensor([features]).to(device)
        
        with torch.no_grad():
            prob = detector(features_tensor).item()
            pred_label = "Stego" if prob > 0.5 else "Clean"
            is_correct = pred_label == true_label
            
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"   {i+1}. {true_label:5} → {pred_label:5} ({prob:.3f}) {status}")
    
    accuracy = correct / len(test_texts)
    print(f"\nQuick test accuracy: {accuracy:.1%} ({correct}/{len(test_texts)})")

if __name__ == "__main__":
    try:
        # Train models
        best_acc, results = train_simple_models()
        
        # Quick test
        quick_test()
        
        print(f"\n🎉 Simple Balanced Training Complete!")
        print(f"   Results saved to 'simple_balanced_results.json'")
        print(f"   Models saved to 'models/simple_balanced/'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
