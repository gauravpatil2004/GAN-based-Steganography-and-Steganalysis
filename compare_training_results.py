"""
Comprehensive Comparison: Original vs Enhanced Steganalysis Training

This script compares the performance of the original and enhanced steganalysis models
and provides detailed analysis of the improvements.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from datetime import datetime
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from steganalysis_system import SteganalysisSystem
    print("✅ Successfully imported steganalysis system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def load_both_models():
    """Load both original and enhanced models for comparison."""
    
    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load original models
    models['original'] = SteganalysisSystem(device)
    original_path = os.path.join('models', 'steganalysis')
    if os.path.exists(original_path):
        try:
            models['original'].load_model_weights(original_path)
            print("✅ Loaded original trained models")
        except Exception as e:
            print(f"⚠️ Error loading original models: {e}")
    
    # Load enhanced models
    models['enhanced'] = SteganalysisSystem(device)
    enhanced_path = os.path.join('models', 'steganalysis_enhanced')
    if os.path.exists(enhanced_path):
        try:
            models['enhanced'].load_model_weights(enhanced_path)
            print("✅ Loaded enhanced trained models")
        except Exception as e:
            print(f"⚠️ Error loading enhanced models: {e}")
    
    return models


def create_comprehensive_test_data(num_samples=200):
    """Create diverse test data for comprehensive evaluation."""
    
    print(f"📊 Creating comprehensive test dataset ({num_samples} samples)...")
    
    images = []
    labels = []
    texts = []
    metadata = []
    
    # Different categories of test data
    categories = [
        ('short_plain', 25),      # Short plain text
        ('long_plain', 25),       # Long plain text  
        ('encrypted', 25),        # Encrypted-like text
        ('mixed', 25),           # Mixed content
        ('clean', 100)           # Clean images
    ]
    
    for category, count in categories:
        for i in range(count):
            if category == 'clean':
                # Clean images
                image = torch.randn(3, 32, 32)
                images.append(image)
                labels.append(False)
                texts.append("")
                metadata.append({'category': 'clean', 'length': 0})
                
            else:
                # Create steganographic images
                base_image = torch.randn(3, 32, 32)
                
                # Generate text based on category
                if category == 'short_plain':
                    words = ['secret', 'message', 'hidden', 'text', 'hello', 'world']
                    text = ' '.join(np.random.choice(words, np.random.randint(2, 4)))
                elif category == 'long_plain':
                    words = ['this', 'is', 'a', 'longer', 'secret', 'message', 'with', 
                            'more', 'words', 'and', 'content', 'hidden', 'inside']
                    text = ' '.join(np.random.choice(words, np.random.randint(8, 15)))
                elif category == 'encrypted':
                    import string
                    length = np.random.randint(15, 35)
                    chars = list(string.ascii_letters + string.digits)
                    text = ''.join(np.random.choice(chars, length))
                elif category == 'mixed':
                    if np.random.random() > 0.5:
                        text = f"Message {i}: " + "secret data " * np.random.randint(2, 5)
                    else:
                        chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        text = ''.join(np.random.choice(chars, np.random.randint(10, 25)))
                
                # Simulate steganographic embedding
                noise_strength = np.random.uniform(0.005, 0.02)  # Variable noise
                noise = torch.randn_like(base_image) * noise_strength
                stego_image = base_image + noise
                
                images.append(stego_image)
                labels.append(True)
                texts.append(text)
                metadata.append({'category': category, 'length': len(text)})
    
    # Convert to tensors and shuffle
    indices = np.random.permutation(len(images))
    
    shuffled_images = torch.stack([images[i] for i in indices])
    shuffled_labels = [labels[i] for i in indices]
    shuffled_texts = [texts[i] for i in indices]
    shuffled_metadata = [metadata[i] for i in indices]
    
    print(f"✅ Created diverse test dataset:")
    category_counts = {}
    for meta in shuffled_metadata:
        cat = meta['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in category_counts.items():
        print(f"   {cat}: {count} samples")
    
    return shuffled_images, shuffled_labels, shuffled_texts, shuffled_metadata


def evaluate_model_comprehensive(model, images, labels, texts, metadata, model_name):
    """Comprehensive evaluation of a single model."""
    
    print(f"\n🔍 Evaluating {model_name} model...")
    
    # Run analysis
    results = model.batch_analysis(images, texts)
    
    # Extract predictions
    predictions = [r.has_hidden_text for r in results]
    confidences = [r.confidence_score for r in results]
    capacities = [r.estimated_capacity for r in results]
    text_types = [r.text_type for r in results]
    
    # Calculate overall metrics
    correct = sum(1 for pred, actual in zip(predictions, labels) if pred == actual)
    accuracy = correct / len(labels)
    
    # Confusion matrix
    tp = sum(1 for i, pred in enumerate(predictions) if pred and labels[i])
    fp = sum(1 for i, pred in enumerate(predictions) if pred and not labels[i])
    tn = sum(1 for i, pred in enumerate(predictions) if not pred and not labels[i])
    fn = sum(1 for i, pred in enumerate(predictions) if not pred and labels[i])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Capacity analysis for steganographic samples
    stego_indices = [i for i, label in enumerate(labels) if label]
    true_capacities = [len(texts[i]) for i in stego_indices]
    est_capacities = [capacities[i] for i in stego_indices]
    
    capacity_mae = np.mean([abs(true - est) for true, est in zip(true_capacities, est_capacities)])
    
    # Category-wise analysis
    category_performance = {}
    for category in ['short_plain', 'long_plain', 'encrypted', 'mixed', 'clean']:
        cat_indices = [i for i, meta in enumerate(metadata) if meta['category'] == category]
        if cat_indices:
            cat_correct = sum(1 for i in cat_indices if predictions[i] == labels[i])
            cat_accuracy = cat_correct / len(cat_indices)
            category_performance[category] = {
                'accuracy': cat_accuracy,
                'samples': len(cat_indices),
                'correct': cat_correct
            }
    
    # Confidence distribution analysis
    stego_confidences = [confidences[i] for i, label in enumerate(labels) if label]
    clean_confidences = [confidences[i] for i, label in enumerate(labels) if not label]
    
    metrics = {
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'capacity_mae': capacity_mae
        },
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'category_performance': category_performance,
        'confidence_stats': {
            'stego_mean': np.mean(stego_confidences),
            'stego_std': np.std(stego_confidences),
            'clean_mean': np.mean(clean_confidences),
            'clean_std': np.std(clean_confidences),
            'separation': abs(np.mean(stego_confidences) - np.mean(clean_confidences))
        },
        'predictions': predictions,
        'confidences': confidences,
        'capacities': capacities,
        'text_types': text_types
    }
    
    return metrics


def create_comparison_visualizations(original_metrics, enhanced_metrics, test_labels, metadata):
    """Create comprehensive comparison visualizations."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall Performance Comparison
    ax1 = plt.subplot(3, 4, 1)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    original_values = [original_metrics['overall']['accuracy'],
                      original_metrics['overall']['precision'],
                      original_metrics['overall']['recall'],
                      original_metrics['overall']['f1_score']]
    enhanced_values = [enhanced_metrics['overall']['accuracy'],
                      enhanced_metrics['overall']['precision'],
                      enhanced_metrics['overall']['recall'],
                      enhanced_metrics['overall']['f1_score']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax1.bar(x - width/2, original_values, width, label='Original', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, enhanced_values, width, label='Enhanced', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence Distribution Comparison
    ax2 = plt.subplot(3, 4, 2)
    
    # Original model confidences
    orig_stego_conf = [original_metrics['confidences'][i] for i, label in enumerate(test_labels) if label]
    orig_clean_conf = [original_metrics['confidences'][i] for i, label in enumerate(test_labels) if not label]
    
    # Enhanced model confidences
    enh_stego_conf = [enhanced_metrics['confidences'][i] for i, label in enumerate(test_labels) if label]
    enh_clean_conf = [enhanced_metrics['confidences'][i] for i, label in enumerate(test_labels) if not label]
    
    ax2.hist(orig_stego_conf, alpha=0.5, label='Original Stego', bins=20, color='red')
    ax2.hist(orig_clean_conf, alpha=0.5, label='Original Clean', bins=20, color='blue')
    ax2.hist(enh_stego_conf, alpha=0.5, label='Enhanced Stego', bins=20, color='darkred', histtype='step', linewidth=2)
    ax2.hist(enh_clean_conf, alpha=0.5, label='Enhanced Clean', bins=20, color='darkblue', histtype='step', linewidth=2)
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Category-wise Performance
    ax3 = plt.subplot(3, 4, 3)
    categories = list(original_metrics['category_performance'].keys())
    orig_cat_acc = [original_metrics['category_performance'][cat]['accuracy'] for cat in categories]
    enh_cat_acc = [enhanced_metrics['category_performance'][cat]['accuracy'] for cat in categories]
    
    x = np.arange(len(categories))
    ax3.bar(x - width/2, orig_cat_acc, width, label='Original', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, enh_cat_acc, width, label='Enhanced', alpha=0.8, color='lightcoral')
    ax3.set_xlabel('Text Categories')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Category-wise Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Capacity Estimation Comparison
    ax4 = plt.subplot(3, 4, 4)
    
    # Get true capacities for steganographic samples
    stego_indices = [i for i, label in enumerate(test_labels) if label]
    true_capacities = [metadata[i].get('length', 0) if metadata[i]['category'] != 'clean' else 0 
                      for i in stego_indices]
    
    orig_capacities = [original_metrics['capacities'][i] for i in stego_indices]
    enh_capacities = [enhanced_metrics['capacities'][i] for i in stego_indices]
    
    if true_capacities:
        # Calculate true capacities properly
        true_caps = []
        orig_caps = []
        enh_caps = []
        
        for i, is_stego in enumerate(test_labels):
            if is_stego and i < len(metadata):
                if 'length' in metadata[i]:
                    true_caps.append(metadata[i]['length'])
                else:
                    # Estimate from text if available
                    text_idx = i
                    if text_idx < len(original_metrics['capacities']):
                        # Use a reasonable estimate
                        true_caps.append(max(10, min(50, len(str(i)))))
                
                if i < len(original_metrics['capacities']):
                    orig_caps.append(original_metrics['capacities'][i])
                    enh_caps.append(enhanced_metrics['capacities'][i])
        
        if len(true_caps) == len(orig_caps):
            ax4.scatter(true_caps, orig_caps, alpha=0.6, label='Original', color='skyblue')
            ax4.scatter(true_caps, enh_caps, alpha=0.6, label='Enhanced', color='lightcoral')
            
            max_cap = max(max(true_caps), max(orig_caps), max(enh_caps))
            ax4.plot([0, max_cap], [0, max_cap], 'r--', alpha=0.5, label='Perfect Estimation')
            
            ax4.set_xlabel('True Capacity')
            ax4.set_ylabel('Estimated Capacity')
            ax4.set_title('Capacity Estimation Accuracy')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    # 5. ROC Curves
    ax5 = plt.subplot(3, 4, 5)
    
    def calculate_roc(confidences, labels):
        sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
        tpr_points = []
        fpr_points = []
        
        tp = fp = 0
        total_pos = sum(labels)
        total_neg = len(labels) - total_pos
        
        for i in sorted_indices:
            if labels[i]:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / total_pos if total_pos > 0 else 0
            fpr = fp / total_neg if total_neg > 0 else 0
            tpr_points.append(tpr)
            fpr_points.append(fpr)
        
        return fpr_points, tpr_points
    
    orig_fpr, orig_tpr = calculate_roc(original_metrics['confidences'], test_labels)
    enh_fpr, enh_tpr = calculate_roc(enhanced_metrics['confidences'], test_labels)
    
    orig_auc = np.trapz(orig_tpr, orig_fpr)
    enh_auc = np.trapz(enh_tpr, enh_fpr)
    
    ax5.plot(orig_fpr, orig_tpr, label=f'Original (AUC={orig_auc:.3f})', linewidth=2, color='skyblue')
    ax5.plot(enh_fpr, enh_tpr, label=f'Enhanced (AUC={enh_auc:.3f})', linewidth=2, color='lightcoral')
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training Improvement Summary
    ax6 = plt.subplot(3, 4, 6)
    ax6.axis('off')
    
    # Calculate improvements
    acc_improvement = enhanced_metrics['overall']['accuracy'] - original_metrics['overall']['accuracy']
    prec_improvement = enhanced_metrics['overall']['precision'] - original_metrics['overall']['precision']
    recall_improvement = enhanced_metrics['overall']['recall'] - original_metrics['overall']['recall']
    mae_improvement = original_metrics['overall']['capacity_mae'] - enhanced_metrics['overall']['capacity_mae']
    
    summary_text = f"""
Training Improvement Summary

Accuracy:    {original_metrics['overall']['accuracy']:.1%} → {enhanced_metrics['overall']['accuracy']:.1%}
             {acc_improvement:+.1%} improvement

Precision:   {original_metrics['overall']['precision']:.1%} → {enhanced_metrics['overall']['precision']:.1%}
             {prec_improvement:+.1%} improvement

Recall:      {original_metrics['overall']['recall']:.1%} → {enhanced_metrics['overall']['recall']:.1%}
             {recall_improvement:+.1%} improvement

Capacity MAE: {original_metrics['overall']['capacity_mae']:.1f} → {enhanced_metrics['overall']['capacity_mae']:.1f}
              {mae_improvement:+.1f} chars improvement

Confidence Separation:
Original: {original_metrics['confidence_stats']['separation']:.3f}
Enhanced: {enhanced_metrics['confidence_stats']['separation']:.3f}

AUC Score:
Original: {orig_auc:.3f}
Enhanced: {enh_auc:.3f}
"""
    
    ax6.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            transform=ax6.transAxes)
    
    # 7-12. Additional detailed plots
    # Add more specific analysis plots for remaining subplots
    
    plt.tight_layout()
    plt.savefig('comprehensive_steganalysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison function."""
    
    print("🔬 Comprehensive Steganalysis Model Comparison")
    print("=" * 55)
    
    # Load models
    models = load_both_models()
    
    # Create comprehensive test data
    test_images, test_labels, test_texts, test_metadata = create_comprehensive_test_data(200)
    
    # Evaluate both models
    original_metrics = evaluate_model_comprehensive(
        models['original'], test_images, test_labels, test_texts, test_metadata, "Original"
    )
    
    enhanced_metrics = evaluate_model_comprehensive(
        models['enhanced'], test_images, test_labels, test_texts, test_metadata, "Enhanced"
    )
    
    # Display comparison
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE COMPARISON RESULTS")
    print("="*60)
    
    print(f"\n🎯 Overall Performance:")
    print(f"                    Original    Enhanced    Improvement")
    print(f"   Accuracy:        {original_metrics['overall']['accuracy']:8.1%}    {enhanced_metrics['overall']['accuracy']:8.1%}    {enhanced_metrics['overall']['accuracy'] - original_metrics['overall']['accuracy']:+8.1%}")
    print(f"   Precision:       {original_metrics['overall']['precision']:8.1%}    {enhanced_metrics['overall']['precision']:8.1%}    {enhanced_metrics['overall']['precision'] - original_metrics['overall']['precision']:+8.1%}")
    print(f"   Recall:          {original_metrics['overall']['recall']:8.1%}    {enhanced_metrics['overall']['recall']:8.1%}    {enhanced_metrics['overall']['recall'] - original_metrics['overall']['recall']:+8.1%}")
    print(f"   F1-Score:        {original_metrics['overall']['f1_score']:8.3f}    {enhanced_metrics['overall']['f1_score']:8.3f}    {enhanced_metrics['overall']['f1_score'] - original_metrics['overall']['f1_score']:+8.3f}")
    print(f"   Capacity MAE:    {original_metrics['overall']['capacity_mae']:8.1f}    {enhanced_metrics['overall']['capacity_mae']:8.1f}    {original_metrics['overall']['capacity_mae'] - enhanced_metrics['overall']['capacity_mae']:+8.1f}")
    
    print(f"\n🎛️ Category-wise Performance:")
    for category in original_metrics['category_performance']:
        orig_acc = original_metrics['category_performance'][category]['accuracy']
        enh_acc = enhanced_metrics['category_performance'][category]['accuracy']
        improvement = enh_acc - orig_acc
        print(f"   {category:12s}: {orig_acc:8.1%} → {enh_acc:8.1%} ({improvement:+.1%})")
    
    # Create visualizations
    create_comparison_visualizations(original_metrics, enhanced_metrics, test_labels, test_metadata)
    
    # Save detailed results
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': len(test_images),
        'original_metrics': original_metrics,
        'enhanced_metrics': enhanced_metrics,
        'improvements': {
            'accuracy': enhanced_metrics['overall']['accuracy'] - original_metrics['overall']['accuracy'],
            'precision': enhanced_metrics['overall']['precision'] - original_metrics['overall']['precision'],
            'recall': enhanced_metrics['overall']['recall'] - original_metrics['overall']['recall'],
            'f1_score': enhanced_metrics['overall']['f1_score'] - original_metrics['overall']['f1_score'],
            'capacity_mae': original_metrics['overall']['capacity_mae'] - enhanced_metrics['overall']['capacity_mae']
        }
    }
    
    # Save to file (excluding non-serializable objects)
    serializable_results = {
        'timestamp': comparison_results['timestamp'],
        'test_samples': comparison_results['test_samples'],
        'improvements': comparison_results['improvements'],
        'original_overall': original_metrics['overall'],
        'enhanced_overall': enhanced_metrics['overall'],
        'original_categories': original_metrics['category_performance'],
        'enhanced_categories': enhanced_metrics['category_performance']
    }
    
    with open('steganalysis_comparison_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to:")
    print(f"   📊 steganalysis_comparison_results.json")
    print(f"   📈 comprehensive_steganalysis_comparison.png")
    
    print(f"\n🎉 Comparison completed!")
    print(f"Enhanced training {'achieved significant improvements!' if enhanced_metrics['overall']['accuracy'] > original_metrics['overall']['accuracy'] else 'shows baseline performance.'}")


if __name__ == "__main__":
    main()
