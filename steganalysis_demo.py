"""
Steganalysis System Demo

This script demonstrates the complete steganalysis detection system,
showing how to detect hidden text in images and analyze the results.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os
import sys
from datetime import datetime
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from steganalysis_system import SteganalysisSystem, SteganalysisResult
from text_gan_architecture import TextSteganoGenerator, TextExtractor
from text_processor import TextProcessor


class SteganalysisDemo:
    """Demonstration of the steganalysis detection system."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Initialize systems
        self.steganalysis = SteganalysisSystem(device)
        self.setup_steganography_system()
        
        # Demo data
        self.demo_texts = [
            "This is a secret message hidden in the image",
            "Hello world from steganography",
            "aB3xY9zK2mN8qP",  # Encrypted-like
            "The quick brown fox jumps over the lazy dog",
            "x1Y9zA3bC7dE5fG",  # More encrypted-like
            "",  # Empty for clean images
        ]
        
        self.demo_results = []
    
    def setup_steganography_system(self):
        """Setup steganography system for generating test images."""
        try:
            self.text_processor = TextProcessor()
            self.generator = TextSteganoGenerator().to(self.device)
            self.extractor = TextExtractor().to(self.device)
            
            # Try to load pre-trained weights
            model_path = os.path.join('models', 'best_stego_model.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'generator' in checkpoint:
                    self.generator.load_state_dict(checkpoint['generator'])
                if 'extractor' in checkpoint:
                    self.extractor.load_state_dict(checkpoint['extractor'])
                print("✅ Loaded pre-trained steganography models")
            else:
                print("⚠️ Using untrained models for demonstration")
                
        except Exception as e:
            print(f"⚠️ Error loading steganography models: {e}")
            self.generator = None
            self.extractor = None
    
    def create_test_images(self) -> Tuple[torch.Tensor, List[str], List[bool]]:
        """Create test images for steganalysis demonstration."""
        
        print("Creating test images for steganalysis demo...")
        
        images = []
        texts = []
        has_text_labels = []
        
        for i, text in enumerate(self.demo_texts):
            # Create cover image
            cover_image = torch.randn(1, 3, 32, 32).to(self.device)
            
            if text and self.generator is not None:
                try:
                    # Create steganographic image
                    text_tokens = self.text_processor.encode_text(text)
                    text_embedding = self.text_processor.tokens_to_embedding(text_tokens).to(self.device)
                    
                    with torch.no_grad():
                        stego_image = self.generator(cover_image, text_embedding)
                    
                    images.append(stego_image.squeeze(0))
                    texts.append(text)
                    has_text_labels.append(True)
                    
                except Exception as e:
                    print(f"Error creating stego image: {e}")
                    # Fallback: slightly modified image
                    noise = torch.randn_like(cover_image) * 0.01
                    images.append((cover_image + noise).squeeze(0))
                    texts.append(text)
                    has_text_labels.append(True)
            else:
                # Clean image
                images.append(cover_image.squeeze(0))
                texts.append("")
                has_text_labels.append(False)
        
        return torch.stack(images), texts, has_text_labels
    
    def run_detection_demo(self) -> List[SteganalysisResult]:
        """Run complete steganalysis detection demonstration."""
        
        print("🔍 Running Steganalysis Detection Demo")
        print("=" * 50)
        
        # Create test images
        test_images, test_texts, ground_truth = self.create_test_images()
        
        # Analyze each image
        results = []
        
        print("\nAnalyzing images...")
        for i, (image, text, true_label) in enumerate(zip(test_images, test_texts, ground_truth)):
            print(f"\n📷 Image {i+1}:")
            print(f"   Ground Truth: {'Has Text' if true_label else 'Clean'}")
            if text:
                print(f"   Hidden Text: '{text[:30]}{'...' if len(text) > 30 else ''}'")
            
            # Run steganalysis
            image_batch = image.unsqueeze(0)
            result = self.steganalysis.analyze_image(image_batch, text if true_label else None)
            
            # Display results
            print(f"   🎯 Detection: {'POSITIVE' if result.has_hidden_text else 'NEGATIVE'}")
            print(f"   📊 Confidence: {result.confidence_score:.3f}")
            print(f"   📏 Est. Capacity: {result.estimated_capacity} chars")
            print(f"   🔤 Text Type: {result.text_type}")
            
            # Check accuracy
            is_correct = result.has_hidden_text == true_label
            print(f"   ✅ Correct: {is_correct}")
            
            results.append(result)
        
        self.demo_results = results
        return results
    
    def visualize_detection_results(self, results: List[SteganalysisResult], 
                                   ground_truth: List[bool]):
        """Visualize steganalysis detection results."""
        
        print("\n📊 Generating Detection Visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Steganalysis Detection Results', fontsize=16)
        
        # 1. Detection Confidence Distribution
        confidences = [r.confidence_score for r in results]
        colors = ['red' if gt else 'blue' for gt in ground_truth]
        
        axes[0, 0].scatter(range(len(confidences)), confidences, c=colors, alpha=0.7)
        axes[0, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Detection Confidence Scores')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Confidence Score')
        
        # Create legend handles separately to avoid matplotlib warning
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', label='Threshold'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Has Text'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Clean')
        ]
        axes[0, 0].legend(handles=legend_elements)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Capacity Estimation
        capacities = [r.estimated_capacity for r in results]
        true_capacities = [len(self.demo_texts[i]) if ground_truth[i] else 0 
                          for i in range(len(results))]
        
        axes[0, 1].scatter(true_capacities, capacities, alpha=0.7)
        max_cap = max(max(capacities), max(true_capacities))
        axes[0, 1].plot([0, max_cap], [0, max_cap], 'r--', alpha=0.5)
        axes[0, 1].set_title('Capacity Estimation Accuracy')
        axes[0, 1].set_xlabel('True Capacity (characters)')
        axes[0, 1].set_ylabel('Estimated Capacity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Text Type Distribution
        type_counts = {}
        for result in results:
            type_counts[result.text_type] = type_counts.get(result.text_type, 0) + 1
        
        axes[1, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Detected Text Type Distribution')
        
        # 4. Performance Metrics
        true_positives = sum(1 for i, r in enumerate(results) 
                           if r.has_hidden_text and ground_truth[i])
        false_positives = sum(1 for i, r in enumerate(results) 
                            if r.has_hidden_text and not ground_truth[i])
        true_negatives = sum(1 for i, r in enumerate(results) 
                           if not r.has_hidden_text and not ground_truth[i])
        false_negatives = sum(1 for i, r in enumerate(results) 
                            if not r.has_hidden_text and ground_truth[i])
        
        confusion_matrix = np.array([[true_negatives, false_positives],
                                   [false_negatives, true_positives]])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', 
                   xticklabels=['Clean', 'Has Text'],
                   yticklabels=['Clean', 'Has Text'],
                   ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('steganalysis_demo_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as 'steganalysis_demo_results.png'")
    
    def analyze_feature_importance(self, results: List[SteganalysisResult]):
        """Analyze which features are most important for detection."""
        
        print("\n🔬 Analyzing Feature Importance...")
        
        # Collect features from all results
        all_features = {}
        for i, result in enumerate(results):
            for feature_name, value in result.features.items():
                if feature_name not in all_features:
                    all_features[feature_name] = []
                all_features[feature_name].append(value)
        
        # Calculate feature statistics
        feature_stats = {}
        for feature_name, values in all_features.items():
            if len(values) > 0:
                feature_stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Display top features
        print("\nTop Feature Statistics:")
        for feature_name, stats in list(feature_stats.items())[:10]:
            print(f"  {feature_name}:")
            print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        return feature_stats
    
    def generate_comprehensive_report(self, results: List[SteganalysisResult], 
                                    ground_truth: List[bool]) -> str:
        """Generate a comprehensive analysis report."""
        
        print("\n📝 Generating Comprehensive Report...")
        
        # Calculate metrics
        total_images = len(results)
        detected_positives = sum(1 for r in results if r.has_hidden_text)
        actual_positives = sum(ground_truth)
        
        true_positives = sum(1 for i, r in enumerate(results) 
                           if r.has_hidden_text and ground_truth[i])
        false_positives = sum(1 for i, r in enumerate(results) 
                            if r.has_hidden_text and not ground_truth[i])
        true_negatives = sum(1 for i, r in enumerate(results) 
                           if not r.has_hidden_text and not ground_truth[i])
        false_negatives = sum(1 for i, r in enumerate(results) 
                            if not r.has_hidden_text and ground_truth[i])
        
        accuracy = (true_positives + true_negatives) / total_images
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Generate report
        report = f"""
# Steganalysis System Demonstration Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
The steganalysis detection system has been successfully demonstrated on {total_images} test images, 
achieving an overall accuracy of {accuracy:.1%} in detecting hidden text.

## Performance Metrics
- **Total Images Analyzed**: {total_images}
- **Actual Images with Hidden Text**: {actual_positives}
- **Detected Images with Hidden Text**: {detected_positives}
- **Accuracy**: {accuracy:.3f} ({accuracy:.1%})
- **Precision**: {precision:.3f} ({precision:.1%})
- **Recall**: {recall:.3f} ({recall:.1%})
- **F1-Score**: {f1_score:.3f}

## Confusion Matrix
|              | Predicted Clean | Predicted Has Text |
|--------------|----------------|--------------------|
| **Actual Clean**     | {true_negatives}              | {false_positives}                  |
| **Actual Has Text**  | {false_negatives}              | {true_positives}                   |

## Detection Analysis
"""
        
        # Add individual results
        for i, (result, gt) in enumerate(zip(results, ground_truth)):
            text_preview = self.demo_texts[i][:20] + "..." if len(self.demo_texts[i]) > 20 else self.demo_texts[i]
            status = "✅ CORRECT" if result.has_hidden_text == gt else "❌ INCORRECT"
            
            report += f"""
### Image {i+1}
- **Ground Truth**: {'Has Text' if gt else 'Clean'}
- **Hidden Text**: "{text_preview}" ({len(self.demo_texts[i])} chars)
- **Detection**: {'POSITIVE' if result.has_hidden_text else 'NEGATIVE'}
- **Confidence**: {result.confidence_score:.3f}
- **Estimated Capacity**: {result.estimated_capacity} characters
- **Text Type**: {result.text_type}
- **Result**: {status}
"""
        
        report += f"""

## Text Type Analysis
"""
        type_counts = {}
        for result in results:
            type_counts[result.text_type] = type_counts.get(result.text_type, 0) + 1
        
        for text_type, count in type_counts.items():
            percentage = count / total_images * 100
            report += f"- **{text_type.title()}**: {count} images ({percentage:.1f}%)\n"
        
        report += f"""

## Capacity Estimation Analysis
"""
        capacity_errors = []
        for i, result in enumerate(results):
            if ground_truth[i]:
                true_cap = len(self.demo_texts[i])
                est_cap = result.estimated_capacity
                error = abs(true_cap - est_cap)
                capacity_errors.append(error)
                
        if capacity_errors:
            avg_error = np.mean(capacity_errors)
            max_error = np.max(capacity_errors)
            report += f"- **Average Capacity Error**: {avg_error:.1f} characters\n"
            report += f"- **Maximum Capacity Error**: {max_error} characters\n"
        
        report += f"""

## Conclusions
{'✅' if accuracy > 0.7 else '⚠️'} The steganalysis system demonstrates {'good' if accuracy > 0.7 else 'moderate'} performance in detecting hidden text.
{'✅' if precision > 0.8 else '⚠️'} Precision is {'high' if precision > 0.8 else 'moderate'}, indicating {'few' if precision > 0.8 else 'some'} false positives.
{'✅' if recall > 0.8 else '⚠️'} Recall is {'high' if recall > 0.8 else 'moderate'}, indicating {'few' if recall > 0.8 else 'some'} missed detections.

## Recommendations
1. {'✅ System is production-ready' if accuracy > 0.8 else '⚠️ Consider additional training with larger dataset'}
2. {'✅ Capacity estimation is reliable' if not capacity_errors or avg_error < 5 else '⚠️ Improve capacity estimation accuracy'}
3. {'✅ Text type classification is working well' if len(type_counts) > 1 else '⚠️ Enhance text type detection'}
"""
        
        return report
    
    def save_demo_results(self, results: List[SteganalysisResult], 
                         ground_truth: List[bool]):
        """Save demonstration results to files."""
        
        # Save results as JSON
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'results': []
        }
        
        for i, (result, gt) in enumerate(zip(results, ground_truth)):
            results_data['results'].append({
                'image_id': i,
                'ground_truth': gt,
                'hidden_text': self.demo_texts[i],
                'detection': {
                    'has_hidden_text': result.has_hidden_text,
                    'confidence_score': result.confidence_score,
                    'estimated_capacity': result.estimated_capacity,
                    'text_type': result.text_type
                },
                'features': result.features
            })
        
        with open('steganalysis_demo_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report(results, ground_truth)
        with open('steganalysis_demo_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("💾 Results saved:")
        print("  - steganalysis_demo_results.json")
        print("  - steganalysis_demo_report.md")
    
    def run_complete_demo(self):
        """Run the complete steganalysis demonstration."""
        
        print("🚀 Starting Complete Steganalysis Demonstration")
        print("=" * 60)
        
        # Run detection demo
        results = self.run_detection_demo()
        
        # Get ground truth
        _, _, ground_truth = self.create_test_images()
        
        # Visualize results
        self.visualize_detection_results(results, ground_truth)
        
        # Analyze features
        self.analyze_feature_importance(results)
        
        # Generate and display report
        report = self.generate_comprehensive_report(results, ground_truth)
        print(report)
        
        # Save results
        self.save_demo_results(results, ground_truth)
        
        print("\n🎉 Steganalysis Demonstration Complete!")
        print("\nFiles generated:")
        print("  📊 steganalysis_demo_results.png - Visualization")
        print("  📋 steganalysis_demo_results.json - Raw results")
        print("  📝 steganalysis_demo_report.md - Comprehensive report")


def main():
    """Main demonstration function."""
    
    print("🔍 Steganalysis Detection System Demo")
    print("====================================")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize demo
    demo = SteganalysisDemo(device)
    
    # Run complete demonstration
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
