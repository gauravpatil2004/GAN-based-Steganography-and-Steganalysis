"""
Quick Model Comparison - Simplified Version

This script provides a quick comparison between the original and enhanced models.
"""

import torch
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

def load_models():
    """Load both model sets."""
    try:
        from steganalysis_system import SteganalysisSystem
        
        # Initialize systems
        original_system = SteganalysisSystem()
        enhanced_system = SteganalysisSystem()
        
        # Load models
        original_path = os.path.join('models', 'steganalysis')
        enhanced_path = os.path.join('models', 'steganalysis_enhanced')
        
        if os.path.exists(original_path):
            original_system.load_models(original_path)
            print("✅ Loaded original models")
        else:
            print("❌ Original models not found")
            return None, None
            
        if os.path.exists(enhanced_path):
            enhanced_system.load_models(enhanced_path)
            print("✅ Loaded enhanced models")
        else:
            print("❌ Enhanced models not found")
            return None, None
            
        return original_system, enhanced_system
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

def quick_test():
    """Run a quick test comparison."""
    
    print("🔬 Quick Model Comparison")
    print("=" * 40)
    
    # Load models
    original_system, enhanced_system = load_models()
    
    if not original_system or not enhanced_system:
        print("Cannot proceed without both model sets")
        return
    
    # Test samples
    test_texts = [
        "This is a clean text sample with no hidden content.",
        "A short hidden message for testing detection capabilities.",
        "This is a longer hidden message that contains more substantial content for comprehensive testing of the steganalysis system's detection capabilities.",
        "🔒ENCRYPTED🔒: This represents encrypted steganographic content that should be detected by the system."
    ]
    
    test_labels = ["clean", "short_stego", "long_stego", "encrypted_stego"]
    
    print(f"\n📊 Testing {len(test_texts)} samples:")
    print(f"{'Sample':<15} {'Original':<12} {'Enhanced':<12} {'Difference':<12}")
    print("-" * 55)
    
    original_correct = 0
    enhanced_correct = 0
    
    for i, (text, label) in enumerate(zip(test_texts, test_labels)):
        
        # Get predictions
        try:
            orig_result = original_system.analyze_text(text)
            enh_result = enhanced_system.analyze_text(text)
            
            orig_pred = orig_result['is_steganographic']
            enh_pred = enh_result['is_steganographic']
            
            # Determine if predictions are correct
            is_stego = label != "clean"
            orig_correct_pred = orig_pred == is_stego
            enh_correct_pred = enh_pred == is_stego
            
            if orig_correct_pred:
                original_correct += 1
            if enh_correct_pred:
                enhanced_correct += 1
            
            # Format output
            orig_status = "✅ Correct" if orig_correct_pred else "❌ Wrong"
            enh_status = "✅ Correct" if enh_correct_pred else "❌ Wrong"
            
            difference = "Enhanced better" if enh_correct_pred and not orig_correct_pred else \
                        "Original better" if orig_correct_pred and not enh_correct_pred else \
                        "Same result"
            
            print(f"{label:<15} {orig_status:<12} {enh_status:<12} {difference:<12}")
            
        except Exception as e:
            print(f"{label:<15} Error: {str(e)[:30]}")
    
    # Summary
    print("\n" + "=" * 55)
    print(f"📊 Summary:")
    print(f"   Original Accuracy: {original_correct}/{len(test_texts)} ({original_correct/len(test_texts)*100:.1f}%)")
    print(f"   Enhanced Accuracy: {enhanced_correct}/{len(test_texts)} ({enhanced_correct/len(test_texts)*100:.1f}%)")
    
    if enhanced_correct > original_correct:
        print(f"   🎯 Enhanced model performs better!")
    elif original_correct > enhanced_correct:
        print(f"   ⚠️ Original model performs better")
    else:
        print(f"   🤝 Both models perform equally")
    
    # Load training summaries if available
    print(f"\n📈 Training Summary:")
    
    if os.path.exists('enhanced_training_summary.json'):
        with open('enhanced_training_summary.json', 'r') as f:
            summary = json.load(f)
            print(f"   Enhanced Training:")
            print(f"   - Final Accuracy: {summary.get('final_metrics', {}).get('accuracy', 0)*100:.1f}%")
            print(f"   - Training Time: {summary.get('training_time', 'Unknown')}")
            print(f"   - Epochs: {summary.get('epochs', 'Unknown')}")
            print(f"   - Samples: {summary.get('samples', 'Unknown')}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if enhanced_correct < original_correct:
        print(f"   - Enhanced training may need more epochs")
        print(f"   - Consider adjusting learning rate")
        print(f"   - Add more diverse training data")
    elif enhanced_correct == original_correct:
        print(f"   - Both models show similar performance")
        print(f"   - Consider ensemble approach")
        print(f"   - Test on larger dataset for better comparison")
    else:
        print(f"   - Enhanced training successful!")
        print(f"   - Deploy enhanced model for production")
        print(f"   - Continue with web interface testing")

if __name__ == "__main__":
    quick_test()
    print(f"\n🎉 Quick comparison complete!")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
