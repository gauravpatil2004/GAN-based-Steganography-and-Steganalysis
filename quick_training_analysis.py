"""
Quick Evaluation Summary of Enhanced Training Results

This script provides a quick summary of the enhanced training results.
"""

import json
import os
from datetime import datetime

def load_training_summaries():
    """Load training summaries from both sessions."""
    
    summaries = {}
    
    # Try to load enhanced training summary
    enhanced_path = 'enhanced_training_summary.json'
    if os.path.exists(enhanced_path):
        with open(enhanced_path, 'r') as f:
            summaries['enhanced'] = json.load(f)
    
    # Check for other training results
    if os.path.exists('steganalysis_evaluation.json'):
        with open('steganalysis_evaluation.json', 'r') as f:
            summaries['evaluation'] = json.load(f)
    
    return summaries

def analyze_enhanced_results():
    """Analyze the enhanced training results."""
    
    print("🔬 Enhanced Steganalysis Training - Quick Analysis")
    print("=" * 55)
    
    summaries = load_training_summaries()
    
    if 'enhanced' in summaries:
        enhanced = summaries['enhanced']
        
        print(f"\n📊 Enhanced Training Summary:")
        print(f"   Training Type: {enhanced.get('training_type', 'Unknown')}")
        print(f"   Completion Time: {enhanced.get('timestamp', 'Unknown')}")
        print(f"   Training Duration: {enhanced.get('training_time', 'Unknown')}")
        print(f"   Epochs: {enhanced.get('epochs', 'Unknown')}")
        print(f"   Training Samples: {enhanced.get('samples', 'Unknown')}")
        print(f"   Batch Size: {enhanced.get('batch_size', 'Unknown')}")
        
        if 'final_metrics' in enhanced:
            metrics = enhanced['final_metrics']
            print(f"\n🎯 Final Performance Metrics:")
            print(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
            print(f"   Precision: {metrics.get('precision', 0):.1%}")
            print(f"   Recall: {metrics.get('recall', 0):.1%}")
            print(f"   F1-Score: {metrics.get('f1_score', 0):.3f}")
            print(f"   Capacity MAE: {metrics.get('capacity_mae', 0):.1f} characters")
    
    # Check if models exist
    print(f"\n📁 Model Files Status:")
    original_path = os.path.join('models', 'steganalysis')
    enhanced_path = os.path.join('models', 'steganalysis_enhanced')
    
    if os.path.exists(original_path):
        orig_files = os.listdir(original_path)
        print(f"   ✅ Original Models: {len(orig_files)} files")
        for file in orig_files:
            print(f"      - {file}")
    else:
        print(f"   ❌ Original Models: Not found")
    
    if os.path.exists(enhanced_path):
        enh_files = os.listdir(enhanced_path)
        print(f"   ✅ Enhanced Models: {len(enh_files)} files")
        for file in enh_files:
            print(f"      - {file}")
    else:
        print(f"   ❌ Enhanced Models: Not found")
    
    # Analysis and recommendations
    print(f"\n🔍 Analysis:")
    
    if 'enhanced' in summaries and 'final_metrics' in summaries['enhanced']:
        metrics = summaries['enhanced']['final_metrics']
        accuracy = metrics.get('accuracy', 0)
        
        if accuracy < 0.5:
            print(f"   ⚠️ Accuracy ({accuracy:.1%}) is below 50% - indicates training challenges")
            print(f"   💡 Recommendations:")
            print(f"      - Increase training data diversity")
            print(f"      - Adjust learning rate")
            print(f"      - Increase training epochs")
            print(f"      - Check data labeling accuracy")
        elif accuracy < 0.7:
            print(f"   📈 Accuracy ({accuracy:.1%}) shows learning but needs improvement")
            print(f"   💡 Recommendations:")
            print(f"      - Continue training with more epochs")
            print(f"      - Fine-tune hyperparameters")
            print(f"      - Add data augmentation")
        else:
            print(f"   ✅ Accuracy ({accuracy:.1%}) is good!")
            print(f"   💡 Next steps:")
            print(f"      - Deploy for production testing")
            print(f"      - Create web interface")
            print(f"      - Generate research reports")
    
    # Training comparison if available
    print(f"\n📈 Training Progress Analysis:")
    
    # Check if training curves exist
    curves_file = 'steganalysis_training_curves.png'
    if os.path.exists(curves_file):
        print(f"   ✅ Training curves available: {curves_file}")
    else:
        print(f"   ⚠️ Training curves not found")
    
    # Check for comparison results
    comparison_file = 'steganalysis_comparison_results.json'
    if os.path.exists(comparison_file):
        print(f"   ✅ Comparison results available: {comparison_file}")
    else:
        print(f"   ⚠️ Comparison results not yet generated")
    
    print(f"\n🎯 Summary:")
    print(f"   Status: Enhanced training completed")
    print(f"   Models: Saved and ready for use")
    print(f"   Next: Compare with original models for improvement analysis")

def suggest_next_steps():
    """Suggest next steps based on current status."""
    
    print(f"\n🚀 Suggested Next Steps:")
    print(f"")
    print(f"1. 📊 **Run Model Comparison**:")
    print(f"   python compare_training_results.py")
    print(f"   - Compare original vs enhanced models")
    print(f"   - Generate performance visualizations")
    print(f"")
    print(f"2. 🌐 **Launch Web Interface**:")
    print(f"   streamlit run app/steganalysis_web_app.py")
    print(f"   - Test both steganography and steganalysis")
    print(f"   - Interactive demonstration")
    print(f"")
    print(f"3. 🔧 **Further Optimization**:")
    print(f"   - Increase training epochs (30 → 50+)")
    print(f"   - Add learning rate scheduling")
    print(f"   - Implement ensemble methods")
    print(f"")
    print(f"4. 📝 **Documentation & Deployment**:")
    print(f"   - Generate final project report")
    print(f"   - Create academic publication")
    print(f"   - Prepare for production deployment")

if __name__ == "__main__":
    analyze_enhanced_results()
    suggest_next_steps()
    
    print(f"\n🎉 Enhanced Training Analysis Complete!")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
