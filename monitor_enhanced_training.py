"""
Real-time Training Monitor

This script monitors the enhanced training progress and provides
live updates on performance improvements.
"""

import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def monitor_training_progress():
    """Monitor and display training progress in real-time."""
    
    print("📊 Real-time Training Monitor")
    print("============================")
    print("Monitoring enhanced steganalysis training...")
    print("Press Ctrl+C to stop monitoring\n")
    
    start_time = datetime.now()
    check_interval = 30  # Check every 30 seconds
    
    try:
        while True:
            # Check for training files
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            print(f"⏰ {current_time.strftime('%H:%M:%S')} - Elapsed: {elapsed}")
            
            # Check for model files
            steg_models = os.path.join('models', 'steganalysis')
            enhanced_models = os.path.join('models', 'steganalysis_enhanced')
            
            if os.path.exists(steg_models):
                print("  ✅ Base models available")
                
            if os.path.exists(enhanced_models):
                print("  🎯 Enhanced models being trained/saved")
                
            # Check for training output files
            if os.path.exists('enhanced_training_summary.json'):
                print("  📊 Training summary available")
                try:
                    with open('enhanced_training_summary.json', 'r') as f:
                        summary = json.load(f)
                    print(f"  📈 Training completed at: {summary.get('timestamp', 'Unknown')}")
                    if 'final_metrics' in summary:
                        metrics = summary['final_metrics']
                        print(f"  🎯 Final Accuracy: {metrics.get('accuracy', 0):.1%}")
                        print(f"  📏 Final MAE: {metrics.get('capacity_mae', 0):.2f}")
                    break
                except:
                    pass
            
            # Check for plots
            plots = ['enhanced_steganalysis_training.png', 'steganalysis_training_curves.png']
            for plot in plots:
                if os.path.exists(plot):
                    print(f"  📊 Plot available: {plot}")
            
            print("  ⏳ Training in progress...\n")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    
    print(f"\n📊 Monitoring completed at {datetime.now().strftime('%H:%M:%S')}")


def display_expected_improvements():
    """Display expected improvements from enhanced training."""
    
    print("\n🎯 Expected Improvements from Enhanced Training")
    print("=" * 50)
    
    improvements = {
        "Training Data": {
            "Before": "1,000 samples",
            "After": "3,000 samples",
            "Impact": "Better generalization"
        },
        "Detection Accuracy": {
            "Before": "~48% (baseline)",
            "After": "75-85% (target)",
            "Impact": "Production-ready performance"
        },
        "Capacity Estimation": {
            "Before": "±4.95 characters MAE",
            "After": "±2-3 characters MAE", 
            "Impact": "High precision estimates"
        },
        "Training Stability": {
            "Before": "Basic Adam optimizer",
            "After": "AdamW + LR scheduling",
            "Impact": "Better convergence"
        },
        "Text Diversity": {
            "Before": "Random text generation",
            "After": "English/encrypted/random mix",
            "Impact": "Robust text type classification"
        },
        "Validation": {
            "Before": "No validation split",
            "After": "80/20 train/validation",
            "Impact": "Proper performance evaluation"
        }
    }
    
    for category, details in improvements.items():
        print(f"\n📈 {category}:")
        print(f"   Before: {details['Before']}")
        print(f"   After:  {details['After']}")
        print(f"   Impact: {details['Impact']}")
    
    print(f"\n⏱️ Expected Training Time: 15-25 minutes")
    print(f"🎯 Target Performance: Research-grade steganalysis system")


def check_training_status():
    """Check current training status."""
    
    print("\n🔍 Current Training Status Check")
    print("=" * 35)
    
    # Check terminal processes (simplified)
    print("📊 Training Status:")
    
    if os.path.exists('enhanced_training_summary.json'):
        print("  ✅ Enhanced training completed!")
        try:
            with open('enhanced_training_summary.json', 'r') as f:
                summary = json.load(f)
            print(f"  📊 Completion time: {summary.get('timestamp', 'Unknown')}")
            print(f"  📈 Training samples: {summary.get('samples', 'Unknown')}")
            print(f"  ⏱️ Duration: {summary.get('training_time', 'Unknown')}")
        except:
            print("  ⚠️ Could not read training summary")
    else:
        print("  ⏳ Enhanced training in progress...")
        
    # Check model directories
    base_models = os.path.join('models', 'steganalysis')
    enhanced_models = os.path.join('models', 'steganalysis_enhanced')
    
    if os.path.exists(base_models):
        print(f"  ✅ Base models: {len(os.listdir(base_models))} files")
    
    if os.path.exists(enhanced_models):
        print(f"  🎯 Enhanced models: {len(os.listdir(enhanced_models))} files")
    else:
        print("  ⏳ Enhanced models: Training in progress")


if __name__ == "__main__":
    print("🔬 Enhanced Steganalysis Training - Option C")
    print("==========================================")
    
    display_expected_improvements()
    check_training_status()
    
    print("\n🎛️ Monitoring Options:")
    print("1. Real-time monitoring (live updates)")
    print("2. Status check only")
    print("3. Wait for completion")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        monitor_training_progress()
    elif choice == "2":
        check_training_status()
    else:
        print("⏳ Enhanced training running in background...")
        print("   Check back in 15-25 minutes for completion")
        print("   Or run this script again for status updates")
    
    print("\n🎉 Enhanced training will significantly improve performance!")
    print("📈 Expected: 48% → 75-85% detection accuracy")
    print("📏 Expected: ±5 → ±2-3 character capacity estimation")
