"""
Quick Test for Steganalysis System

This script runs a quick test to verify that the steganalysis components
are working correctly before running the full training or demo.
"""

import torch
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from steganalysis_system import (
        SteganalysisSystem, 
        BinaryTextDetector, 
        CapacityEstimator, 
        TextTypeClassifier,
        ImageFeatureExtractor,
        TextPatternAnalyzer
    )
    print("✅ Successfully imported steganalysis components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_image_feature_extractor():
    """Test the image feature extractor."""
    print("\n🧪 Testing Image Feature Extractor...")
    
    try:
        extractor = ImageFeatureExtractor()
        
        # Test with random image batch
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 32, 32)
        
        features = extractor(test_images)
        
        print(f"   Input shape: {test_images.shape}")
        print(f"   Output shape: {features.shape}")
        print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
        
        assert features.shape[0] == batch_size, "Batch size mismatch"
        assert features.shape[1] == 128, "Feature dimension mismatch"
        
        print("   ✅ Image Feature Extractor working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Image Feature Extractor: {e}")
        return False


def test_binary_detector():
    """Test the binary text detector."""
    print("\n🧪 Testing Binary Text Detector...")
    
    try:
        detector = BinaryTextDetector()
        
        # Test with random image batch
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 32, 32)
        
        predictions = detector(test_images)
        
        print(f"   Input shape: {test_images.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Predictions: {predictions.squeeze().tolist()}")
        
        assert predictions.shape[0] == batch_size, "Batch size mismatch"
        assert torch.all((predictions >= 0) & (predictions <= 1)), "Predictions not in [0,1] range"
        
        print("   ✅ Binary Text Detector working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Binary Text Detector: {e}")
        return False


def test_capacity_estimator():
    """Test the capacity estimator."""
    print("\n🧪 Testing Capacity Estimator...")
    
    try:
        estimator = CapacityEstimator(max_capacity=100)
        
        # Test with random image batch
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 32, 32)
        
        capacities = estimator(test_images)
        
        print(f"   Input shape: {test_images.shape}")
        print(f"   Output shape: {capacities.shape}")
        print(f"   Estimated capacities: {capacities.squeeze().tolist()}")
        
        assert capacities.shape[0] == batch_size, "Batch size mismatch"
        assert torch.all((capacities >= 0) & (capacities <= 100)), "Capacities not in valid range"
        
        print("   ✅ Capacity Estimator working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Capacity Estimator: {e}")
        return False


def test_text_type_classifier():
    """Test the text type classifier."""
    print("\n🧪 Testing Text Type Classifier...")
    
    try:
        classifier = TextTypeClassifier()
        
        # Test with random image batch
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 32, 32)
        
        type_probs = classifier(test_images)
        
        print(f"   Input shape: {test_images.shape}")
        print(f"   Output shape: {type_probs.shape}")
        print(f"   Type probabilities (first sample): {type_probs[0].tolist()}")
        
        assert type_probs.shape[0] == batch_size, "Batch size mismatch"
        assert type_probs.shape[1] == 3, "Should have 3 text types"
        assert torch.allclose(type_probs.sum(dim=1), torch.ones(batch_size)), "Probabilities don't sum to 1"
        
        print("   ✅ Text Type Classifier working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Text Type Classifier: {e}")
        return False


def test_text_pattern_analyzer():
    """Test the text pattern analyzer."""
    print("\n🧪 Testing Text Pattern Analyzer...")
    
    try:
        analyzer = TextPatternAnalyzer()
        
        # Test with different text types
        test_texts = [
            "Hello world this is a normal English text",
            "aB3xY9zK2mN8qP7dE5fG1hJ4lM",  # Encrypted-like
            "",  # Empty
            "The quick brown fox jumps over the lazy dog"
        ]
        
        for i, text in enumerate(test_texts):
            features = analyzer.analyze_text(text)
            print(f"   Text {i+1}: '{text[:20]}{'...' if len(text) > 20 else ''}'")
            
            if features:
                print(f"      Entropy: {features.get('entropy', 0):.2f}")
                print(f"      Is Encrypted: {features.get('is_encrypted', 0):.2f}")
                print(f"      Chi-squared: {features.get('chi_squared', 0):.2f}")
            else:
                print(f"      No features (empty text)")
        
        print("   ✅ Text Pattern Analyzer working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Text Pattern Analyzer: {e}")
        return False


def test_complete_steganalysis_system():
    """Test the complete steganalysis system."""
    print("\n🧪 Testing Complete Steganalysis System...")
    
    try:
        steganalysis = SteganalysisSystem(device='cpu')
        
        # Test with a single image
        test_image = torch.randn(1, 3, 32, 32)
        test_text = "This is a test message"
        
        result = steganalysis.analyze_image(test_image, test_text)
        
        print(f"   Detection: {result.has_hidden_text}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Estimated Capacity: {result.estimated_capacity}")
        print(f"   Text Type: {result.text_type}")
        print(f"   Features count: {len(result.features)}")
        
        # Test batch analysis
        test_images = torch.randn(3, 3, 32, 32)
        test_texts = ["Message 1", "Message 2", "Message 3"]
        
        batch_results = steganalysis.batch_analysis(test_images, test_texts)
        print(f"   Batch analysis results: {len(batch_results)} results")
        
        print("   ✅ Complete Steganalysis System working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in Complete Steganalysis System: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Steganalysis System Quick Test")
    print("================================")
    
    tests = [
        test_image_feature_extractor,
        test_binary_detector,
        test_capacity_estimator,
        test_text_type_classifier,
        test_text_pattern_analyzer,
        test_complete_steganalysis_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Steganalysis system is ready.")
        print("\nNext steps:")
        print("1. Run 'python train_steganalysis.py' to train the models")
        print("2. Run 'python steganalysis_demo.py' to see the demo")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
