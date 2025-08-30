"""Minimal test for steganalysis components"""
import torch
import torch.nn as nn

# Test dataclass import
try:
    from dataclasses import dataclass
    print("✅ dataclasses imported successfully")
except ImportError:
    print("❌ dataclasses not available, creating manual class")

# Define a simple version without dataclass for testing
class SteganalysisResult:
    def __init__(self, has_hidden_text, confidence_score, estimated_capacity, text_type, features):
        self.has_hidden_text = has_hidden_text
        self.confidence_score = confidence_score
        self.estimated_capacity = estimated_capacity
        self.text_type = text_type
        self.features = features

# Test basic torch functionality
print(f"PyTorch version: {torch.__version__}")

# Test a simple neural network
class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        x = torch.sigmoid(self.fc(x))
        return x

# Test the detector
detector = SimpleDetector()
test_image = torch.randn(1, 3, 32, 32)
output = detector(test_image)
print(f"✅ Simple detector test successful: {output.item():.3f}")

# Test result creation
result = SteganalysisResult(
    has_hidden_text=True,
    confidence_score=0.8,
    estimated_capacity=25,
    text_type="plain",
    features={"test": 1.0}
)
print(f"✅ SteganalysisResult created: {result.confidence_score}")

print("🎉 Minimal test completed successfully!")
