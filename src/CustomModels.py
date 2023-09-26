import torch

class DinoVisionClassifier(torch.nn.Module):
    def __init__(self, dinov2, num_classes=5):
        super(DinoVisionClassifier, self).__init__()
        self.transformer = dinov2
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(384, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x