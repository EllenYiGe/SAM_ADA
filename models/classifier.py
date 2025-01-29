import torch.nn as nn

class Classifier(nn.Module):
    """
    Simple FC classifier
    Input: (batch_size, feat_dim)
    Output: (batch_size, num_classes)
    """
    def __init__(self, in_dim=2048, num_classes=31):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
