import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial training
    Input: (batch_size, in_dim)
    Output: (batch_size, 1) -> source domain probability (before sigmoid)
    """
    def __init__(self, in_dim=2048, hidden_dim=1024):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layer(x)
