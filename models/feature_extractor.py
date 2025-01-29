import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet(backbone='resnet50', pretrained=True):
    """
    Returns specified ResNet without the final fc layer
    """
    if backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    else:
        raise NotImplementedError("Only resnet50/resnet101 are supported.")
    
    modules = list(model.children())[:-1]  # Remove final FC
    feature_extractor = nn.Sequential(*modules)
    return feature_extractor

class FeatureExtractor(nn.Module):
    """
    ResNet as feature extractor
    Output: (batch_size, 2048)
    """
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.net = get_resnet(backbone=backbone, pretrained=True)

    def forward(self, x):
        x = self.net(x)            # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 2048]
        return x
