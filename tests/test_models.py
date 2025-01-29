import torch
import pytest
from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier
from models.domain_discriminator import DomainDiscriminator
from models.interbn import InterBN

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def input_shape():
    return (3, 224, 224)

def test_feature_extractor(device, batch_size, input_shape):
    model = FeatureExtractor(backbone='resnet50').to(device)
    x = torch.randn(batch_size, *input_shape).to(device)
    out = model(x)
    assert out.shape == (batch_size, 2048)

def test_classifier(device, batch_size):
    model = Classifier(in_dim=2048, num_classes=31).to(device)
    x = torch.randn(batch_size, 2048).to(device)
    out = model(x)
    assert out.shape == (batch_size, 31)

def test_domain_discriminator(device, batch_size):
    model = DomainDiscriminator(in_dim=2048, hidden_dim=1024).to(device)
    x = torch.randn(batch_size, 2048).to(device)
    out = model(x)
    assert out.shape == (batch_size, 1)

def test_interbn(device, batch_size, input_shape):
    model = InterBN(num_features=input_shape[0]).to(device)
    x_s = torch.randn(batch_size, *input_shape).to(device)
    x_t = torch.randn(batch_size, *input_shape).to(device)
    out_s, out_t = model(x_s, x_t)
    assert out_s.shape == (batch_size, *input_shape)
    assert out_t.shape == (batch_size, *input_shape)

def test_interbn_channel_exchange(device):
    batch_size = 2
    channels = 3
    spatial_size = 4
    model = InterBN(num_features=channels, threshold=0.5).to(device)
    
    # Set some gammas below threshold
    with torch.no_grad():
        model.gamma_s[0] = 0.1  # Below threshold
        model.gamma_t[1] = 0.1  # Below threshold
    
    x_s = torch.randn(batch_size, channels, spatial_size, spatial_size).to(device)
    x_t = torch.randn(batch_size, channels, spatial_size, spatial_size).to(device)
    
    out_s, out_t = model(x_s, x_t)
    
    # Check shapes
    assert out_s.shape == x_s.shape
    assert out_t.shape == x_t.shape
