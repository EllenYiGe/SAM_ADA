import torch
from datasets.office31 import OfficeDataset
from datasets.transforms_config import get_train_transform
from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier
from models.domain_discriminator import DomainDiscriminator
from models.interbn import InterBN
from utils.losses import domain_adv_loss, sparse_reg_loss
from utils.sam_optimizer import SAMOptimizer

def test_dataset():
    transform = get_train_transform()
    ds = OfficeDataset(
        root="data/office31/amazon",
        file_list="data/office31/amazon_train_list.txt",
        domain='source',
        transform=transform
    )
    print("Dataset length:", len(ds))
    img, label = ds[0]
    print("Image shape:", img.shape)
    print("Label:", label)

def test_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test models with dummy input
    netF = FeatureExtractor(backbone='resnet50').to(device)
    netC = Classifier(in_dim=2048, num_classes=31).to(device)
    netD = DomainDiscriminator(in_dim=2048, hidden_dim=1024).to(device)
    inter_bn = InterBN(num_features=3).to(device)

    # Create dummy inputs
    x_s = torch.randn(4, 3, 224, 224).to(device)
    x_t = torch.randn(4, 3, 224, 224).to(device)

    # Test forward passes
    out_s_bn, out_t_bn = inter_bn(x_s, x_t)
    print("InterBN output shapes:", out_s_bn.shape, out_t_bn.shape)

    fs = netF(out_s_bn)
    ft = netF(out_t_bn)
    print("Feature extractor output shapes:", fs.shape, ft.shape)

    logits = netC(fs)
    print("Classifier output shape:", logits.shape)

    d_out = netD(fs)
    print("Discriminator output shape:", d_out.shape)

def test_losses():
    # Test losses with dummy inputs
    d_out_s = torch.randn(4, 1)
    d_out_t = torch.randn(4, 1)
    loss = domain_adv_loss(d_out_s, d_out_t)
    print("Domain adversarial loss:", loss.item())

def test_sam_optimizer():
    model = torch.nn.Linear(10, 1)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sam_optimizer = SAMOptimizer(base_optimizer, rho=0.05)
    
    # Test optimization step
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # First forward-backward pass
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    sam_optimizer.first_step(zero_grad=True)
    
    # Second forward-backward pass
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    sam_optimizer.second_step(zero_grad=True)
    
    print("SAM optimization test passed")

if __name__ == "__main__":
    print("Testing dataset...")
    test_dataset()
    print("\nTesting models...")
    test_models()
    print("\nTesting losses...")
    test_losses()
    print("\nTesting SAM optimizer...")
    test_sam_optimizer()
    print("\nAll tests completed!")
