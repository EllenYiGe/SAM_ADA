import torch
from torch.utils.data import DataLoader
from datasets.office31 import OfficeDataset
from datasets.transforms_config import get_test_transform
from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier
from models.domain_discriminator import DomainDiscriminator
from models.interbn import InterBN

def evaluate_model(checkpoint_path, 
                  target_root, target_list_txt,
                  num_classes=31, device='cuda'):
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Create networks
    netF = FeatureExtractor(backbone='resnet50').to(device)
    netC = Classifier(in_dim=2048, num_classes=num_classes).to(device)
    netD = DomainDiscriminator(in_dim=2048, hidden_dim=1024).to(device)
    inter_bn = InterBN(num_features=3).to(device)

    netF.load_state_dict(ckpt['netF'])
    netC.load_state_dict(ckpt['netC'])
    netD.load_state_dict(ckpt['netD'])
    inter_bn.load_state_dict(ckpt['inter_bn'])

    netF.eval()
    netC.eval()
    inter_bn.eval()

    test_transform = get_test_transform()
    ds_test = OfficeDataset(root=target_root, file_list=target_list_txt, 
                           domain='target', transform=test_transform)
    loader_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader_test:
            x, y = x.to(device), y.to(device)
            # For target domain, use x,x as input (simplified)
            out_bn, _ = inter_bn(x, x)
            feats = netF(out_bn)
            logits = netC(feats)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct/total
    print(f"Loaded checkpoint from epoch={ckpt['epoch']}, EMA_Acc={ckpt['acc_ema']*100:.2f}%")
    print(f"Evaluation on target set => Acc={acc*100:.2f}%")

if __name__ == '__main__':
    checkpoint = "best_model_ema.pth"
    # Modify paths according to actual setup:
    target_root = "data/office31/webcam/images"
    target_list_txt = "data/office31/webcam_test_list.txt"
    
    evaluate_model(checkpoint, target_root, target_list_txt, 
                  num_classes=31, device='cuda')
