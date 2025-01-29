import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

# Import RandAugment if available
try:
    from torchvision.transforms import RandAugment
    HAS_RANDAUG = True
except ImportError:
    HAS_RANDAUG = False

from datasets.office31 import OfficeDataset
from datasets.transforms_config import get_train_transform, get_test_transform
from models.feature_extractor import FeatureExtractor
from models.classifier import Classifier
from models.domain_discriminator import DomainDiscriminator
from models.interbn import InterBN

from utils.sam_optimizer import SAMOptimizer
from utils.losses import domain_adv_loss, sparse_reg_loss
from utils.ema import ModelEMA
from utils.metric_logger import MetricLogger

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    netF, netC, netD, inter_bn,
    source_loader, target_loader,
    sam_optimizer,
    device='cuda',
    scaler=None,
    lambda_sparse=1e-2,
    alpha_adv=1.0,
    use_ema=False,
    emaF=None, emaC=None, emaD=None, emaBN=None,
    logger=None
):
    netF.train()
    netC.train()
    netD.train()
    inter_bn.train()

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    num_iter = min(len(source_loader), len(target_loader))
    total_loss = 0.0

    for _ in range(num_iter):
        try:
            xs, ys = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            xs, ys = next(source_iter)

        try:
            xt, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            xt, _ = next(target_iter)

        xs, ys = xs.to(device), ys.to(device)
        xt = xt.to(device)

        # ====== forward #1 ======
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # InterBN
            out_s_bn, out_t_bn = inter_bn(xs, xt)
            fs = netF(out_s_bn)
            ft = netF(out_t_bn)

            # Classification & Domain discrimination
            logits_s = netC(fs)
            ds_s = netD(fs)
            ds_t = netD(ft)

            loss_cls = F.cross_entropy(logits_s, ys)
            loss_adv = alpha_adv * domain_adv_loss(ds_s, ds_t)
            loss_sp = (sparse_reg_loss(netF, lambda_sparse) +
                      sparse_reg_loss(netC, lambda_sparse) +
                      sparse_reg_loss(netD, lambda_sparse) +
                      sparse_reg_loss(inter_bn, lambda_sparse))
            loss = loss_cls + loss_adv + loss_sp

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # SAM first step
        sam_optimizer.first_step(zero_grad=True)

        # ===== forward #2 (with perturbed weights) =====
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out_s_bn2, out_t_bn2 = inter_bn(xs, xt)
            fs2 = netF(out_s_bn2)
            ft2 = netF(out_t_bn2)

            logits_s2 = netC(fs2)
            ds_s2 = netD(fs2)
            ds_t2 = netD(ft2)

            loss_cls2 = F.cross_entropy(logits_s2, ys)
            loss_adv2 = alpha_adv * domain_adv_loss(ds_s2, ds_t2)
            loss_sp2 = (sparse_reg_loss(netF, lambda_sparse) +
                       sparse_reg_loss(netC, lambda_sparse) +
                       sparse_reg_loss(netD, lambda_sparse) +
                       sparse_reg_loss(inter_bn, lambda_sparse))
            loss2 = loss_cls2 + loss_adv2 + loss_sp2

        if scaler:
            scaler.scale(loss2).backward()
            scaler.step(sam_optimizer.base_optimizer)
            scaler.update()
        else:
            loss2.backward()
            sam_optimizer.second_step(zero_grad=True)

        # When using scaler, since step() already called base_optimizer.step(),
        # we still need to call second_step() to remove perturbation:
        if scaler:
            sam_optimizer.second_step(zero_grad=True)

        total_loss += loss.item()

        # Update EMA
        if use_ema:
            if emaF:  emaF.update(netF)
            if emaC:  emaC.update(netC)
            if emaD:  emaD.update(netD)
            if emaBN: emaBN.update(inter_bn)

        # Update metrics
        logger.update("train_loss", loss.item())
        logger.update("cls_loss", loss_cls.item())
        logger.update("adv_loss", loss_adv.item())
        logger.update("sparse_loss", loss_sp.item())

    return total_loss / num_iter

def evaluate(netF, netC, inter_bn, loader, device='cuda',
            use_ema=False, emaF=None, emaC=None, emaBN=None):
    """
    Evaluation function, optionally using EMA weights
    """
    backupF = None
    backupC = None
    backupBN = None

    if use_ema and (emaF is not None) and (emaC is not None) and (emaBN is not None):
        # Backup original weights
        backupF = netF.state_dict()
        backupC = netC.state_dict()
        backupBN = inter_bn.state_dict()

        # Apply EMA weights
        emaF.apply_shadow(netF)
        emaC.apply_shadow(netC)
        emaBN.apply_shadow(inter_bn)

    netF.eval()
    netC.eval()
    inter_bn.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # For target evaluation, use x,x as input
            out_bn, _ = inter_bn(x, x)
            feats = netF(out_bn)
            logits = netC(feats)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total

    # Restore weights
    if use_ema and backupF is not None:
        netF.load_state_dict(backupF)
        netC.load_state_dict(backupC)
        inter_bn.load_state_dict(backupBN)

    return acc

def main():
    set_random_seed(2023)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: Office31 A->W task
    source_domain_root = "data/office31/amazon/images"
    target_domain_root = "data/office31/webcam/images"
    source_list_txt = "data/office31/amazon_train_list.txt"
    target_list_txt = "data/office31/webcam_train_list.txt"
    target_test_list = "data/office31/webcam_test_list.txt"

    max_epoch = 20
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-3
    backbone = 'resnet50'
    num_classes = 31
    threshold_gamma = 0.5
    alpha_adv = 1.0
    lambda_sparse = 1e-2
    rho_sam = 0.05

    # ---------- Data -----------
    transform_train = get_train_transform()
    transform_test = get_test_transform()

    # Optional RandAugment
    if HAS_RANDAUG:
        transform_train.transforms.insert(0, RandAugment(num_ops=2, magnitude=9))

    ds_source = OfficeDataset(source_domain_root, source_list_txt, domain='source', transform=transform_train)
    ds_target = OfficeDataset(target_domain_root, target_list_txt, domain='target', transform=transform_train)
    ds_test = OfficeDataset(target_domain_root, target_test_list, domain='target', transform=transform_test)

    source_loader = DataLoader(ds_source, batch_size=batch_size, shuffle=True, num_workers=4)
    target_loader = DataLoader(ds_target, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------- Models -----------
    netF = FeatureExtractor(backbone=backbone).to(device)
    netC = Classifier(in_dim=2048, num_classes=num_classes).to(device)
    netD = DomainDiscriminator(in_dim=2048, hidden_dim=1024).to(device)
    inter_bn = InterBN(num_features=3, threshold=threshold_gamma).to(device)  # 3 channels for RGB images

    # ---------- Optimizer & SAM -----------
    base_optimizer = optim.AdamW(
        params = list(netF.parameters())
               + list(netC.parameters())
               + list(netD.parameters())
               + list(inter_bn.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    sam_optimizer = SAMOptimizer(base_optimizer, rho=rho_sam)

    # Optional scheduler (e.g., cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_0=10, T_mult=1)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- EMA (optional) ----------
    use_ema = True
    emaF = ModelEMA(netF, decay=0.9998) if use_ema else None
    emaC = ModelEMA(netC, decay=0.9998) if use_ema else None
    emaD = ModelEMA(netD, decay=0.9998) if use_ema else None
    emaBN = ModelEMA(inter_bn, decay=0.9998) if use_ema else None

    # Initialize metric logger
    logger = MetricLogger(
        log_dir="logs",
        experiment_name=f"office31_amazon_to_webcam",
        use_tensorboard=True,
        log_interval=10
    )

    best_acc_ema = 0.0
    for epoch in range(1, max_epoch+1):
        # Train one epoch
        train_loss = train_one_epoch(
            netF, netC, netD, inter_bn,
            source_loader, target_loader,
            sam_optimizer,
            device=device,
            scaler=scaler,
            lambda_sparse=lambda_sparse,
            alpha_adv=alpha_adv,
            use_ema=use_ema,
            emaF=emaF, emaC=emaC, emaD=emaD, emaBN=emaBN,
            logger=logger
        )

        # Optional: LR scheduling
        scheduler.step()

        # Evaluate (normal weights)
        acc_normal = evaluate(netF, netC, inter_bn, test_loader, device=device, use_ema=False)

        # Evaluate (EMA weights)
        acc_ema = 0.0
        if use_ema and emaF and emaC and emaBN:
            acc_ema = evaluate(
                netF, netC, inter_bn, test_loader, device=device,
                use_ema=True, emaF=emaF, emaC=emaC, emaBN=emaBN
            )
            if acc_ema > best_acc_ema:
                best_acc_ema = acc_ema
                # Save best model
                torch.save({
                    'netF': netF.state_dict(),
                    'netC': netC.state_dict(),
                    'netD': netD.state_dict(),
                    'inter_bn': inter_bn.state_dict(),
                    'acc_ema': acc_ema,
                    'epoch': epoch,
                }, "best_model_ema.pth")

        logger.update("target_acc", acc_ema)
        logger.log_epoch(
            epoch=epoch,
            total_epochs=max_epoch,
            learning_rate=base_optimizer.param_groups[0]["lr"]
        )

    logger.close()
    print("Training finished. Best EMA Acc= {:.2f}%".format(best_acc_ema*100))

if __name__ == '__main__':
    main()
