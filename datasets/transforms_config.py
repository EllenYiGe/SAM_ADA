import torchvision.transforms as T

def get_train_transform():
    """
    Returns training stage image transformations.
    Can add RandAugment / ColorJitter etc. as needed.
    """
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])
    return transform

def get_test_transform():
    """
    Returns test stage image transformations (typically simpler than training)
    """
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])
    return transform
