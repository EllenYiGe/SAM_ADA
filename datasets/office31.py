import os
from torch.utils.data import Dataset
from PIL import Image

class OfficeDataset(Dataset):
    """
    Office-31 dataset implementation.
    file_list format for source domain: "relative_path label"
                     for target domain: "relative_path [label or -1]"
    """
    def __init__(self, root, file_list, domain='source', transform=None):
        """
        Args:
            root: Image root directory
            file_list: Text file containing image paths and labels
            domain: 'source' or 'target'
            transform: Image transformations
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.domain = domain

        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if domain == 'source':
                    # e.g. "images/back_pack/image_0001.jpg 0"
                    path, label_str = parts[0], parts[1]
                    label = int(label_str)
                else:
                    # target: may have no label or -1
                    path = parts[0]
                    label = -1 if len(parts) < 2 else int(parts[1])
                # Remove 'images/' prefix if present in the path
                if path.startswith('images/'):
                    path = path[7:]  # Remove 'images/' prefix
                self.samples.append((path, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        fullpath = os.path.join(self.root, path)
        img = Image.open(fullpath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)
