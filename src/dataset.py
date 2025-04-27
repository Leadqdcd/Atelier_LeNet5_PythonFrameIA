import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class NISTZipDataset(Dataset):
    """
    Lit les images 0–9 directement depuis by_class.zip sans extraire.
    """
    def __init__(self, zip_path, classes=None, transform=None):
        self.zip = zipfile.ZipFile(zip_path, 'r')
        self.transform = transform
        classes = classes or [str(i) for i in range(10)]
        self.samples = []
        for idx, cls in enumerate(classes):
            prefix = f"by_class/{cls}/"
            for fn in self.zip.namelist():
                if fn.startswith(prefix) and fn.lower().endswith('.png'):
                    self.samples.append((fn, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fn, label = self.samples[i]
        data = self.zip.read(fn)
        img = Image.open(BytesIO(data)).convert('L')
        img = img.resize((32, 32), Image.ANTIALIAS)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,32,32)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label
