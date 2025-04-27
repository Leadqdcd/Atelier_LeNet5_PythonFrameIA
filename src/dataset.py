import os
import zipfile
from io import BytesIO

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class NISTZipDataset(Dataset):
    """
    Lit les images 0–9 directement depuis data/by_class.zip sans extraire.
    """
    def __init__(self, zip_path, classes=None, transform=None):
        assert os.path.isfile(zip_path), f"ZIP introuvable : {zip_path}"
        self.zip = zipfile.ZipFile(zip_path, 'r')
        self.transform = transform
        classes = classes or [str(i) for i in range(10)]

        names = self.zip.namelist()
        # Détecte automatiquement si les images sont sous "by_class/" ou à la racine
        prefix = "by_class/" if any(n.startswith("by_class/") for n in names) else ""
        self.samples = []
        for idx, cls in enumerate(classes):
            p = f"{prefix}{cls}/"
            for fn in names:
                if fn.startswith(p) and fn.lower().endswith('.png'):
                    self.samples.append((fn, idx))
        assert self.samples, "Aucun fichier trouvé dans le ZIP : vérifiez son contenu."

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
