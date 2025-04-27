from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# 1) transforms : resize → 32×32 + ToTensor + normalize
tf = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2) dataset MNIST (ou EMNIST(split="digits") si vous préférez)
full_ds = MNIST(root="data", train=True,  download=True, transform=tf)
test_ds = MNIST(root="data", train=False, download=True, transform=tf)

# 3) split train / val 80/20
n_val = int(len(full_ds) * 0.2)
train_ds, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val])

# 4) DataLoaders
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
