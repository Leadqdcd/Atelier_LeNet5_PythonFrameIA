import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from model import LeNet5

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / (len(loader.dataset))
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--output',     required=True,
                        help="Chemin du .pth de sortie")
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Transforms : resize 32×32 + ToTensor + normalize
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2) Chargement MNIST
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    full_train = MNIST(root=data_root, train=True,  download=True, transform=transform)
    test_ds    = MNIST(root=data_root, train=False, download=True, transform=transform)

    # 3) Split train/val 80/20
    n_val = int(len(full_train) * 0.2)
    train_ds, val_ds = random_split(full_train, [len(full_train) - n_val, n_val])

    # 4) DataLoaders (CPU-friendly)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    print(f"✅ Train : {len(train_ds)}  Val : {len(val_ds)}  Test : {len(test_ds)} images")

    # 5) Modèle, critère, optimiseur
    model     = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6) Boucle d'entraînement + validation
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} — train loss: {tr_loss:.4f}  "
              f"val loss: {val_loss:.4f}  val acc: {val_acc*100:.2f}%")

    # 7) Évaluation finale sur test
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"▶ Test   — loss: {test_loss:.4f}  acc: {test_acc*100:.2f}%")

    # 8) Sauvegarde du modèle
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print("💾 Modèle sauvegardé dans", args.output)

if __name__ == '__main__':
    main()
