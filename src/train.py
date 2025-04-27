import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
from dataset import NISTSD19Dataset
from model import LeNet5

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = NISTSD19Dataset(args.data_dir, classes=[str(i) for i in range(10)])
    n_val = int(len(ds) * 0.2)
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs} — train loss: {train_loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print("Modèle sauvegardé dans", args.output)

if __name__ == '__main__':
    main()
