import argparse
import cv2
import numpy as np
import torch
from model import LeNet5

def preprocess_digit(roi: np.ndarray, size: int = 32) -> torch.Tensor:
    """
    Prend un ROI binaire (digits blancs sur fond noir),
    centre dans un carré noir et redimensionne à 32×32,
    puis normalise comme pendant l'entraînement MNIST.
    """
    h, w = roi.shape
    S = max(h, w)
    # fond noir
    square = np.zeros((S, S), dtype=np.uint8)
    # place le ROI blanc centré
    y_off, x_off = (S - h) // 2, (S - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = roi

    # redimensionne à 32×32
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    # de [0,255] → [0,1]
    arr = resized.astype(np.float32) / 255.0
    # normalisation (x−0.5)/0.5 → [-1,1]
    arr = (arr - 0.5) / 0.5
    # tenseur (1 x 1 x 32 x 32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def segment_and_predict(img_path: str, model: LeNet5, device: torch.device) -> str:
    # 1) Lire et convertir en niveaux de gris
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Binarisation INV : digits → blanc (255), fond → noir (0)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # 3) Détection des contours sur fond noir
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digits = []
    # trier de gauche à droite
    for c in sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0]):
        x, y, w, h = cv2.boundingRect(c)
        # filtrer le bruit
        if w * h < 100: 
            continue
        roi = thresh[y:y+h, x:x+w]           # ROI blanc sur fond noir
        tensor = preprocess_digit(roi).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred = out.argmax(dim=1).item()
        digits.append(str(pred))

    return ''.join(digits)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help="Chemin vers le fichier .pth entraîné")
    p.add_argument('--image', required=True,
                   help="Chemin vers la photo du chèque")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    result = segment_and_predict(args.image, model, device)
    print("Montant reconnu :", result)

if __name__ == '__main__':
    main()
