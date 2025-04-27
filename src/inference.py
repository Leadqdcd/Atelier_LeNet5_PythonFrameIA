import argparse
import cv2
import numpy as np
import torch
from model import LeNet5

def preprocess_digit(roi: np.ndarray, size: int = 32) -> torch.Tensor:
    """
    A partir d'un ROI binaire (255=chiffre, 0=fond),
    on centre le chiffre dans un carré noir, on resize à 32×32,
    puis on normalise [(x-0.5)/0.5] comme pendant l'entraînement.
    """
    h, w = roi.shape
    S = max(h, w)
    # fond noir
    square = np.zeros((S, S), dtype=np.uint8)
    # centrer le ROI blanc
    y_off = (S - h) // 2
    x_off = (S - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = roi

    # resize
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def segment_and_predict(img_path: str, model: LeNet5, device: torch.device) -> str:
    # 1) Lecture + gris + blur
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 2) Seuillage inversé (chiffres blancs)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # 3) Détection des contours externes
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digits = []
    # 4) Trier de gauche à droite
    for c in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(c)
        # ignorer trop petits éléments
        if w * h < 200:  
            continue
        roi = thresh[y:y+h, x:x+w]
        tensor = preprocess_digit(roi).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred = out.argmax(dim=1).item()
        digits.append(str(pred))

    return ''.join(digits)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help="Chemin vers le .pth entraîné")
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
