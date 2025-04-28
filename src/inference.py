import argparse
import cv2
import numpy as np
import torch
from model import LeNet5

def preprocess_digit(roi: np.ndarray, size: int = 32) -> torch.Tensor:
    # Place le ROI (255 = blanc) centré sur fond noir comme MNIST
    h, w = roi.shape
    S = max(h, w)
    square = np.zeros((S, S), dtype=np.uint8)
    y_off, x_off = (S - h) // 2, (S - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = roi

    # Redim + normalize → [-1,1]
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def segment_and_predict(img_path: str, model: LeNet5, device: torch.device) -> str:
    # 1) Lecture + gris
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Optionnel pour réduire artefacts papier
    gray = cv2.medianBlur(gray, 3)

    # 2) Seuillage inversé (chiffres blancs sur fond noir)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # 3) Fermeture verticale pour reconnecter les traits du '2'
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vert_kernel)

    # 4) Détection des contours externes
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digits = []
    # 5) Trier contours de gauche à droite
    for c in sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0]):
        x, y, w, h = cv2.boundingRect(c)
        # filtrer le bruit
        if w * h < 200:
            continue
        roi = thresh[y:y+h, x:x+w]  # blanc (255) sur noir (0)
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
