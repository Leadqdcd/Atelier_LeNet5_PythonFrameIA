import argparse
import cv2
import numpy as np
import torch
from model import LeNet5

def preprocess_digit(roi: np.ndarray, size: int = 32) -> torch.Tensor:
    """
    Transforme un ROI binaire (255 = chiffre, 0 = fond) en tenseur (1×1×32×32)
    normalisé comme les données MNIST ((x−0.5)/0.5).
    """
    h, w = roi.shape
    S = max(h, w)
    # 1) Crée un carré noir de taille S×S
    square = np.zeros((S, S), dtype=np.uint8)
    # 2) Centre le ROI (chiffre blanc) dans ce carré
    y_off, x_off = (S - h) // 2, (S - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = roi

    # 3) Redimensionne à 32×32 pixels
    resized = cv2.resize(square, (size, size),
                         interpolation=cv2.INTER_AREA)
    # 4) Passe en float [0,1]
    arr = resized.astype(np.float32) / 255.0
    # 5) Normalise en [-1,1] comme pour l’entraînement
    arr = (arr - 0.5) / 0.5
    # 6) Convertit en tenseur PyTorch (1,1,32,32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def segment_and_predict(img_path: str, model: LeNet5, device: torch.device) -> str:
    """
    Lit l'image, segmente chaque caractère, applique le prétraitement
    et prédit la classe avec le modèle LeNet-5.
    """
    # 1) Lecture et conversion en niveaux de gris
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Lissage pour atténuer le bruit de fond
    gray = cv2.medianBlur(gray, 3)

    # 3) Seuillage inversé : chiffres en blanc (255), fond en noir (0)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # 4) Petite fermeture verticale pour reconnecter les bouts du "2"
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vert_kernel)

    # 5) Détection des contours externes (un contour = un chiffre potentiel)
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digits = []
    # 6) Parcours des contours triés de gauche à droite
    for c in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(c)
        # 7) Filtre les petits artefacts (bruit)
        if w * h < 200:
            continue
        # 8) Extraction du ROI binaire pour chaque chiffre
        roi = thresh[y:y+h, x:x+w]
        # 9) Prétraitement et mise en forme du tenseur
        tensor = preprocess_digit(roi).to(device)
        # 10) Prédiction sans calcul de gradients
        with torch.no_grad():
            out = model(tensor)
            pred = out.argmax(dim=1).item()
        digits.append(str(pred))

    # 11) Concatène tous les chiffres reconnus
    return ''.join(digits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help="Chemin vers le fichier .pth entraîné")
    parser.add_argument('--image', required=True,
                        help="Chemin vers la photo du chèque contenant les chiffres")
    args = parser.parse_args()

    # 1) Configuration de l’appareil (GPU si dispo, sinon CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) Chargement du modèle LeNet-5
    model = LeNet5(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()  # mode évaluation (désactive dropout, etc.)

    # 3) Segmenter et prédire la séquence de chiffres
    result = segment_and_predict(args.image, model, device)
    print("Montant reconnu :", result)

if __name__ == '__main__':
    main()
