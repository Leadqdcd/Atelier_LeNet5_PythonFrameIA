import argparse
import cv2
import torch
from model import LeNet5

def segment_and_predict(img_path, model, device):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    for c in sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0]):
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 100: continue
        roi = thresh[y:y+h, x:x+w]
        size = max(w, h)
        square = 255 - cv2.copyMakeBorder(
            roi, (size-h)//2, (size-h)-(size-h)//2,
            (size-w)//2, (size-w)-(size-w)//2,
            cv2.BORDER_CONSTANT, value=0
        )
        digit = cv2.resize(square, (32,32))
        tensor = torch.from_numpy(digit.astype('float32')/255.0
                 ).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred = out.argmax(dim=1).item()
        digits.append(str(pred))
    return ''.join(digits)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    result = segment_and_predict(args.image, model, device)
    print("Montant reconnu :", result)

if __name__ == '__main__':
    main()
