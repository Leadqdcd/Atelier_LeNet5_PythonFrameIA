# LeNet-5 on NIST Special Database 19

Reconnaissance de chiffres manuscrits à l’aide du modèle LeNet-5 de Yann LeCun  
sur le jeu NIST SD-19 (0–9 uniquement).

## Arborescence

- `data/` : (sera rempli par le script)  
- `src/` : code source (model, dataset, entraînement, inference).  
- `outputs/` : modèles entraînés.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Télécharger le dataset

Rendez-vous sur le site officiel NIST ou exécutez le script suivant :

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

## Entraînement

```bash
cd src
python train.py \
  --data-dir ../data/by_class \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3 \
  --output ../outputs/lenet5_sd19.pth
```

## Inference

```bash
python inference.py \
  --model ../outputs/lenet5_sd19.pth \
  --image path/to/cheque.jpg
```