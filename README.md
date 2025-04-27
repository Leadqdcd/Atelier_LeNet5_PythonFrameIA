# LeNet-5 Digit Recognition

Reconnaissance de chiffres manuscrits avec LeNet-5 de Yann LeCun  
Entraîné sur **MNIST** (0–9), download automatique.

## Arborescence

Atelier_LeNet5_PythonFrameIA/ 
├─ data/ ← MNIST auto-téléchargé ici 
├─ outputs/ ← modèles entraînés (.pth)
 └─ src/ ← code source
 ├─ model.py 
 ├─ train.py 
 └─ inference.py

# Note : 
pip install -r requirements.txt
Nous utilisons MNIST pour sa légèreté et sa simplicité d’accès, mais l’architecture LeNet-5 reste inchangée.