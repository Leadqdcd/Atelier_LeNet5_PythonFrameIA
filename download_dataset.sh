#!/bin/bash
# Téléchargement automatique du dataset NIST SD-19 (chiffres seulement)
DATA_URL="https://s3.amazonaws.com/nist-sd19/by_class.zip"
mkdir -p data
wget -c $DATA_URL -P data
unzip -o data/by_class.zip -d data
