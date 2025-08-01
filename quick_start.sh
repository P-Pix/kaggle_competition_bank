#!/bin/bash
# Script de démarrage rapide pour la compétition Bank Marketing

echo "🚀 Bank Marketing Classification - Quick Start"
echo "=============================================="

# Vérifier les dépendances
echo "📋 Vérification des dépendances..."
python3 -c "import pandas, numpy, sklearn, xgboost, lightgbm; print('✅ Toutes les dépendances sont installées')" 2>/dev/null || {
    echo "❌ Dépendances manquantes. Installation..."
    pip install -r docs/requirements.txt
}

# Vérifier les données
if [ -d "playground-series-s5e8" ] && [ -f "playground-series-s5e8/train.csv" ]; then
    echo "✅ Données trouvées"
else
    echo "❌ Données manquantes dans playground-series-s5e8/"
    echo "   Téléchargez les données depuis Kaggle et placez-les dans ce dossier"
    exit 1
fi

# Créer les dossiers de sortie si nécessaire
mkdir -p submissions models

echo "🎯 Lancement de la solution optimale..."
cd src
python3 ultimate_optimizer.py

echo "✨ Terminé! Vérifiez le fichier de soumission dans submissions/"
