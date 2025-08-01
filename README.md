# 🏆 Bank Marketing Classification - Kaggle Competition

**Score**: 0.96790 | **Classement**: 92ème mondial (15h après ouverture)

Ce repository contient ma solution complète pour la compétition Kaggle "Playground Series S5E8 - Bank Marketing Classification".

## 🎯 Résultats

- **Score final**: 0.96790 AUC
- **Classement**: 92ème place mondiale
- **Amélioration**: +0.00357 par rapport au baseline (0.96433)
- **Temps**: 15 heures après l'ouverture de la compétition

## 📁 Structure du Projet

```
├── src/                           # Scripts Python principaux
│   ├── bank_classification.py     # Version de base
│   ├── optimized_bank_classification.py  # Version optimisée
│   ├── ultimate_optimizer.py      # Solution finale (BEST)
│   ├── ultra_fine_tuning.py      # Fine-tuning avec Optuna
│   └── advanced_feature_engineering.py  # Features avancées
│
├── notebooks/                     # Jupyter Notebooks
│   ├── bank_classification_notebook.ipynb  # Analyse complète
│   └── data_viz.ipynb            # Visualisations et EDA
│
├── docs/                          # Documentation
│   ├── requirements.txt          # Dépendances Python
│   └── OPTIMIZATION_GUIDE.md     # Guide d'optimisation
│
├── submissions/                   # Fichiers de soumission
│   ├── ultimate_submission_0.96648.csv  # Meilleure soumission
│   └── other_submissions/        # Autres tentatives
│
└── playground-series-s5e8/       # Données (non commitées)
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/P-Pix/kaggle_competition_bank.git
cd kaggle_competition_bank
pip install -r docs/requirements.txt
```

### 2. Télécharger les données
Placez les fichiers de données Kaggle dans `playground-series-s5e8/`

### 3. Exécuter la solution optimale
```bash
cd src
python3 ultimate_optimizer.py
```

## 💡 Stratégie Gagnante

### Feature Engineering Avancé
- **Target Encoding** avec smoothing anti-overfitting
- **Features cycliques** temporelles (sin/cos)
- **Clustering K-means** pour patterns cachés
- **Interactions complexes** entre variables importantes
- **Transformations mathématiques** (log, sqrt, cbrt)

### Modèles Ensemble
- **XGBoost Ultra** (score: 0.96791)
- **LightGBM Ultra** (score: 0.96777) 
- **XGBoost Deep** (score: 0.96746)
- **LightGBM Deep** (score: 0.96447)
- **CatBoost Ultra** (score: 0.96480)

### Ensemble Final
Moyenne pondérée basée sur la performance de validation croisée 10-fold.

## 🔧 Techniques Utilisées

### Data Processing
- QuantileTransformer pour normalisation robuste
- Label Encoding + Target Encoding hybride
- Gestion intelligente des outliers

### Model Optimization
- Validation croisée stratifiée 10-fold
- Early stopping avec validation split
- Régularisation adaptée aux données synthétiques

### Ensemble Strategy
- Pondération basée sur performance CV
- 5 modèles complémentaires
- Robustesse contre l'overfitting

## 📊 Insights Clés

1. **Données Synthétiques**: Nécessitent plus de régularisation que les données réelles
2. **Features Temporelles**: Month/Day cycliques très importants
3. **Target Encoding**: Crucial avec smoothing fort (α=100)
4. **Ensemble Diversity**: Modèles avec profondeurs différentes se complètent

## 🏅 Performance Détaillée

| Modèle | CV Score | Weight | Contribution |
|--------|----------|--------|--------------|
| XGBoost_Ultra | 0.96791 | 0.203 | ⭐ Meilleur |
| LightGBM_Ultra | 0.96777 | 0.202 | 🥈 Très proche |
| XGBoost_Deep | 0.96746 | 0.201 | 🥉 Solide |
| LightGBM_Deep | 0.96447 | 0.197 | 📊 Diversité |
| CatBoost_Ultra | 0.96480 | 0.197 | 🎯 Catégories |

**Ensemble Final**: 0.96648

## 🔄 Reproductibilité

Tous les scripts utilisent `random_state=42` pour la reproductibilité complète.

## 📝 Citation

Si vous utilisez ce code, merci de citer :
```
@misc{kaggle_bank_classification_2025,
  author = {Guillaume L.},
  title = {Bank Marketing Classification - 92nd Place Solution},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/P-Pix/kaggle_competition_bank}}
}
```

## 🤝 Contact

- **LinkedIn**: [Votre profil LinkedIn]
- **Kaggle**: [Votre profil Kaggle]
- **Email**: [Votre email]

---

⭐ **N'hésitez pas à star le repo si vous trouvez la solution utile !**
