# Guide d'Optimisation - Bank Marketing Classification

## Scripts d'Optimisation Disponibles

### 1. `ultimate_optimizer.py` ⭐ RECOMMANDÉ
**Objectif**: Solution complète combinant toutes les techniques
**Score attendu**: 0.968+
**Temps d'exécution**: 45-90 minutes

```bash
cd src
python3 ultimate_optimizer.py
```

### 2. `ultra_fine_tuning.py`
**Objectif**: Optimisation automatique avec Optuna
**Score attendu**: 0.966+
**Temps d'exécution**: 2-3 heures

```bash
pip install optuna
python3 ultra_fine_tuning.py
```

### 3. `advanced_feature_engineering.py`
**Objectif**: Features de niveau expert
**Score attendu**: 0.967+
**Temps d'exécution**: 30-60 minutes

```bash
python3 advanced_feature_engineering.py
```

## Techniques d'Optimisation

### Feature Engineering
- Target encoding avec smoothing
- Features cycliques temporelles
- Clustering-based features
- Interactions polynomiales
- Transformations mathématiques

### Modélisation
- Ensemble de 5 modèles
- Validation croisée 10-fold
- Early stopping optimisé
- Pondération intelligente

### Hyperparamètres Optimaux
Voir les scripts pour les configurations détaillées.

## Amélioration Continue

Pour pousser encore plus loin:
1. Augmentez `n_trials` dans ultra_fine_tuning.py
2. Testez différents nombres de clusters
3. Modifiez les poids d'ensemble manuellement
4. Experimentez avec de nouvelles features
