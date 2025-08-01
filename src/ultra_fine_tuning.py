#!/usr/bin/env python3
"""
Fine-tuning ultra-optimisé pour améliorer le score 0.96433
Techniques avancées de ML pour la compétition bancaire
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class UltraOptimizer:
    def __init__(self):
        self.best_params = {}
        
    def load_and_prepare_data(self):
        """Charger et préparer les données pour l'optimisation"""
        train_df = pd.read_csv('playground-series-s5e8/train.csv')
        test_df = pd.read_csv('playground-series-s5e8/test.csv')
        
        X = train_df.drop(['id', 'y'], axis=1)
        y = train_df['y']
        X_test = test_df.drop(['id'], axis=1)
        
        # Encodage simple et rapide pour l'optimisation
        from sklearn.preprocessing import LabelEncoder
        le_dict = {}
        
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            le_dict[col] = le
            
        return X, y, X_test, train_df['id'], test_df['id']
    
    def objective_xgboost(self, trial):
        """Fonction objectif pour XGBoost avec Optuna"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**params)
        
        cv_scores = cross_val_score(
            model, self.X, self.y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    def objective_lightgbm(self, trial):
        """Fonction objectif pour LightGBM avec Optuna"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'num_leaves': trial.suggest_int('num_leaves', 31, 300),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        
        cv_scores = cross_val_score(
            model, self.X, self.y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    def objective_catboost(self, trial):
        """Fonction objectif pour CatBoost avec Optuna"""
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42,
            'verbose': False
        }
        
        model = CatBoostClassifier(**params)
        
        cv_scores = cross_val_score(
            model, self.X, self.y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    def optimize_models(self, n_trials=100):
        """Optimiser les hyperparamètres avec Optuna"""
        print("=== OPTIMISATION AVEC OPTUNA ===")
        
        results = {}
        
        # XGBoost
        print("\n🔧 Optimisation XGBoost...")
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(self.objective_xgboost, n_trials=n_trials)
        results['XGBoost'] = {
            'best_score': study_xgb.best_value,
            'best_params': study_xgb.best_params
        }
        print(f"Meilleur score XGBoost: {study_xgb.best_value:.5f}")
        
        # LightGBM
        print("\n🔧 Optimisation LightGBM...")
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(self.objective_lightgbm, n_trials=n_trials)
        results['LightGBM'] = {
            'best_score': study_lgb.best_value,
            'best_params': study_lgb.best_params
        }
        print(f"Meilleur score LightGBM: {study_lgb.best_value:.5f}")
        
        # CatBoost
        print("\n🔧 Optimisation CatBoost...")
        study_cat = optuna.create_study(direction='maximize')
        study_cat.optimize(self.objective_catboost, n_trials=n_trials)
        results['CatBoost'] = {
            'best_score': study_cat.best_value,
            'best_params': study_cat.best_params
        }
        print(f"Meilleur score CatBoost: {study_cat.best_value:.5f}")
        
        return results
    
    def create_ultra_ensemble(self, optimization_results):
        """Créer un ensemble ultra-optimisé"""
        print("\n=== CRÉATION DE L'ENSEMBLE ULTRA-OPTIMISÉ ===")
        
        models = []
        weights = []
        
        for model_name, result in optimization_results.items():
            if model_name == 'XGBoost':
                model = xgb.XGBClassifier(**result['best_params'])
            elif model_name == 'LightGBM':
                model = lgb.LGBMClassifier(**result['best_params'])
            elif model_name == 'CatBoost':
                model = CatBoostClassifier(**result['best_params'])
            
            model.fit(self.X, self.y)
            models.append(model)
            weights.append(result['best_score'])
        
        # Normaliser les poids
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return models, weights
    
    def predict_ensemble(self, models, weights, X_test):
        """Prédictions d'ensemble pondéré"""
        predictions = []
        
        for model in models:
            pred = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
        
        # Moyenne pondérée
        ensemble_pred = np.average(predictions, weights=weights, axis=0)
        
        return ensemble_pred
    
    def run_ultra_optimization(self, n_trials=50):
        """Exécuter l'optimisation ultra-avancée"""
        print("=== ULTRA-OPTIMISATION POUR AMÉLIORER 0.96433 ===")
        
        # Charger les données
        self.X, self.y, self.X_test, train_ids, test_ids = self.load_and_prepare_data()
        
        # Optimiser les modèles
        results = self.optimize_models(n_trials)
        
        # Créer l'ensemble
        models, weights = self.create_ultra_ensemble(results)
        
        # Prédictions finales
        final_predictions = self.predict_ensemble(models, weights, self.X_test)
        
        # Sauvegarder
        submission = pd.DataFrame({
            'id': test_ids,
            'y': final_predictions
        })
        
        submission.to_csv('ultra_optimized_submission.csv', index=False)
        
        print(f"\n🎯 RÉSULTATS FINAUX:")
        for model_name, result in results.items():
            print(f"{model_name}: {result['best_score']:.5f}")
        
        print(f"\n✨ Ensemble ultra-optimisé créé!")
        print(f"Poids: {dict(zip(['XGBoost', 'LightGBM', 'CatBoost'], weights))}")
        print(f"Fichier: ultra_optimized_submission.csv")
        
        return submission

if __name__ == "__main__":
    optimizer = UltraOptimizer()
    submission = optimizer.run_ultra_optimization(n_trials=100)  # Augmentez à 200+ pour de meilleurs résultats
