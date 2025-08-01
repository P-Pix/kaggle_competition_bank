#!/usr/bin/env python3
"""
Solution ULTIME pour améliorer le score 0.96433
Combine toutes les techniques d'optimisation avancées
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
except ImportError:
    print("CatBoost non disponible, utilisation d'alternatives")
    CatBoostClassifier = None

import warnings
warnings.filterwarnings('ignore')

class UltimateBankOptimizer:
    def __init__(self):
        self.best_score = 0.96433  # Score à battre
        
    def load_data(self):
        """Charger les données"""
        train_df = pd.read_csv('playground-series-s5e8/train.csv')
        test_df = pd.read_csv('playground-series-s5e8/test.csv')
        return train_df, test_df
    
    def ultimate_feature_engineering(self, train_df, test_df):
        """Feature engineering ultime combinant toutes les techniques"""
        print("=== FEATURE ENGINEERING ULTIME ===")
        
        X_train = train_df.drop(['id', 'y'], axis=1)
        y_train = train_df['y']
        X_test = test_df.drop(['id'], axis=1)
        
        # 1. Encodage Label + Target Encoding hybride
        le_dict = {}
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            le_dict[col] = le
            
            # Target encoding avec smoothing
            category_means = y_train.groupby(X_train[col]).mean()
            global_mean = y_train.mean()
            category_counts = X_train[col].value_counts()
            alpha = 100  # Smoothing plus fort
            smoothed_means = (category_counts * category_means + alpha * global_mean) / (category_counts + alpha)
            
            X_train[f'{col}_target_smooth'] = X_train[col].map(smoothed_means)
            X_test[f'{col}_target_smooth'] = X_test[col].map(smoothed_means).fillna(global_mean)
        
        # 2. Features mathématiques avancées
        # Transformations Box-Cox approximées
        X_train['age_sqrt'] = np.sqrt(X_train['age'])
        X_test['age_sqrt'] = np.sqrt(X_test['age'])
        
        X_train['balance_cbrt'] = np.cbrt(X_train['balance'])
        X_test['balance_cbrt'] = np.cbrt(X_test['balance'])
        
        X_train['duration_log1p'] = np.log1p(X_train['duration'])
        X_test['duration_log1p'] = np.log1p(X_test['duration'])
        
        # 3. Interactions de haut niveau
        X_train['age_balance_duration'] = X_train['age'] * X_train['balance'] * X_train['duration']
        X_test['age_balance_duration'] = X_test['age'] * X_test['balance'] * X_test['duration']
        
        X_train['complex_ratio'] = (X_train['balance'] * X_train['duration']) / (X_train['age'] * X_train['campaign'] + 1)
        X_test['complex_ratio'] = (X_test['balance'] * X_test['duration']) / (X_test['age'] * X_test['campaign'] + 1)
        
        # 4. Features de rang et percentiles
        for col in ['age', 'balance', 'duration', 'campaign']:
            X_train[f'{col}_rank'] = X_train[col].rank(pct=True)
            X_test[f'{col}_rank'] = X_test[col].rank(pct=True)
            
            # Binning adaptatif
            X_train[f'{col}_qcut'] = pd.qcut(X_train[col], q=20, labels=False, duplicates='drop')
            X_test[f'{col}_qcut'] = pd.qcut(X_test[col], q=20, labels=False, duplicates='drop')
        
        # 5. Features temporelles avancées
        # Cycliques
        X_train['month_sin'] = np.sin(2 * np.pi * X_train['month'] / 12)
        X_train['month_cos'] = np.cos(2 * np.pi * X_train['month'] / 12)
        X_test['month_sin'] = np.sin(2 * np.pi * X_test['month'] / 12)
        X_test['month_cos'] = np.cos(2 * np.pi * X_test['month'] / 12)
        
        X_train['day_sin'] = np.sin(2 * np.pi * X_train['day'] / 31)
        X_train['day_cos'] = np.cos(2 * np.pi * X_train['day'] / 31)
        X_test['day_sin'] = np.sin(2 * np.pi * X_test['day'] / 31)
        X_test['day_cos'] = np.cos(2 * np.pi * X_test['day'] / 31)
        
        # 6. Features de fréquence et rareté
        for col in categorical_cols:
            value_counts = X_train[col].value_counts()
            X_train[f'{col}_freq'] = X_train[col].map(value_counts)
            X_test[f'{col}_freq'] = X_test[col].map(value_counts).fillna(1)
            
            X_train[f'{col}_rarity'] = 1 / X_train[f'{col}_freq']
            X_test[f'{col}_rarity'] = 1 / X_test[f'{col}_freq']
        
        # 7. Clustering-based features
        from sklearn.cluster import KMeans
        
        numeric_features = ['age', 'balance', 'duration', 'campaign']
        kmeans = KMeans(n_clusters=8, random_state=42)
        
        X_train['cluster'] = kmeans.fit_predict(X_train[numeric_features])
        X_test['cluster'] = kmeans.predict(X_test[numeric_features])
        
        # Distance aux centroïdes
        train_distances = kmeans.transform(X_train[numeric_features])
        test_distances = kmeans.transform(X_test[numeric_features])
        
        X_train['min_cluster_dist'] = train_distances.min(axis=1)
        X_train['mean_cluster_dist'] = train_distances.mean(axis=1)
        X_test['min_cluster_dist'] = test_distances.min(axis=1)
        X_test['mean_cluster_dist'] = test_distances.mean(axis=1)
        
        print(f"Features créées: Train {X_train.shape[1]}, Test {X_test.shape[1]}")
        
        return X_train, X_test, y_train
    
    def get_ultimate_models(self):
        """Modèles ultra-optimisés avec les meilleurs hyperparamètres"""
        models = {
            'XGBoost_Ultra': xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                colsample_bynode=0.8,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'LightGBM_Ultra': lgb.LGBMClassifier(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_samples=20,
                num_leaves=100,
                boosting_type='gbdt',
                objective='binary',
                random_state=42,
                verbose=-1
            ),
            
            'XGBoost_Deep': xgb.XGBClassifier(
                n_estimators=800,
                max_depth=12,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'LightGBM_Deep': lgb.LGBMClassifier(
                n_estimators=800,
                max_depth=12,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                verbose=-1
            )
        }
        
        # Ajouter CatBoost si disponible
        if CatBoostClassifier:
            models['CatBoost_Ultra'] = CatBoostClassifier(
                iterations=1000,
                depth=8,
                learning_rate=0.02,
                l2_leaf_reg=5,
                border_count=128,
                random_seed=42,
                verbose=False
            )
        
        return models
    
    def evaluate_and_ensemble(self, X_train, y_train, X_test):
        """Évaluation et création d'ensemble optimal"""
        print("\n=== ÉVALUATION ET ENSEMBLE ===")
        
        models = self.get_ultimate_models()
        results = {}
        predictions = []
        
        # Évaluation avec CV 10-fold
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\nÉvaluation {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            results[name] = {
                'model': model,
                'cv_mean': mean_score,
                'cv_std': std_score,
                'cv_scores': cv_scores
            }
            
            print(f"CV AUC: {mean_score:.5f} (+/- {std_score * 2:.5f})")
            
            # Entraîner sur toutes les données et prédire avec early stopping si XGBoost
            if 'XGBoost' in name:
                # Créer un split pour early stopping
                from sklearn.model_selection import train_test_split
                X_temp_train, X_temp_val, y_temp_train, y_temp_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
                )
                
                # Configurer l'early stopping
                model.set_params(early_stopping_rounds=50)
                model.fit(
                    X_temp_train, y_temp_train,
                    eval_set=[(X_temp_val, y_temp_val)],
                    verbose=False
                )
                
                # Réentraîner sur toutes les données avec le nombre d'estimateurs optimal
                optimal_rounds = model.get_booster().best_iteration + 1
                model.set_params(n_estimators=optimal_rounds, early_stopping_rounds=None)
                model.fit(X_train, y_train)
            else:
                # Pour les autres modèles, entraînement normal
                model.fit(X_train, y_train)
                
            pred = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
        
        # Créer ensemble pondéré basé sur les performances CV
        scores = [results[name]['cv_mean'] for name in models.keys()]
        weights = np.array(scores)
        weights = weights / weights.sum()
        
        # Ensemble final
        ensemble_pred = np.average(predictions, weights=weights, axis=0)
        
        # Estimation du score d'ensemble
        ensemble_score = np.average(scores, weights=weights)
        
        print(f"\n🎯 RÉSULTATS FINAUX:")
        for name, result in results.items():
            print(f"{name}: {result['cv_mean']:.5f}")
        
        print(f"\nEnsemble Score Estimé: {ensemble_score:.5f}")
        print(f"Amélioration par rapport à {self.best_score}: {ensemble_score - self.best_score:+.5f}")
        
        return ensemble_pred, ensemble_score, results
    
    def run_ultimate_optimization(self):
        """Exécuter l'optimisation ultime"""
        print("=" * 60)
        print("🚀 OPTIMISATION ULTIME - OBJECTIF: BATTRE 0.96433")
        print("=" * 60)
        
        # Charger les données
        train_df, test_df = self.load_data()
        
        # Feature engineering ultime
        X_train, X_test, y_train = self.ultimate_feature_engineering(train_df, test_df)
        
        # Normalisation adaptative
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        X_train_transformed = qt.fit_transform(X_train)
        X_test_transformed = qt.transform(X_test)
        
        X_train = pd.DataFrame(X_train_transformed, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_transformed, columns=X_test.columns)
        
        # Évaluation et ensemble
        final_predictions, estimated_score, results = self.evaluate_and_ensemble(
            X_train, y_train, X_test
        )
        
        # Sauvegarder le résultat
        submission = pd.DataFrame({
            'id': test_df['id'],
            'y': final_predictions
        })
        
        filename = f'ultimate_submission_{estimated_score:.5f}.csv'
        submission.to_csv(filename, index=False)
        
        print(f"\n✨ MISSION ACCOMPLIE!")
        print(f"Fichier sauvegardé: {filename}")
        
        if estimated_score > self.best_score:
            print(f"🎉 OBJECTIF ATTEINT! Amélioration: +{estimated_score - self.best_score:.5f}")
        else:
            print(f"⚠️  Score estimé: {estimated_score:.5f} (besoin de plus d'optimisation)")
        
        return submission, estimated_score

if __name__ == "__main__":
    optimizer = UltimateBankOptimizer()
    submission, score = optimizer.run_ultimate_optimization()
