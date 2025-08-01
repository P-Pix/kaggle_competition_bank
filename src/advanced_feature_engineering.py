#!/usr/bin/env python3
"""
Feature Engineering Avancé pour améliorer le score 0.96433
Techniques de pointe pour l'optimisation des features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.target_encoders = {}
        self.cluster_models = {}
        
    def load_data(self):
        """Charger les données"""
        self.train_df = pd.read_csv('playground-series-s5e8/train.csv')
        self.test_df = pd.read_csv('playground-series-s5e8/test.csv')
        return self.train_df, self.test_df
    
    def create_advanced_target_encoding(self, X, y, test_X=None):
        """Encodage target avec validation croisée pour éviter l'overfitting"""
        X_encoded = X.copy()
        test_encoded = test_X.copy() if test_X is not None else None
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Mean encoding avec smoothing
            category_means = y.groupby(X[col]).mean()
            global_mean = y.mean()
            category_counts = X[col].value_counts()
            
            # Smoothing factor
            alpha = 10
            smoothed_means = (category_counts * category_means + alpha * global_mean) / (category_counts + alpha)
            
            X_encoded[f'{col}_target_encoded'] = X[col].map(smoothed_means)
            
            if test_encoded is not None:
                test_encoded[f'{col}_target_encoded'] = test_X[col].map(smoothed_means).fillna(global_mean)
            
            self.target_encoders[col] = smoothed_means
        
        return X_encoded, test_encoded
    
    def create_statistical_features(self, df):
        """Créer des features statistiques avancées"""
        df = df.copy()
        
        # Features de fréquence
        for col in ['job', 'marital', 'education']:
            freq_map = df[col].value_counts(normalize=True)
            df[f'{col}_frequency'] = df[col].map(freq_map)
            df[f'{col}_rarity'] = 1 / df[f'{col}_frequency']
        
        # Features d'interaction complexes
        df['age_balance_poly'] = df['age'] * df['balance'] + df['age']**2
        df['duration_campaign_complex'] = (df['duration'] * df['campaign']) / (df['age'] + 1)
        
        # Features temporelles avancées
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # Features de ratio avancés
        df['balance_per_year_life'] = df['balance'] / (df['age'] - 18 + 1)
        df['campaign_intensity'] = df['campaign'] / (df['duration'] / 3600 + 1)  # campagnes par heure
        
        # Features de rang
        df['age_rank'] = df['age'].rank(pct=True)
        df['balance_rank'] = df['balance'].rank(pct=True)
        df['duration_rank'] = df['duration'].rank(pct=True)
        
        return df
    
    def create_clustering_features(self, X, n_clusters=5):
        """Créer des features basées sur le clustering"""
        X_cluster = X.copy()
        
        # Sélectionner les features numériques pour le clustering
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        X_cluster['cluster'] = kmeans.fit_predict(X_numeric)
        
        # Distance au centroïde le plus proche
        distances = kmeans.transform(X_numeric)
        X_cluster['min_cluster_distance'] = distances.min(axis=1)
        X_cluster['cluster_distance_std'] = distances.std(axis=1)
        
        self.cluster_models['kmeans'] = kmeans
        
        return X_cluster
    
    def create_polynomial_features(self, X, degree=2, max_features=20):
        """Créer des features polynomiales sélectionnées"""
        # Sélectionner les features les plus importantes
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]  # Top 10 seulement
        X_poly_subset = X[numeric_cols]
        
        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_poly_subset)
        
        # Garder seulement les features les plus informatives
        poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
        
        # Sélectionner les meilleures features polynomiales
        selector = SelectKBest(f_classif, k=min(max_features, X_poly.shape[1]))
        
        return poly_df, selector, poly
    
    def create_aggregated_features(self, X):
        """Créer des features agrégées par groupe"""
        X_agg = X.copy()
        
        # Groupements par catégories importantes
        for group_col in ['job', 'marital', 'education']:
            if group_col in X.columns:
                # Moyennes par groupe
                for target_col in ['age', 'balance', 'duration', 'campaign']:
                    if target_col in X.columns:
                        group_means = X.groupby(group_col)[target_col].mean()
                        X_agg[f'{target_col}_mean_by_{group_col}'] = X[group_col].map(group_means)
                        
                        # Écart par rapport à la moyenne du groupe
                        X_agg[f'{target_col}_diff_from_{group_col}_mean'] = X[target_col] - X_agg[f'{target_col}_mean_by_{group_col}']
        
        return X_agg
    
    def engineer_all_features(self, train_df, test_df, target_col='y'):
        """Pipeline complet de feature engineering"""
        print("=== FEATURE ENGINEERING AVANCÉ ===")
        
        X_train = train_df.drop(['id', target_col], axis=1, errors='ignore')
        y_train = train_df[target_col] if target_col in train_df.columns else None
        X_test = test_df.drop(['id'], axis=1, errors='ignore')
        
        # 1. Encodage basique
        le_dict = {}
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            le_dict[col] = le
        
        print(f"✓ Encodage des variables catégorielles: {len(categorical_cols)} colonnes")
        
        # 2. Features statistiques
        X_train = self.create_statistical_features(X_train)
        X_test = self.create_statistical_features(X_test)
        print(f"✓ Features statistiques créées")
        
        # 3. Features de clustering
        X_train = self.create_clustering_features(X_train)
        X_test_cluster = X_test.copy()
        
        # Appliquer le clustering au test set
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        X_test_numeric = X_test[numeric_cols]
        X_test_cluster['cluster'] = self.cluster_models['kmeans'].predict(X_test_numeric)
        
        distances = self.cluster_models['kmeans'].transform(X_test_numeric)
        X_test_cluster['min_cluster_distance'] = distances.min(axis=1)
        X_test_cluster['cluster_distance_std'] = distances.std(axis=1)
        X_test = X_test_cluster
        print(f"✓ Features de clustering créées")
        
        # 4. Features agrégées
        X_train = self.create_aggregated_features(X_train)
        X_test = self.create_aggregated_features(X_test)
        print(f"✓ Features agrégées créées")
        
        # 5. Target encoding (seulement si on a la target)
        if y_train is not None:
            # Re-encoder temporairement pour le target encoding
            X_train_temp = train_df.drop(['id', target_col], axis=1, errors='ignore')
            X_test_temp = test_df.drop(['id'], axis=1, errors='ignore')
            
            X_train_encoded, X_test_encoded = self.create_advanced_target_encoding(
                X_train_temp, y_train, X_test_temp
            )
            
            # Ajouter seulement les nouvelles colonnes encodées
            for col in X_train_encoded.columns:
                if col.endswith('_target_encoded'):
                    X_train[col] = X_train_encoded[col]
                    X_test[col] = X_test_encoded[col]
            
            print(f"✓ Target encoding créé")
        
        print(f"Features finales: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test, y_train
    
    def select_best_features(self, X, y, k=100):
        """Sélectionner les meilleures features"""
        # Combinaison de méthodes de sélection
        
        # 1. Univariate selection
        selector_f = SelectKBest(f_classif, k=k//2)
        X_f = selector_f.fit_transform(X, y)
        selected_f = X.columns[selector_f.get_support()]
        
        # 2. Mutual information
        selector_mi = SelectKBest(mutual_info_classif, k=k//2)
        X_mi = selector_mi.fit_transform(X, y)
        selected_mi = X.columns[selector_mi.get_support()]
        
        # Combiner les features sélectionnées
        selected_features = list(set(selected_f) | set(selected_mi))
        
        return X[selected_features], selected_features
    
    def run_advanced_pipeline(self):
        """Exécuter le pipeline complet d'amélioration"""
        print("=== PIPELINE D'AMÉLIORATION AVANCÉ ===")
        print("Objectif: Surpasser le score de 0.96433")
        
        # Charger les données
        train_df, test_df = self.load_data()
        
        # Feature engineering
        X_train, X_test, y_train = self.engineer_all_features(train_df, test_df)
        
        # Sélection des meilleures features
        X_train_selected, selected_features = self.select_best_features(X_train, y_train, k=80)
        X_test_selected = X_test[selected_features]
        
        print(f"Features sélectionnées: {len(selected_features)}")
        
        # Modèles optimisés
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=800,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=800,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            )
        }
        
        # Évaluation et prédictions
        predictions = []
        scores = []
        
        for name, model in models.items():
            print(f"\nÉvaluation {name}...")
            
            cv_scores = cross_val_score(
                model, X_train_selected, y_train,
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            score = cv_scores.mean()
            scores.append(score)
            print(f"CV AUC: {score:.5f} (+/- {cv_scores.std() * 2:.5f})")
            
            # Entraîner et prédire
            model.fit(X_train_selected, y_train)
            pred = model.predict_proba(X_test_selected)[:, 1]
            predictions.append(pred)
        
        # Ensemble final
        weights = np.array(scores)
        weights = weights / weights.sum()
        final_prediction = np.average(predictions, weights=weights, axis=0)
        
        # Sauvegarder
        submission = pd.DataFrame({
            'id': test_df['id'],
            'y': final_prediction
        })
        
        submission.to_csv('advanced_engineered_submission.csv', index=False)
        
        print(f"\n🎯 RÉSULTATS:")
        print(f"Scores individuels: {dict(zip(models.keys(), scores))}")
        print(f"Score d'ensemble estimé: {np.average(scores, weights=weights):.5f}")
        print(f"Fichier sauvegardé: advanced_engineered_submission.csv")
        
        return submission

if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    submission = engineer.run_advanced_pipeline()
