#!/usr/bin/env python3
"""
Classification binaire optimisée pour le dataset bancaire
Playground Series S5E8 - Kaggle Competition
Version avec fine-tuning avancé pour améliorer le score de 0.96433
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
warnings.filterwarnings('ignore')

class OptimizedBankClassifier:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.quantile_transformer = QuantileTransformer(output_distribution='normal')
        self.feature_selector = None
        self.model = None
        
    def load_data(self):
        """Charger les datasets"""
        print("Chargement des données...")
        self.train_df = pd.read_csv('playground-series-s5e8/train.csv')
        self.test_df = pd.read_csv('playground-series-s5e8/test.csv')
        self.sample_submission = pd.read_csv('playground-series-s5e8/sample_submission.csv')
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
    def advanced_feature_engineering(self, df):
        """Feature engineering avancé"""
        df = df.copy()
        
        # Features existantes améliorées
        df['balance_age_ratio'] = df['balance'] / (df['age'] + 1)
        df['balance_positive'] = (df['balance'] > 0).astype(int)
        df['balance_negative'] = (df['balance'] < 0).astype(int)
        df['balance_zero'] = (df['balance'] == 0).astype(int)
        
        # Transformations log pour les variables asymétriques
        df['duration_log'] = np.log1p(df['duration'])
        df['campaign_log'] = np.log1p(df['campaign'])
        df['balance_abs_log'] = np.log1p(np.abs(df['balance']))
        
        # Binning intelligent
        df['age_bins'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 60, 100], labels=[0, 1, 2, 3, 4])
        df['balance_bins'] = pd.qcut(df['balance'], q=10, labels=False, duplicates='drop')
        df['duration_bins'] = pd.qcut(df['duration'], q=10, labels=False, duplicates='drop')
        
        # Interactions importantes
        df['age_balance_interaction'] = df['age'] * df['balance_positive']
        df['duration_campaign_ratio'] = df['duration'] / (df['campaign'] + 1)
        df['housing_loan_interaction'] = df['housing'] * df['loan']
        
        # Features temporelles
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)  # juin, juillet, août
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)  # déc, jan, fév
        df['is_weekend'] = (df['day'] % 7).isin([5, 6]).astype(int)  # approximation
        
        # Ratios avancés
        df['previous_success_rate'] = np.where(df['previous'] > 0, 
                                             (df['poutcome'] == 1).astype(int), 0)
        df['contact_efficiency'] = df['duration'] / (df['campaign'] + 1)
        
        # Features basées sur les moyennes de groupe
        if hasattr(self, 'job_means'):
            df['job_balance_mean'] = df['job'].map(self.job_means)
            df['education_duration_mean'] = df['education'].map(self.education_means)
        
        return df
        
    def create_target_encoding(self, X, y):
        """Créer des encodages basés sur la target"""
        # Moyennes par job
        job_target_mean = y.groupby(X['job']).mean()
        self.job_means = job_target_mean.to_dict()
        
        # Moyennes par education
        education_duration = X['duration'].groupby(X['education']).mean()
        self.education_means = education_duration.to_dict()
        
    def preprocess_data(self):
        """Préprocessing avancé des données"""
        print("\n=== PRÉPROCESSING AVANCÉ ===")
        
        # Séparer les features et la target
        X = self.train_df.drop(['id', 'y'], axis=1)
        y = self.train_df['y']
        X_test = self.test_df.drop(['id'], axis=1)
        
        # Encoder les variables catégorielles
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            self.label_encoders[col] = le
        
        # Créer des encodages basés sur la target
        self.create_target_encoding(X, y)
        
        # Feature engineering avancé
        X = self.advanced_feature_engineering(X)
        X_test = self.advanced_feature_engineering(X_test)
        
        # Traitement des outliers avec quantile transformation
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_quantile = self.quantile_transformer.fit_transform(X[numerical_cols])
        X_test_quantile = self.quantile_transformer.transform(X_test[numerical_cols])
        
        # Combiner avec les features catégorielles
        X_processed = pd.DataFrame(X_quantile, columns=numerical_cols)
        X_test_processed = pd.DataFrame(X_test_quantile, columns=numerical_cols)
        
        # Sélection de features
        self.feature_selector = SelectKBest(f_classif, k=min(50, X_processed.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_processed, y)
        X_test_selected = self.feature_selector.transform(X_test_processed)
        
        # Conversion en DataFrame
        selected_features = X_processed.columns[self.feature_selector.get_support()]
        X_final = pd.DataFrame(X_selected, columns=selected_features)
        X_test_final = pd.DataFrame(X_test_selected, columns=selected_features)
        
        print(f"Features sélectionnées: {len(selected_features)}")
        
        return X_final, y, X_test_final
        
    def get_optimized_models(self):
        """Modèles avec hyperparamètres optimisés"""
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            ),
            'RandomForest_Optimized': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Optimized': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }
        return models
        
    def train_with_hyperparameter_tuning(self, X, y):
        """Entraînement avec tuning des hyperparamètres"""
        print("\n=== ENTRAÎNEMENT AVEC HYPERPARAMETER TUNING ===")
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = self.get_optimized_models()
        results = {}
        
        for name, model in models.items():
            print(f"\nEntraînement {name}...")
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # AUC
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            results[name] = {
                'model': model,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_proba
            }
            
            print(f"AUC Validation: {auc:.5f}")
            print(f"CV AUC: {cv_scores.mean():.5f} (+/- {cv_scores.std() * 2:.5f})")
        
        # Créer un ensemble (Voting)
        print("\n=== CRÉATION D'UN ENSEMBLE ===")
        voting_models = [(name, results[name]['model']) for name in ['XGBoost', 'LightGBM', 'RandomForest_Optimized']]
        
        voting_classifier = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        voting_classifier.fit(X_train, y_train)
        voting_pred_proba = voting_classifier.predict_proba(X_val)[:, 1]
        voting_auc = roc_auc_score(y_val, voting_pred_proba)
        
        # CV pour le voting
        voting_cv_scores = cross_val_score(
            voting_classifier, X, y,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        results['VotingClassifier'] = {
            'model': voting_classifier,
            'auc': voting_auc,
            'cv_mean': voting_cv_scores.mean(),
            'cv_std': voting_cv_scores.std(),
            'predictions': voting_pred_proba
        }
        
        print(f"Ensemble AUC Validation: {voting_auc:.5f}")
        print(f"Ensemble CV AUC: {voting_cv_scores.mean():.5f} (+/- {voting_cv_scores.std() * 2:.5f})")
        
        # Sélectionner le meilleur modèle
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.model = results[best_model_name]['model']
        
        print(f"\n🏆 Meilleur modèle: {best_model_name}")
        print(f"CV AUC: {results[best_model_name]['cv_mean']:.5f}")
        
        return results
        
    def create_stacking_ensemble(self, X, y):
        """Créer un ensemble par stacking"""
        print("\n=== CRÉATION D'UN STACKING ENSEMBLE ===")
        
        # Modèles de base
        base_models = [
            ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))
        ]
        
        # Meta-modèle
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Stacking classifier
        stacking_classifier = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        # Entraînement et évaluation
        cv_scores = cross_val_score(
            stacking_classifier, X, y,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"Stacking CV AUC: {cv_scores.mean():.5f} (+/- {cv_scores.std() * 2:.5f})")
        
        # Entraîner sur toutes les données
        stacking_classifier.fit(X, y)
        
        return stacking_classifier, cv_scores.mean()
        
    def generate_optimized_predictions(self, X_test):
        """Générer les prédictions optimisées"""
        print("\n=== GÉNÉRATION DES PRÉDICTIONS OPTIMISÉES ===")
        
        # Prédictions du meilleur modèle
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Créer le fichier de soumission
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            'y': y_pred_proba
        })
        
        submission.to_csv('optimized_submission.csv', index=False)
        print("Fichier de soumission optimisé sauvegardé: optimized_submission.csv")
        
        return submission
        
    def run_optimized_pipeline(self):
        """Exécuter le pipeline optimisé complet"""
        print("=== PIPELINE OPTIMISÉ DE CLASSIFICATION BANCAIRE ===")
        print("Objectif: Améliorer le score de 0.96433")
        
        # 1. Charger les données
        self.load_data()
        
        # 2. Préprocessing avancé
        X, y, X_test = self.preprocess_data()
        
        # 3. Entraîner avec tuning
        results = self.train_with_hyperparameter_tuning(X, y)
        
        # 4. Tester le stacking
        stacking_model, stacking_score = self.create_stacking_ensemble(X, y)
        
        # 5. Comparer et choisir le meilleur
        best_cv_score = max([r['cv_mean'] for r in results.values()])
        
        if stacking_score > best_cv_score:
            print(f"\n🎯 Stacking sélectionné! Score: {stacking_score:.5f}")
            self.model = stacking_model
        else:
            print(f"\n🎯 Modèle individuel conservé! Score: {best_cv_score:.5f}")
        
        # 6. Générer les prédictions
        submission = self.generate_optimized_predictions(X_test)
        
        print("\n=== PIPELINE OPTIMISÉ TERMINÉ ===")
        print(f"Amélioration ciblée du score 0.96433")
        
        return submission

if __name__ == "__main__":
    # Créer et exécuter le classificateur optimisé
    classifier = OptimizedBankClassifier()
    submission = classifier.run_optimized_pipeline()
