#!/usr/bin/env python3
"""
Classification binaire pour le dataset bancaire
Playground Series S5E8 - Kaggle Competition
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BankClassifier:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        """Charger les datasets"""
        print("Chargement des données...")
        self.train_df = pd.read_csv('playground-series-s5e8/train.csv')
        self.test_df = pd.read_csv('playground-series-s5e8/test.csv')
        self.sample_submission = pd.read_csv('playground-series-s5e8/sample_submission.csv')
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
    def explore_data(self):
        """Exploration des données"""
        print("\n=== EXPLORATION DES DONNÉES ===")
        
        # Distribution de la variable cible
        print(f"\nDistribution de la variable cible:")
        print(self.train_df['y'].value_counts())
        print(f"Pourcentage de classe positive: {self.train_df['y'].mean():.2%}")
        
        # Informations générales
        print(f"\nInformations sur le dataset:")
        print(self.train_df.info())
        
        # Valeurs manquantes
        print(f"\nValeurs manquantes:")
        print(self.train_df.isnull().sum())
        
        # Statistiques descriptives pour les variables numériques
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        print(f"\nStatistiques descriptives (variables numériques):")
        print(self.train_df[numeric_cols].describe())
        
        # Variables catégorielles
        categorical_cols = self.train_df.select_dtypes(include=['object']).columns
        print(f"\nVariables catégorielles et leurs valeurs uniques:")
        for col in categorical_cols:
            print(f"{col}: {self.train_df[col].nunique()} valeurs uniques")
            
    def visualize_data(self):
        """Visualisation des données"""
        print("\n=== VISUALISATION DES DONNÉES ===")
        
        # Distribution de la variable cible
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        self.train_df['y'].value_counts().plot(kind='bar')
        plt.title('Distribution de la variable cible')
        plt.xlabel('y')
        plt.ylabel('Count')
        
        # Distribution de l'âge
        plt.subplot(2, 3, 2)
        plt.hist(self.train_df['age'], bins=30, alpha=0.7)
        plt.title('Distribution de l\'âge')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # Balance par classe
        plt.subplot(2, 3, 3)
        self.train_df.boxplot(column='balance', by='y', ax=plt.gca())
        plt.title('Balance par classe')
        plt.suptitle('')
        
        # Job distribution
        plt.subplot(2, 3, 4)
        job_counts = self.train_df['job'].value_counts().head(10)
        job_counts.plot(kind='bar')
        plt.title('Top 10 Jobs')
        plt.xticks(rotation=45)
        
        # Education vs target
        plt.subplot(2, 3, 5)
        education_target = pd.crosstab(self.train_df['education'], self.train_df['y'], normalize='index')
        education_target.plot(kind='bar', stacked=True)
        plt.title('Education vs Target')
        plt.xticks(rotation=45)
        
        # Duration distribution
        plt.subplot(2, 3, 6)
        plt.hist(self.train_df['duration'], bins=50, alpha=0.7)
        plt.title('Distribution de la durée')
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Préprocessing des données"""
        print("\n=== PRÉPROCESSING DES DONNÉES ===")
        
        # Séparer les features et la target
        X = self.train_df.drop(['id', 'y'], axis=1)
        y = self.train_df['y']
        X_test = self.test_df.drop(['id'], axis=1)
        
        # Identifier les colonnes catégorielles et numériques
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        print(f"Colonnes catégorielles: {list(categorical_cols)}")
        print(f"Colonnes numériques: {list(numerical_cols)}")
        
        # Encoder les variables catégorielles
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            self.label_encoders[col] = le
            
        # Feature engineering
        X = self.feature_engineering(X)
        X_test = self.feature_engineering(X_test)
        
        # Normalisation des variables numériques
        X_scaled = self.scaler.fit_transform(X)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convertir en DataFrame pour garder les noms de colonnes
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_scaled, y, X_test_scaled
        
    def feature_engineering(self, df):
        """Création de nouvelles features"""
        df = df.copy()
        
        # Ratio balance/age
        df['balance_age_ratio'] = df['balance'] / (df['age'] + 1)
        
        # Indicateur de balance positive/négative
        df['balance_positive'] = (df['balance'] > 0).astype(int)
        
        # Durée en minutes
        df['duration_minutes'] = df['duration'] / 60
        
        # Combinaison housing + loan
        df['housing_loan'] = df['housing'] + df['loan']
        
        # Age en catégories
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])
        df['age_group'] = df['age_group'].astype(int)
        
        return df
        
    def train_models(self, X, y):
        """Entraîner plusieurs modèles"""
        print("\n=== ENTRAÎNEMENT DES MODÈLES ===")
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nEntraînement {name}...")
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Métriques
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
        # Sélectionner le meilleur modèle
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        self.model = results[best_model_name]['model']
        
        print(f"\nMeilleur modèle: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
        
        return results
        
    def generate_predictions(self, X_test):
        """Générer les prédictions pour le test set"""
        print("\n=== GÉNÉRATION DES PRÉDICTIONS ===")
        
        # Prédictions probabilistes
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Créer le fichier de soumission
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            'y': y_pred_proba
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Fichier de soumission sauvegardé: submission.csv")
        
        return submission
        
    def run_complete_pipeline(self):
        """Exécuter le pipeline complet"""
        print("=== PIPELINE DE CLASSIFICATION BANCAIRE ===")
        
        # 1. Charger les données
        self.load_data()
        
        # 2. Explorer les données
        self.explore_data()
        
        # 3. Visualiser les données
        self.visualize_data()
        
        # 4. Préprocessing
        X, y, X_test = self.preprocess_data()
        
        # 5. Entraîner les modèles
        results = self.train_models(X, y)
        
        # 6. Générer les prédictions
        submission = self.generate_predictions(X_test)
        
        print("\n=== PIPELINE TERMINÉ ===")
        print(f"Fichier de soumission créé avec {len(submission)} prédictions")
        
        return submission

if __name__ == "__main__":
    # Créer et exécuter le classificateur
    classifier = BankClassifier()
    submission = classifier.run_complete_pipeline()
