# 👤 Membre 2 : Ingénieur Données & Pipeline de Prétraitement

> **Rôle :** Data Engineer & Preprocessing Specialist  
> **Périmètre :** Ingestion multi-formats, Jeux de données de référence, Nettoyage des données, Imputation des valeurs manquantes, Encodage catégoriel, Normalisation/Standardisation, Train/Test Split et Persistance du Pipeline Scikit-Learn.  
> **Stack Technique :** Python 3.10+, Pandas, NumPy, Scikit-Learn (`sklearn.preprocessing`, `sklearn.impute`, `sklearn.model_selection`, `sklearn.pipeline`), Joblib.

---

## 🎯 Objectifs de la Mission

En tant que **Membre 2**, vous êtes le garant de la qualité et de l'intégrité de la donnée. Sans un prétraitement irréprochable, aucun algorithme de Machine Learning ne peut fonctionner (*"Garbage In, Garbage Out"*).

Vous construisez le moteur complet d'ingestion et de préparation de données répondant directement aux fonctionnalités **1 (Interface Utilisateur Intuitive - Import)** et **3 (Gestion des Données)** du cahier des charges :
* Permettre à l'utilisateur d'importer n'importe quel fichier de données (ou d'utiliser des jeux de test intégrés).
* Fournir une interface interactive de nettoyage (valeurs nulles, doublons, types).
* Appliquer les transformations avancées (One-Hot, Scalers) sans fuite de données (*data leakage*).
* Exposer un pipeline sérialisé réutilisable par le Membre 4 pour la prédiction en direct.

---

## 📦 Modules & Applications Django Sous Votre Responsabilité

1. **`apps/datasets/`** : Ingestion, validation des fichiers, stockage, prévisualisation tabulaire et jeux de démo.
2. **`apps/preprocessing/`** : Moteur de nettoyage, imputation, encodage, normalisation, séparation Train/Test.
3. **`sample_data/`** : Répertoire de jeux de données prédéfinis (Iris, Titanic, Housing, Wine, Mall Customers).

---

## 📋 Liste Détaillée des Tâches (Roadmap Membre 2)

### Phase 1 : Module d'Ingestion Multi-Sources & Jeux de Démo
- [ ] **Tâche 1.1 : Formulaire d'Upload & Lecteur Universel**
  - Supporter les formats : `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`.
  - Implémenter la détection automatique du séparateur CSV (virgule, point-virgule, tabulation via `csv.Sniffer`).
  - Gérer l'encodage de caractères (tentative en `utf-8`, puis repli automatique en `latin-1` ou `cp1252`).
  - Permettre l'import via URL directe (téléchargement d'un lien brut GitHub/web).
- [ ] **Tâche 1.2 : Jeux de Données de Démonstration Intégrés (1-Click Load)**
  - Précharger des datasets classiques représentatifs des différents problèmes du sujet :
    - *Classification Binaire :* Titanic (`titanic.csv`)
    - *Classification Multi-classe :* Iris (`iris.csv`), Qualité du Vin (`wine.csv`)
    - *Régression :* Prix de l'immobilier (`california_housing.csv`)
    - *Clustering :* Segmentation Clients (`mall_customers.csv`)
  - Fournir un bouton "Charger un jeu d'exemple" dans l'UI pour tester instantanément la plateforme.
- [ ] **Tâche 1.3 : Inspection & Validation Préliminaire**
  - Contrôler la taille du fichier (limite paramétrable, ex: 50 Mo pour garantir la fluidité).
  - Générer un aperçu tabulaire paginé (les 10, 50 ou 100 premières lignes).
  - Détecter et afficher immédiatement : le nombre total de lignes, le nombre de colonnes, et la mémoire consommée.

### Phase 2 : Nettoyage des Données & Gestion des Valeurs Manquantes
- [ ] **Tâche 2.1 : Détection & Profiling des Anomalies**
  - Calculer par colonne le nombre et le pourcentage de valeurs manquantes (`NaN` / `None` / `?` / chaînes vides).
  - Identifier les lignes dupliquées et afficher le compte exact.
  - Détecter les types de colonnes (Numérique continue, Numérique discrète, Catégorielle, Texte, Date).
- [ ] **Tâche 2.2 : Stratégies de Traitement des Valeurs Manquantes**
  - Fournir des options interactives paramétrables par colonne ou de manière globale :
    - **Suppression :** Supprimer les lignes contenant des valeurs nulles, ou supprimer les colonnes ayant plus de $X\%$ de valeurs nulles (ex: seuil à 40%).
    - **Imputation Statistique (Numérique) :** Remplacement par la Moyenne (`mean`) ou la Médiane (`median`).
    - **Imputation Catégorielle :** Remplacement par le Mode (valeur la plus fréquente) ou une constante (ex: `"Inconnu"`).
    - **Imputation Avancée (Bonus) :** `KNNImputer` de Scikit-Learn pour une imputation par proximité.
- [ ] **Tâche 2.3 : Détection & Traitement des Valeurs Aberrantes (Outliers)**
  - Implémenter la méthode de l'Écart Interquartile (IQR) :
    $$\text{IQR} = Q_3 - Q_1 \quad \Longrightarrow \quad [\text{Borne Inf} = Q_1 - 1.5 \times \text{IQR}, \; \text{Borne Sup} = Q_3 + 1.5 \times \text{IQR}]$$
  - Proposer le choix : écrêtage (*clipping*) ou suppression des lignes aberrantes.

### Phase 3 : Encodage Catégoriel, Normalisation & Train/Test Split
- [ ] **Tâche 3.1 : Encodage des Variables Qualitatives**
  - **One-Hot Encoding (`get_dummies` ou `OneHotEncoder`) :** Idéal pour les variables nominales sans ordre (ex: `Ville`, `Sexe`). Gérer l'option `drop_first=True` pour éviter la multicolinéarité.
  - **Label / Ordinal Encoding :** Conversion des catégories ordonnées en entiers (ex: `Faible=0, Moyen=1, Élevé=2`).
  - Gérer proprement les catégories inconnues lors de l'inférence future (`handle_unknown='ignore'`).
- [ ] **Tâche 3.2 : Normalisation et Standardisation des Features**
  - Implémenter et expliquer dans l'interface le choix entre 3 scalers de référence :
    1. **StandardScaler (Z-score) :** Centrage-réduction ($z = \frac{x - \mu}{\sigma}$) — Indispensable pour SVM, KNN, Régression Logistique et Réseaux de Neurones.
    2. **MinMaxScaler :** Mise à l'échelle dans l'intervalle $[0, 1]$ ou $[a, b]$ — Adapté quand les distributions ne sont pas gaussiennes.
    3. **RobustScaler :** Basé sur la médiane et les quartiles — Résistant aux outliers résiduels.
- [ ] **Tâche 3.3 : Sélection de la Cible ($y$), des Features ($X$) et Découpage Train/Test**
  - Permettre à l'utilisateur de sélectionner la colonne cible ($y$) via un menu déroulant dynamique.
  - Identifier automatiquement le type de tâche :
    - Si $y$ est catégorielle ou possède peu de valeurs uniques distinctes $\le 10$ $\rightarrow$ **Classification**.
    - Si $y$ est numérique continue $\rightarrow$ **Régression**.
    - Si l'utilisateur choisit le mode "Sans cible" $\rightarrow$ **Clustering non supervisé (K-Means)**.
  - Permettre de sélectionner ou d'exclure manuellement certaines variables prédictives ($X$).
  - Réaliser le découpage :
    - Curseur de proportion : $70/30$, $80/20$, $75/25$.
    - Option de stratification : `stratify=y` pour conserver la proportion des classes en classification.
    - Fixation de la graine aléatoire (`random_state=42`) pour assurer la reproductibilité scientifique.

### Phase 4 : Sérialisation du Pipeline & Export des Données Propres
- [ ] **Tâche 4.1 : Création du Pipeline Scikit-Learn Persistant**
  - Assembler un `ColumnTransformer` et un `Pipeline` unifié englobant l'imputation, l'encodage et le scaler.
  - Sauvegarder ce pipeline sous format `.joblib` dans le dossier projet :
    `media/projects/<project_id>/preprocessing_pipeline.joblib`.
  - Ce fichier sera directement réutilisé par le **Membre 4** pour la prédiction en ligne.
- [ ] **Tâche 4.2 : Export du Jeu de Données Nettoyé**
  - Proposer un bouton de téléchargement "Exporter les données nettoyées (.csv)".

---

## 🛠️ Architecture du Code & Contrat d'Interface

### Classe Centrale de Prétraitement (`apps/preprocessing/pipeline.py`)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple, Dict, Any, List

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, target_col: str = None):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.target_col = target_col
        self.pipeline: Pipeline = None
        self.feature_names_out: List[str] = []

    def handle_missing_values(self, num_strategy: str = "mean", cat_strategy: str = "most_frequent") -> pd.DataFrame:
        """Nettoie les valeurs manquantes selon les stratégies choisies."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        if self.target_col in num_cols:
            num_cols.remove(self.target_col)
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)

        # Imputation numérique
        if num_cols:
            num_imputer = SimpleImputer(strategy=num_strategy)
            self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])

        # Imputation catégorielle
        if cat_cols:
            cat_imputer = SimpleImputer(strategy=cat_strategy)
            self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])

        return self.df

    def build_and_fit_transform(
        self,
        scaler_type: str = "standard",   # "standard" | "minmax" | "robust" | "none"
        encoding_type: str = "onehot",   # "onehot" | "ordinal"
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Construit le ColumnTransformer complet, découpe en Train/Test et transforme X.
        Retourne un dictionnaire standardisé prêt pour le Membre 4 et le Membre 3.
        """
        # Séparation X et y
        if self.target_col and self.target_col in self.df.columns:
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
        else:
            X = self.df.copy()
            y = None

        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Choix du Scaler
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "none": "passthrough"
        }
        selected_scaler = scalers.get(scaler_type, StandardScaler())

        # Choix de l'Encoder
        encoders = {
            "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        }
        selected_encoder = encoders.get(encoding_type, OneHotEncoder(handle_unknown="ignore", sparse_output=False))

        # Assemblage ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", selected_scaler, num_features),
                ("cat", selected_encoder, cat_features)
            ],
            remainder="drop"
        )

        # Train/Test Split
        if y is not None:
            is_stratified = y if (y.nunique() <= 10 and y.dtype == 'object') else None
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=is_stratified
            )
            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)
        else:
            X_train = preprocessor.fit_transform(X)
            X_test, y_train, y_test = None, None, None

        self.pipeline = preprocessor

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": num_features + cat_features,
            "raw_features": X.columns.tolist(),
            "pipeline": self.pipeline
        }

    def save_pipeline(self, filepath: str):
        """Sérialise le pipeline pour le service d'inférence."""
        joblib.dump(self.pipeline, filepath)
```

---

## 📅 Planning & Jalons Personnels

* **Fin Semaine 1 :** Upload de fichiers CSV/Excel/JSON fonctionnel avec gestion des encodages et détection des séparateurs. Datasets d'exemples intégrés.
* **Fin Semaine 2 :** Interface de détection et traitement des valeurs manquantes et doublons. Détection des outliers par IQR.
* **Fin Semaine 3 :** Intégration des scalers et encodeurs Scikit-Learn. Séparation Train/Test sans fuite de données et sérialisation du pipeline (`.joblib`).
* **Fin Semaine 4 :** Tests avec des datasets extrêmes (valeurs manquantes massives, formats variés) et finalisation de l'export CSV propre.

---

## 🏆 Critères de Réussite & Check-list d'Évaluation
- [ ] Aucun crash lors de l'import d'un fichier CSV avec encodage inhabituel ou délimiteur point-virgule.
- [ ] Les jeux d'exemples (Iris, Titanic, Housing) se chargent instantanément en un clic.
- [ ] Les valeurs manquantes sont correctement imputées et les catégories correctement encodées.
- [ ] Le pipeline Scikit-Learn est exporté proprement pour permettre au Membre 4 d'inférer de nouvelles données sans refaire le calcul.

