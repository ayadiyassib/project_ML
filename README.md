# 🚀 ML Studio — Plateforme Machine Learning Polyvalente

> **Projet de Master** &bull; Écosystème **100% Python** (Django 5.x, Scikit-Learn, Plotly, Tailwind, HTMX, Alpine.js & KaTeX sans Node.js).

---

## ⚡ Guide de Démarrage Rapide (Quick Start)

Pour lancer le projet et obtenir le même résultat en local :

### 1. Cloner et se placer dans le projet
```bash
cd projet_ML
```

### 2. Créer et activer l'environnement virtuel Python (3.11+)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Sur Linux / macOS
# ou : .venv\Scripts\activate sous Windows
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Appliquer les migrations de base de données
```bash
python manage.py migrate
```

### 5. (Optionnel) Créer le compte administrateur
```bash
python manage.py createsuperuser
# Ou utiliser le compte par défaut déjà configuré si présent : admin / 123456
```

### 6. Lancer le serveur local
```bash
python manage.py runserver
```

> **Note Windows :** Si le port 8000 est bloqué par le pare-feu, utilisez un port alternatif :
> ```powershell
> python manage.py runserver 127.0.0.1:8001
> ```
> Si la connexion est toujours refusée, autorisez `python.exe` dans **Windows Defender Firewall → Autoriser une application**.

---

## 🌐 Liens d'Accès

| Module | URL |
| --- | --- |
| **Dashboard principal** | [http://127.0.0.1:8000/](http://127.0.0.1:8000/) |
| **Ingestion de données** | [http://127.0.0.1:8000/datasets/](http://127.0.0.1:8000/datasets/) |
| **Analyse Exploratoire (EDA)** | [http://127.0.0.1:8000/eda/](http://127.0.0.1:8000/eda/) |
| **Prétraitement** | [http://127.0.0.1:8000/preprocessing/](http://127.0.0.1:8000/preprocessing/) |
| **Moteur ML — Entraînement & Évaluation** | [http://127.0.0.1:8000/ml_engine/](http://127.0.0.1:8000/ml_engine/) |
| **Visualisations Diagnostiques** | [http://127.0.0.1:8000/model_viz/](http://127.0.0.1:8000/model_viz/) |
| **Export & Inférence en direct** | [http://127.0.0.1:8000/export_inference/](http://127.0.0.1:8000/export_inference/) |
| **Centre Pédagogique & KaTeX** | [http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/) |
| **Panneau d'Administration Django** | [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/) *(admin / 123456)* |

---

## 📂 Architecture des Applications (`apps/`)

* `apps/core/` : Layout global, session du projet, dashboard principal.
* `apps/datasets/` : Ingestion de fichiers (CSV, XLSX, JSON) & jeux de données de référence.
* `apps/eda/` : Analyse exploratoire & graphiques Plotly interactifs.
* `apps/preprocessing/` : Nettoyage, imputations, encodages et normalisation Scikit-Learn.
* `apps/ml_engine/` : Moteur d'entraînement pour les 11 algorithmes requis.
* `apps/model_viz/` : Visualisations diagnostiques (ROC, Confusion Matrix, Résidus, PCA).
* `apps/export_inference/` : Inférence en direct, exports `.joblib` et rapports PDF.
* `apps/docs_tutorials/` : Documentation intégrée et fiches théoriques avec KaTeX.
* `sample_data/` : Datasets de test prêts à l'emploi (*Iris, Titanic, California Housing, Wine, Mall Customers*).

---

## 🔌 Contrat de données (modules EDA, ML & Export)

Les modules EDA et ML / Export ne doivent **jamais relire le fichier source ni refaire le prétraitement** : ils consomment les artefacts produits par `apps/datasets` et `apps/preprocessing`.

### 1. Dataset actif

```python
from apps.datasets.state import get_active_dataset

dataset = get_active_dataset(request)          # Dataset | None
frame   = apps.datasets.services.load_dataframe(dataset)  # DataFrame brut (copie)
```

La session ne contient que l'identifiant :

```python
request.session["project_state"] = {
    "dataset_id": "…", "dataset_name": "Titanic", "total_rows": 891, "total_cols": 12,
    "target_column": "Survived", "task_type": "classification",
    "preprocessing_done": True, "model_trained": False,
    "selected_model_name": None, "best_score": None, "active_step": 3,
}
```

### 2. Artefacts de prétraitement

Racine unique par projet : `media/projects/<dataset_id>/`

| Fichier | Contenu |
| --- | --- |
| `prepared.joblib` | Objet `PreparedData` complet (matrices + métadonnées) |
| `pipeline.joblib` | `ColumnTransformer` ajusté, à utiliser pour l'inférence |
| `config.json` | `PreprocessingConfig` sérialisée (traçabilité) |

```python
from apps.preprocessing.services import load_prepared, load_pipeline, load_config

prepared  = load_prepared(dataset)   # None si le prétraitement n'a pas encore tourné
pipeline  = load_pipeline(dataset)   # None si absent
config    = load_config(dataset)

X_train, X_test = prepared.X_train, prepared.X_test   # numpy (ndarray)
y_train, y_test = prepared.y_train, prepared.y_test   # pandas Series (None en clustering)
```

`PreparedData` expose aussi : `feature_names`, `raw_features`, `numeric_features`,
`categorical_features`, `task_type`, `target`, `class_labels`, `class_distribution`,
`warnings`, `dropped_columns`, `dropped_rows`, `df_clean` et `to_summary()`.

> ⚠️ **Pas de fuite de données** : imputation, encodage, normalisation et bornes IQR sont ajustés sur `X_train` uniquement, puis appliqués à `X_test`. Le découpage train/test a lieu **avant** tout calcul statistique. Toute réutilisation doit passer par `pipeline.transform(...)`, jamais par un `fit` sur les données de test.

### 3. Entraînement et inférence

```python
# Entraînement supervisé
model.fit(prepared.X_train, prepared.y_train)

# Inférence sur une ligne utilisateur (catégories inconnues gérées)
row = pandas.DataFrame([{"Pclass": 3, "Sex": "female", "Age": 28.0, "Fare": 9.5}])
X = pipeline.transform(row)   # même nombre de colonnes que prepared.n_features_out
```

### 4. Cycle de vie

* `POST /preprocessing/apply/` recalcule et écrase les artefacts, puis met `preprocessing_done=True` et réinitialise l'état du modèle (`model_trained=False`).
* `POST /preprocessing/reset/` supprime tout `media/projects/<dataset_id>/` (prétraitement **et** modèles entraînés).
* Supprimer un `Dataset` supprime automatiquement son fichier et son dossier d'artefacts (signaux `post_delete`).
## 🤖 Moteur Machine Learning (`apps/ml_engine/`)

Le moteur ML expose **11 algorithmes** couvrant l'ensemble des tâches supervisées et non supervisées, tous paramétrables depuis l'interface web sans écrire une ligne de code.

### Algorithmes disponibles

| Famille | Algorithme | Tâche |
| --- | --- | --- |
| **Régression** | Régression Linéaire Simple & Multiple | Régression |
| | Régression Polynomiale (degré 2–5) | Régression |
| | Régression Logistique (L1 / L2) | Classification |
| **Arbres** | Arbre de Décision (Classifier & Regressor) | Classification / Régression |
| | Forêt Aléatoire — Random Forest | Classification / Régression |
| **Géométrique / Probabiliste** | Support Vector Machine — SVM / SVR | Classification / Régression |
| | Naïf Bayésien — GaussianNB & MultinomialNB | Classification |
| | K Plus Proches Voisins — KNN | Classification / Régression |
| **Non supervisé** | K-Means Clustering | Clustering |
| **Réseaux de neurones** | Perceptron Multi-Couches — MLP | Classification / Régression |

### Validation & Métriques

* **Validation croisée** : `StratifiedKFold` (classification) et `KFold` (régression), K ∈ [3, 10].
* **Classification** : Accuracy, Précision, Rappel, F1-Score (macro / weighted), ROC-AUC.
* **Régression** : MAE, MSE, RMSE, R², R² ajusté.
* **Clustering** : Score de Silhouette, Indice de Davies-Bouldin, Inertie WCSS.
* **Leaderboard** : tableau comparatif multi-modèles avec badge « Best Model » automatique.

---

## 📦 Export & Inférence (`apps/export_inference/`)

Une fois un modèle entraîné, le module Export & Inférence permet de valoriser et déployer le résultat.

### Fonctionnalités

| Fonctionnalité | Description |
| --- | --- |
| **Export `.joblib`** | Téléchargement du pipeline complet (préprocesseur + modèle) en un clic. |
| **Export Prédictions CSV** | Fichier CSV contenant features, valeur réelle, valeur prédite et résidu / probabilité. |
| **Rapport PDF** | Rapport d'analyse automatisé (xhtml2pdf) : résumé dataset, prétraitements, hyperparamètres, métriques Train / Test / CV et recommandations. |
| **Inference Playground** | Formulaire web dynamique généré à partir des features du dataset. Retourne instantanément la prédiction et la probabilité de confiance associée. |

### Rechargement du modèle exporté

```python
import joblib, pandas as pd

pipeline = joblib.load("model_export.joblib")
new_data = pd.DataFrame([{"feature_1": 5.1, "feature_2": "value", ...}])
predictions = pipeline.predict(new_data)
```

### URLs du module

| Action | URL |
| --- | --- |
| Page principale Export & Inférence | `GET  /export_inference/` |
| Télécharger le modèle `.joblib` | `GET  /export_inference/download/model/` |
| Télécharger les prédictions `.csv` | `GET  /export_inference/download/predictions/` |
| Télécharger le rapport PDF | `GET  /export_inference/download/pdf/` |
| Prédiction en direct (HTMX / JSON) | `POST /export_inference/predict/` |

---
