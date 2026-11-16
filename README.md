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

---

## 🌐 Liens d'Accès

* **Studio ML (Application principale) :**  
  👉 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

* **Centre Pédagogique & Fiches Mathématiques (Module 8) :**  
  👉 [http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/)

* **Panneau d'Administration Django :**  
  👉 [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/)  
  *(Identifiants : `admin` / `123456`)*

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

Les modules M3 (EDA) et M4 (ML / export) ne doivent **jamais relire le fichier source ni refaire le prétraitement** : ils consomment les artefacts produits par `apps/datasets` et `apps/preprocessing`.

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
# M4 : entraînement
model.fit(prepared.X_train, prepared.y_train)

# M4 : inférence sur une ligne utilisateur (catégories inconnues gérées)
row = pandas.DataFrame([{"Pclass": 3, "Sex": "female", "Age": 28.0, "Fare": 9.5}])
X = pipeline.transform(row)   # même nombre de colonnes que prepared.n_features_out
```

### 4. Cycle de vie

* `POST /preprocessing/apply/` recalcule et écrase les artefacts, puis met `preprocessing_done=True` et réinitialise l'état du modèle (`model_trained=False`).
* `POST /preprocessing/reset/` supprime tout `media/projects/<dataset_id>/` (prétraitement **et** modèles entraînés).
* Supprimer un `Dataset` supprime automatiquement son fichier et son dossier d'artefacts (signaux `post_delete`).
