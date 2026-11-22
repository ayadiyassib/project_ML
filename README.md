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

---

## 📊 Exploration (EDA) & Diagnostics des modèles (`apps/eda/`, `apps/model_viz/`)

Tous les graphiques sont construits en Python avec Plotly, sérialisés en JSON (`plotly.io.to_json`) puis rendus côté navigateur par `Plotly.newPlot` (`static/js/ml_charts.js`) : aucune image générée côté serveur, zoom / survol / masquage de séries natifs.

### Étape 2 — Analyse exploratoire (`/eda/`)

| Onglet | Contenu |
| --- | --- |
| **Vue d'ensemble** | Cartes métriques, constats automatiques (valeurs manquantes, asymétrie, outliers, colinéarité, déséquilibre des classes, doublons), équilibre de la cible, valeurs manquantes par colonne, tableau `describe()` complet (moyenne, médiane, écart-type, variance, min/max, Q1/Q3, IQR, skewness, kurtosis, uniques, cardinalité, outliers) et tableau des variables catégorielles. |
| **Distributions** | Numérique : histogramme + KDE, boîte à moustaches, violon, violon par classe de la cible, test de normalité de D'Agostino. Catégorielle : fréquences absolues / relatives et donut. |
| **Corrélations** | Heatmap Pearson ou Spearman, échelle divergente centrée sur 0, triangle supérieur masquable, paires les plus liées. |
| **Nuages de points** | Scatter 2D (X, Y, couleur, droite MCO), vue 3D rotative, matrice de dispersion (pair plot). |

L'EDA travaille sur le jeu brut (`load_dataframe`) : rien n'est réinjecté dans un modèle, donc aucune fuite de données. Les identifiants (`PassengerId`, `CustomerID`…) sont exclus des corrélations et des nuages de points.

### Étape 5 — Diagnostics (`/diagnostics/`)

| Tâche | Graphiques |
| --- | --- |
| **Régression** | Courbe d'ajustement avec IC 95 % (régression simple / polynomiale à 1 variable), prédit vs réel avec y = x, résidus vs prédits (±1,96σ), histogramme des résidus + loi normale, Q-Q plot. |
| **Classification** | Matrice de confusion (effectifs + % par classe réelle), courbes ROC et Précision-Rappel (binaire ou un-contre-tous + micro-moyenne), précision / rappel / F1 par classe, frontière de décision sur projection PCA 2D. |
| **Clustering** | Méthode du coude (WCSS) + silhouette moyenne pour k = 2…10 (disponible avant même l'entraînement), silhouette par observation, clusters et centroïdes sur PCA 2D. |
| **Interprétabilité** | Importance de Gini (arbres, forêts), coefficients (modèles linéaires), importance par permutation (SVM, KNN, Naïf Bayes, MLP), courbe de perte du MLP. |

### Contrat avec le moteur ML

Le module lit le dernier modèle enregistré par `apps.ml_engine.services` :

```text
media/projects/<dataset_id>/models/latest_model.joblib   # estimateur scikit-learn ajusté
media/projects/<dataset_id>/models/latest_result.json    # TrainingResult.to_dict()
```

* Les prédictions sont recalculées sur `prepared.X_test` avec le modèle sauvegardé : les graphiques décrivent exactement le modèle entraîné.
* Un modèle plus ancien que `prepared.joblib` (prétraitement relancé), ou dont le nombre de variables ne correspond plus, est signalé et ignoré.
* Les seuls ajustements faits ici servent la visualisation : K-Means pour k = 2…10 (coude) et un clone du modèle sur 2 composantes PCA (frontière de décision, signalé comme approximation).

Réutiliser un graphique ailleurs (rapport, autre page) :

```python
from apps.eda.theme import make_chart
from apps.model_viz.diagnostic_plots import ModelDiagnosticPlots

chart = make_chart(ModelDiagnosticPlots.confusion_matrix(cm, labels), "cm", "Matrice de confusion")
# template : {% include 'components/plotly_chart.html' with chart=chart %}
#            + <script src="{% static 'js/ml_charts.js' %}"></script>
```

### Conventions visuelles

* Thème sombre « Omakase » commun (`apps/eda/theme.py`) : fond transparent, encre papier de riz, grille discrète, police Noto Sans JP.
* Couleurs catégorielles dans un ordre fixe, validé pour les daltonismes sur la surface `#1b1918`. La classe *i* garde toujours la même couleur et le même symbole de marqueur.
* Échelle séquentielle bleue pour les magnitudes, divergente bleu ↔ rouge pour les corrélations, aucun graphique à double axe Y.
* Les nuages de points sont échantillonnés (3 000 points en 2D, 2 000 en 3D, 1 200 pour le pair plot) ; les statistiques, elles, utilisent toujours toutes les lignes.

### Tests

```bash
python manage.py test apps.eda apps.model_viz
```
