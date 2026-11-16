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
* `sample_data/` : Datasets de test prêts à l'emploi (*Iris, Titanic, California Housing, Wine*).
