# 👤 Membre 1 : Architecte Logiciel, UI/UX SaaS & Centre de Documentation

> **Rôle :** Tech Lead & Frontend/UI Architect  
> **Périmètre :** Architecture Django globale, Design System (Style Notion / Linear / n8n), Navigation & Stepper, Gestionnaire de sessions/projets, Module 8 (Documentation & Tutoriels Pédagogiques).  
> **Stack Technique :** Django 5.x, Tailwind CSS, Alpine.js, HTMX, Markdown parser Python (`markdown` / `pymdown-extensions`).

---

## 🎯 Objectifs de la Mission

En tant que **Membre 1**, vous êtes le garant de la cohérence technique, visuelle et pédagogique de l'application. Vous construisez le squelette Django sur lequel vos trois coéquipiers vont brancher leurs modules. Vous créez une interface utilisateur digne des meilleurs produits SaaS actuels (Notion, Linear, n8n) avec un mode sombre soigné, et vous réalisez la documentation complète et les tutoriels interactifs imposés par le sujet.

---

## 📦 Modules & Applications Django Sous Votre Responsabilité

1. **`ml_studio/`** : Configuration globale Django (`settings.py`, `urls.py`, gestion des statiques/médias).
2. **`apps/core/`** : Dashboard principal, gestion des sessions/projets, breadcrumbs, toasts de notification.
3. **`apps/docs_tutorials/`** : Système de documentation intégrée, fiches théoriques des algorithmes, visite guidée interactive.
4. **`templates/` & `static/`** : Design system global, layout maître (`base.html`), composants réutilisables, styles Tailwind.

---

## 📋 Liste Détaillée des Tâches (Roadmap Membre 1)

### Phase 1 : Initialisation de l'Architecture & Environnement Django
- [ ] **Tâche 1.1 : Initialisation du Projet Django**
  - Créer l'environnement virtuel Python, installer Django 5.x et configurer la structure modulaire de répertoires (`apps/`).
  - Configurer `settings.py` pour séparer les répertoires d'applications, déclarer `STATICFILES_DIRS`, `MEDIA_ROOT`, et activer le support des sessions.
  - Définir le fichier `.gitignore` professionnel pour Python/Django (exclure `.venv`, `__pycache__`, fichiers temporaires SQLite/media).
- [ ] **Tâche 1.2 : Mise en Place du Design System Moderne**
  - Intégrer Tailwind CSS (via CDN optimisé ou CLI autonome) avec typographie soignée (police Google Inter ou Geist).
  - Configurer la palette de couleurs en Dark Mode (inspiré de **Linear** : `#0b0f19`, `#161b22`, accents Indigo `#6366f1`) et Light Mode (inspiré de **Notion**).
  - Créer `templates/base.html` qui servira de squelette pour tous les écrans (sidebar repliable, navbar avec indicateur de projet, conteneur principal responsive, zone de notifications toasts).
- [ ] **Tâche 1.3 : Intégration de HTMX et Alpine.js**
  - Charger HTMX et Alpine.js directement sans dépendance Node.js/npm.
  - Préparer les templates pour le rechargement partiel (swap dynamique) sans jamais rafraîchir la page entière.

### Phase 2 : Workflow Global & Expérience Utilisateur (Stepper n8n-style)
- [ ] **Tâche 2.1 : Composant Stepper Interactif**
  - Implémenter le composant de navigation par étapes en haut de chaque page :
    $$\text{1. Dataset} \longrightarrow \text{2. EDA} \longrightarrow \text{3. Prétraitement} \longrightarrow \text{4. Modélisation} \longrightarrow \text{5. Évaluation} \longrightarrow \text{6. Export \& Inférence}$$
  - Mettre en surbrillance l'étape active, griser les étapes futures tant que le dataset n'est pas encore prêt.
- [ ] **Tâche 2.2 : Gestion de l'État du Projet & Session Utilisateur**
  - Implémenter un gestionnaire de contexte (`context_processors.py`) qui injecte en permanence dans les templates :
    - Le dataset actuellement actif en session.
    - L'état d'avancement (Nettoyage fait ? Modèle entraîné ?).
    - Les messages d'alertes / flash notifications (succès, avertissement, erreur).
- [ ] **Tâche 2.3 : Bibliothèque de Composants Réutilisables**
  - Créer des fragments dans `templates/components/` :
    - `card.html` : Carte avec bordure discrète, ombrage subtil et titre stylisé.
    - `badge.html` : Badges colorés pour types de colonnes (`numérique`, `catégoriel`, `cible`).
    - `modal.html` : Fenêtre modale accessible gérée avec Alpine.js.
    - `toast.html` : Notification flottante animée en cas de succès ou d'erreur.
    - `loader.html` : Spinner d'attente animé lors des calculs lourds de Machine Learning.

### Phase 3 : Module 8 — Documentation & Tutoriels Intégrés (Exigence Sujet)
- [ ] **Tâche 3.1 : Centre Documentaire "ML Academy" (Style Notion Docs)**
  - Créer une section dédiée `/docs/` accessible depuis la sidebar.
  - Implémenter un moteur de lecture de fichiers Markdown en Python (`markdown` + `codehilite` + KaTeX pour les formules mathématiques).
  - Rédiger les fiches théoriques complètes pour les 11 algorithmes requis par le sujet :
    1. Régression Linéaire (Simple, Multiple, Polynomiale) : équations, hypothèses Gauss-Markov, OLS.
    2. Régression Logistique : fonction sigmoïde, log-odds, frontière de décision.
    3. Arbres de Décision : critère de Gini, entropie, élagage.
    4. Naïf Bayésien : théorème de Bayes, indépendance conditionnelle.
    5. Support Vector Machine (SVM) : marge maximale, noyaux (linéaire, RBF, poly).
    6. Random Forest : bagging, agrégation, vote majoritaire, out-of-bag.
    7. K-Plus Proches Voisins (KNN) : métriques de distance (euclidienne, manhattan), choix de $k$.
    8. K-Means : minimisation de l'inertie intra-cluster, méthode du coude (Elbow).
    9. Réseaux de Neurones Artificiels (MLP) : perceptron multicouche, rétropropagation du gradient, fonctions d'activation (ReLU, Sigmoid, Tanh).
- [ ] **Tâche 3.2 : Tutoriels Guidés et Démonstrations Pas-à-Pas**
  - Concevoir un parcours guidé pour 3 cas d'usage réels :
    - *Tutoriel 1 (Régression) :* Prédire le prix d'un logement (Dataset Housing) avec la Régression Linéaire et Polynomiale.
    - *Tutoriel 2 (Classification) :* Diagnostiquer une maladie ou prédire la survie (Dataset Titanic/Iris) avec Random Forest et SVM.
    - *Tutoriel 3 (Clustering) :* Segmenter des profils clients avec l'algorithme K-Means.
  - Créer un modal interactif d'onboarding lors de la première visite ("Guide de démarrage rapide").

### Phase 4 : Finitions, Tests d'Intégration & Packaging
- [ ] **Tâche 4.1 : Test de la Navigation de Bout en Bout**
  - Vérifier que les redirections entre les pages créées par les membres 2, 3 et 4 fonctionnent sans accroc.
- [ ] **Tâche 4.2 : Responsive Design & Finition Visuelle**
  - Vérifier l'affichage sur différentes résolutions d'écran (laptop, desktop grand écran).
  - Polir les micro-interactions (hover, transitions douces, focus ring sur les inputs).

---

## 🛠️ Spécifications Techniques & Contrats d'Interface

### Modèle de Données Session (Stocké dans `request.session`)
Pour que les modules des Membres 2, 3 et 4 communiquent sans heurts, vous définissez la clé `current_project` en session :

```python
# Exemple de structure standardisée dans request.session['project_state']
{
    "dataset_id": "uuid-or-int",
    "dataset_name": "california_housing.csv",
    "total_rows": 20640,
    "total_cols": 9,
    "target_column": "MedHouseVal",
    "task_type": "regression",  # "classification" | "regression" | "clustering"
    "preprocessing_done": True,
    "model_trained": True,
    "selected_model_name": "Random Forest Regressor",
    "best_score": 0.82
}
```

### Architecture du Layout Maître (`templates/base.html`)
```html
<!DOCTYPE html>
<html lang="fr" class="dark">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}ML Studio | Plateforme Polyvalente{% endblock %}</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- HTMX -->
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <!-- Alpine.js -->
    <script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    <!-- KaTeX pour les formules mathématiques dans la documentation -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 flex min-h-screen font-sans antialiased">
    <!-- Sidebar Navigation (Style Notion) -->
    {% include 'components/sidebar.html' %}

    <!-- Zone Principale de Contenu -->
    <div class="flex-1 flex flex-col min-w-0 overflow-hidden">
        <!-- Topbar avec Stepper n8n -->
        {% include 'components/topbar.html' %}

        <!-- Messages Flash & Toasts -->
        {% include 'components/toasts.html' %}

        <!-- Contenu Dynamique de la page -->
        <main class="flex-1 overflow-y-auto p-8">
            {% block content %}{% endblock %}
        </main>
    </div>
</body>
</html>
```

---

## 📅 Planning & Jalons Personnels

* **Fin Semaine 1 :** Socle Django prêt, `base.html` stylisé en Dark/Light mode, Sidebar fonctionnelle, structure des routes en place.
* **Fin Semaine 2 :** Stepper fonctionnel, intégration des pages préliminaires des membres 2 et 3, moteur de Markdown pour la documentation.
* **Fin Semaine 3 :** Fiches théoriques des 11 algorithmes complétées avec formules KaTeX, mise en page des tutoriels interactifs.
* **Fin Semaine 4 :** Intégration globale finale, responsive testing, guide d'installation dans le `README.md` principal.

---

## 🏆 Critères de Réussite & Check-list d'Évaluation
- [ ] L'application démarre immédiatement avec un simple `python manage.py runserver`.
- [ ] Le design a un rendu propre, aéré et professionnel (inspiré de Linear/Notion).
- [ ] La documentation explique rigoureusement les 11 algorithmes du sujet avec leurs formules.
- [ ] L'utilisateur comprend intuitivement où il se situe dans le workflow grâce au stepper.

