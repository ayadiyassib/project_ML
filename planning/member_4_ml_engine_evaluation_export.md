# 👤 Membre 4 : Ingénieur Machine Learning, Évaluation & Inférence

> **Rôle :** Machine Learning Engineer & MLOps Specialist  
> **Périmètre :** Implémentation des 11 Algorithmes de Machine Learning, Moteur de Validation Croisée (K-Fold), Métriques de performance, Leaderboard comparatif, Sérialisation & Export des modèles (`.joblib`), Génération de Rapports d'analyse (PDF) et Bac à sable d'Inférence en direct (Fonctionnalités 4, 5 et 7 du sujet).  
> **Stack Technique :** Python 3.10+, Scikit-Learn (`sklearn.linear_model`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.svm`, `sklearn.neighbors`, `sklearn.naive_bayes`, `sklearn.cluster`, `sklearn.neural_network`, `sklearn.metrics`, `sklearn.model_selection`), Joblib, NumPy, Pandas, xhtml2pdf / WeasyPrint (Génération PDF 100% Python).

---

## 🎯 Objectifs de la Mission

En tant que **Membre 4**, vous êtes le cœur algorithmique du projet. C'est vous qui donnez à l'application sa puissance prédictive et son intelligence mathématique.

Vous avez la charge exclusive de :
1. **L'implémentation rigoureuse des 11 algorithmes demandés dans le sujet**, avec un contrôle total des hyperparamètres depuis l'interface utilisateur.
2. **La validation scientifique irréprochable des modèles** : métriques exhaustives et validation croisée (*K-Fold*).
3. **Le comparateur de modèles (Leaderboard)** : identifier automatiquement le meilleur algorithme pour le problème donné.
4. **L'exportation et le déploiement** : permettre à l'utilisateur d'emporter son modèle entraîné (`.joblib`), de générer un rapport PDF de synthèse, et de tester son modèle en direct sur de nouvelles données via un formulaire web interactif.

---

## 📦 Modules & Applications Django Sous Votre Responsabilité

1. **`apps/ml_engine/`** : Implémentation des 11 algorithmes, pipeline d'entraînement, réglage d'hyperparamètres et validation croisée.
2. **`apps/export_inference/`** : Moteur d'exportation des artefacts (`.joblib`), générateur de rapport PDF automatisé et simulateur de prédiction en temps réel (*Inference Playground*).

---

## 📋 Liste Détaillée des Tâches (Roadmap Membre 4)

### Phase 1 : Implémentation Exhaustive des 11 Algorithmes (Module 4)
Chaque algorithme doit être encapsulé dans une classe standardisée avec gestion dynamique de ses hyperparamètres :

#### 1. Famille des Régressions
- [ ] **Algorithme 1 : Régression Linéaire Simple**
  - Utilise 1 variable explicative continue $x$ pour prédire $y$ : $\hat{y} = \beta_0 + \beta_1 x$.
  - Afficher les coefficients (ordonnée à l'origine $\beta_0$, pente $\beta_1$).
- [ ] **Algorithme 2 : Régression Linéaire Multiple**
  - Utilise $p$ variables explicatives : $\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j$.
  - Calculer les coefficients pour chaque variable prédictive.
- [ ] **Algorithme 3 : Régression Linéaire Polynomiale**
  - Pipeline combinant `PolynomialFeatures(degree=d)` et `LinearRegression()`.
  - Hyperparamètre ajustable dans l'UI : degré du polynôme $d \in [2, 5]$.
- [ ] **Algorithme 4 : Régression Logistique**
  - Modélisation de probabilité via la fonction sigmoïde : $P(y=1|X) = \frac{1}{1 + e^{-z}}$.
  - Support de la classification binaire et multi-classe (`ovr` / `multinomial`).
  - Hyperparamètres ajustables : régularisation $C \in [0.01, 100]$, type de pénalité (`l1`, `l2`).

#### 2. Arbres & Forêts Aléatoires
- [ ] **Algorithme 5 : Arbres de Décision (Decision Trees)**
  - Support de `DecisionTreeClassifier` et `DecisionTreeRegressor`.
  - Hyperparamètres ajustables : critère (`gini` ou `entropy` pour classification, `squared_error` pour régression), profondeur maximale `max_depth`, nombre minimal d'échantillons par feuille `min_samples_leaf`.
- [ ] **Algorithme 6 : Forêts Aléatoires (Random Forest)**
  - Support de `RandomForestClassifier` et `RandomForestRegressor`.
  - Hyperparamètres ajustables : nombre d'arbres `n_estimators` (ex: 50, 100, 200), `max_depth`, `max_features` (`sqrt`, `log2`). Extraction des importances de features (*Gini Importance*).

#### 3. Modèles Géométriques, Probabilistes & Voisinage
- [ ] **Algorithme 7 : Support Vector Machine (SVM / SVR)**
  - Support de `SVC` (classification avec calcul de probabilités `probability=True`) et `SVR` (régression).
  - Hyperparamètres ajustables : type de noyau `kernel` (`linear`, `rbf`, `poly`), paramètre de marge $C$, coefficient $\gamma$ (`scale`, `auto`), degré pour le noyau polynomial.
- [ ] **Algorithme 8 : Naïf Bayésien (Naive Bayes)**
  - Implémenter `GaussianNB` (données continues) et `MultinomialNB` (données discrètes / comptage).
  - Calcul des probabilités a posteriori basé sur le théorème de Bayes : $P(y|X) = \frac{P(X|y)P(y)}{P(X)}$.
- [ ] **Algorithme 9 : Plus Proche Voisin (KNN - K-Nearest Neighbors)**
  - Support de `KNeighborsClassifier` et `KNeighborsRegressor`.
  - Hyperparamètres ajustables : nombre de voisins $k \in [1, 25]$, pondération des poids `weights` (`uniform`, `distance`), métrique de distance (`euclidean`, `manhattan`).

#### 4. Clustering & Réseaux de Neurones
- [ ] **Algorithme 10 : K-Means (Clustering Non Supervisé)**
  - Partitionnement des données en $k$ grappes par minimisation de l'inertie intra-classe.
  - Hyperparamètres ajustables : nombre de clusters $k \in [2, 10]$, méthode d'initialisation (`k-means++`, `random`), nombre d'itérations max.
  - Calcul de l'inertie ($WCSS$) et des coordonnées des centroïdes finaux.
- [ ] **Algorithme 11 : Réseaux de Neurones Artificiels (MLP - Multi-Layer Perceptron)**
  - Support de `MLPClassifier` et `MLPRegressor` (Scikit-Learn).
  - Hyperparamètres configurables via l'interface :
    - Architecture des couches cachées (ex: `(64, 32)`, `(100,)`, `(32, 16, 8)`).
    - Fonction d'activation (`relu`, `tanh`, `logistic`).
    - Algorithme d'optimisation (`adam`, `sgd`).
    - Taux de régularisation L2 `alpha` et nombre maximal d'époques `max_iter`.
  - Extraction de l'historique de la fonction de perte (`model.loss_curve_`).

---

### Phase 2 : Moteur de Validation Croisée & Calcul des Métriques (Module 5)
- [ ] **Tâche 2.1 : Validation Croisée Robuste (K-Fold)**
  - Implémenter `KFold` (pour la régression) et `StratifiedKFold` (pour la classification avec classes équilibrées).
  - Paramètre de plis $K \in [3, 10]$ (par défaut $K=5$).
  - Calcul de la moyenne et de l'écart-type des scores pour mesurer la stabilité du modèle.
- [ ] **Tâche 2.2 : Calcul Exhaustif des Métriques d'Évaluation**
  - **Tâche de Régression :**
    - Erreur Absolue Moyenne : $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$
    - Erreur Quadratique Moyenne : $\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$
    - Racine de l'Erreur Quadratique Moyenne : $\text{RMSE} = \sqrt{\text{MSE}}$
    - Coefficient de détermination : $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
    - $R^2$ Ajusté tenant compte du nombre de features $p$.
  - **Tâche de Classification :**
    - Exactitude (*Accuracy*), Précision pondérée, Rappel pondéré (*Recall*), F1-Score macro et micro.
    - Aire sous la courbe ROC (*ROC-AUC Score*).
    - Rapport de classification détaillé par classe (`classification_report` sous forme de dictionnaire Python).
  - **Tâche de Clustering (Non supervisé) :**
    - Score de Silhouette moyen ($s \in [-1, 1]$).
    - Indice de Davies-Bouldin (plus il est faible, meilleurs sont les clusters).
    - Inertie totale intra-cluster ($WCSS$).
- [ ] **Tâche 2.3 : Tableau Comparatif (Leaderboard)**
  - Permettre d'entraîner plusieurs algorithmes successivement sur le même dataset.
  - Afficher un tableau comparatif mettant en valeur le modèle champion (ex: couronne ou badge "Best Model" selon le F1-score ou le $R^2$).

---

### Phase 3 : Exportation & Déploiement en Production (Module 7)
- [ ] **Tâche 3.1 : Export du Modèle Sérialisé (.joblib / .pkl)**
  - Assembler le modèle entraîné avec le préprocesseur du Membre 2 dans un objet Scikit-Learn `Pipeline` autonome :
    $$\text{Final Pipeline} = \text{ColumnTransformer} + \text{Model}$$
  - Permettre le téléchargement direct du fichier `.joblib` en un clic.
  - Fournir un extrait de code Python prêt à copier/coller montrant comment recharger le modèle dans n'importe quel script externe :
    ```python
    import joblib
    model = joblib.load("model_export.joblib")
    predictions = model.predict(new_data)
    ```
- [ ] **Tâche 3.2 : Export des Prédictions en CSV**
  - Permettre d'exporter les résultats du test set (Features + Valeur Réelle + Valeur Prédite + Résidu / Probabilité) dans un fichier `predictions.csv`.
- [ ] **Tâche 3.3 : Générateur Automatisé de Rapport d'Analyse (PDF / HTML)**
  - Développer un générateur de rapport en Python pur (`xhtml2pdf` ou `weasyprint`).
  - Le rapport PDF téléchargeable doit contenir :
    1. Résumé du Dataset (nombre de lignes, colonnes, cible).
    2. Liste des prétraitements appliqués (imputation, encodage, scaler).
    3. Tableau récapitulatif des hyperparamètres du modèle champion.
    4. Métriques complètes de performance (Train vs Test vs Cross-Validation).
    5. Conclusion et recommandations d'usage.
- [ ] **Tâche 3.4 : Bac à Sable d'Inférence en Direct (Inference Playground)**
  - Générer dynamiquement dans l'interface web un formulaire avec un champ pour chaque feature originale du dataset (champ numérique avec min/max, champ déroulant pour les variables catégorielles).
  - L'utilisateur entre des valeurs personnalisées et clique sur "Prédire".
  - Le backend transforme les entrées avec le pipeline et retourne instantanément :
    - La prédiction finale (ex: *Prix estimé : 325 000 €* ou *Classe : Iris-Versicolor*).
    - La probabilité associée (ex: *Confiance : 94.2%*).

---

## 🛠️ Architecture du Code & Contrat d'Interface

### Registre des Modèles & Orchestrateur (`apps/ml_engine/trainer.py`)

```python
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
import joblib

class ModelRegistry:
    @staticmethod
    def get_model_instance(model_key: str, params: Dict[str, Any], task_type: str):
        """Instancie l'algorithme choisi parmi les 11 avec les hyperparamètres reçus de l'UI."""
        if model_key == "linear_regression":
            return LinearRegression()
        
        elif model_key == "polynomial_regression":
            degree = int(params.get("degree", 2))
            return Pipeline([
                ("poly", PolynomialFeatures(degree=degree)),
                ("linear", LinearRegression())
            ])
            
        elif model_key == "logistic_regression":
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                penalty=params.get("penalty", "l2"),
                max_iter=int(params.get("max_iter", 500))
            )
            
        elif model_key == "decision_tree":
            criterion = params.get("criterion", "gini" if task_type == "classification" else "squared_error")
            max_depth = int(params["max_depth"]) if params.get("max_depth") else None
            return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth) if task_type == "classification" else DecisionTreeRegressor(max_depth=max_depth)
            
        elif model_key == "random_forest":
            n_estimators = int(params.get("n_estimators", 100))
            max_depth = int(params["max_depth"]) if params.get("max_depth") else None
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth) if task_type == "classification" else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            
        elif model_key == "svm":
            kernel = params.get("kernel", "rbf")
            C = float(params.get("C", 1.0))
            return SVC(kernel=kernel, C=C, probability=True) if task_type == "classification" else SVR(kernel=kernel, C=C)
            
        elif model_key == "naive_bayes":
            return GaussianNB()
            
        elif model_key == "knn":
            n_neighbors = int(params.get("n_neighbors", 5))
            weights = params.get("weights", "uniform")
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights) if task_type == "classification" else KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
            
        elif model_key == "kmeans":
            n_clusters = int(params.get("n_clusters", 3))
            return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            
        elif model_key == "neural_network":
            hidden_layers = tuple(map(int, params.get("hidden_layers", "64,32").split(",")))
            activation = params.get("activation", "relu")
            max_iter = int(params.get("max_iter", 300))
            return MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=max_iter, random_state=42) if task_type == "classification" else MLPRegressor(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=max_iter, random_state=42)
            
        raise ValueError(f"Algorithme inconnu : {model_key}")

class ModelTrainer:
    def __init__(self, model, task_type: str):
        self.model = model
        self.task_type = task_type

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, cv_folds: int = 5) -> Dict[str, Any]:
        """Entraîne le modèle, applique la cross-validation et calcule toutes les métriques."""
        # Cas Clustering non-supervisé (K-Means)
        if self.task_type == "clustering":
            self.model.fit(X_train)
            labels = self.model.labels_
            sil_score = silhouette_score(X_train, labels) if len(np.unique(labels)) > 1 else 0.0
            return {
                "inertia": float(self.model.inertia_),
                "silhouette_score": float(sil_score),
                "n_clusters": self.model.n_clusters,
                "labels": labels.tolist()
            }

        # Entraînement supervisé
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Validation croisée
        cv = StratifiedKFold(n_splits=cv_folds) if self.task_type == "classification" else KFold(n_splits=cv_folds)
        scoring = "accuracy" if self.task_type == "classification" else "r2"
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)

        results = {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "y_pred": y_pred.tolist(),
        }

        if self.task_type == "classification":
            results["accuracy"] = float(accuracy_score(y_test, y_pred))
            results["precision"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            results["recall"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            results["f1"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            if hasattr(self.model, "predict_proba") and len(np.unique(y_test)) == 2:
                results["roc_auc"] = float(roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1]))
        else:
            results["mae"] = float(mean_absolute_error(y_test, y_pred))
            results["mse"] = float(mean_squared_error(y_test, y_pred))
            results["rmse"] = float(np.sqrt(results["mse"]))
            results["r2"] = float(r2_score(y_test, y_pred))

        return results
```

---

## 📅 Planning & Jalons Personnels

* **Fin Semaine 1 :** Architecture de `ModelRegistry` et implémentation des régressions (simple, multiple, polynomiale) et de la régression logistique.
* **Fin Semaine 2 :** Implémentation des arbres de décision, forêts aléatoires, SVM, Naïf Bayes et KNN avec leurs hyperparamètres respectifs.
* **Fin Semaine 3 :** Implémentation de K-Means et du réseau de neurones (MLP). Moteur de validation croisée et calcul de toutes les métriques.
* **Fin Semaine 4 :** Sérialisation `.joblib`, génération de rapport PDF de synthèse, et formulaire d'inférence en direct.

---

## 🏆 Critères de Réussite & Check-list d'Évaluation
- [ ] Les 11 algorithmes du sujet sont tous fonctionnels et paramétrables depuis l'interface web.
- [ ] La validation croisée $K$-Fold donne des résultats robustes sans divergence.
- [ ] L'export `.joblib` se recharge avec succès dans un script Python indépendant.
- [ ] Le playground de prédiction en direct retourne une prédiction instantanée dès la saisie de nouvelles valeurs.
- [ ] Le rapport PDF synthétise fidèlement l'ensemble de la modélisation.

