# 👤 Membre 3 : Spécialiste EDA & Visualisation Interactive des Données

> **Rôle :** Data Visualization Engineer & EDA Specialist  
> **Périmètre :** Représentation des données brutes (EDA), Moteur de statistiques descriptives, Graphiques exploratoires interactifs (Plotly), et Visualisation diagnostique avancée des résultats de Machine Learning (Fonctionnalités 2 et 6 du sujet).  
> **Stack Technique :** Python 3.10+, Plotly (`plotly.express` & `plotly.graph_objects`), Pandas, NumPy, SciPy (statistiques avancées), Scikit-Learn (`PCA` pour projections 2D/3D).

---

## 🎯 Objectifs de la Mission

En tant que **Membre 3**, vous êtes l'artisan de la compréhension visuelle et du *Data Storytelling*. Vous transformez des chiffres abstraits et des matrices complexes en graphiques interactifs spectaculaires et intelligibles.

Votre travail s'articule autour des deux fonctionnalités visuelles fondamentales du sujet :
1. **Fonctionnalité 2 : Représentation des Données (EDA)** — Révéler les corrélations, distributions et déséquilibres d'un dataset avant l'entraînement.
2. **Fonctionnalité 6 : Visualisation des Résultats** — Fournir des graphiques diagnostiques de niveau professionnel pour chaque famille d'algorithme (ROC-AUC, matrices de confusion, résidus de régression, frontières de décision, clusters K-Means en 2D, courbes de convergence des réseaux de neurones).

---

## 📦 Modules & Applications Django Sous Votre Responsabilité

1. **`apps/eda/`** : Moteur de calculs statistiques descriptifs et galerie de visualisations univariées/bivariées.
2. **`apps/model_viz/`** : Moteur de graphiques diagnostiques pour les modèles entraînés par le Membre 4.
3. **`templates/eda/`** & **`templates/model_viz/`** : Gabarits d'intégration des conteneurs Plotly interactifs.

---

## 📋 Liste Détaillée des Tâches (Roadmap Membre 3)

### Phase 1 : Moteur de Statistiques Descriptives & Profiling (EDA)
- [ ] **Tâche 1.1 : Moteur de Métriques Statistiques Unifiées**
  - Concevoir un service Python calculant pour chaque colonne :
    - Moyenne, Médiane, Écart-type, Variance, Minimum, Maximum.
    - Quartiles ($Q_1$, $Q_3$) et Écart Interquartile (IQR).
    - Asymétrie (*Skewness*) et Aplatissement (*Kurtosis*) pour détecter la normalité.
    - Nombre de valeurs uniques et ratio de cardinalité.
  - Générer un tableau récapitulatif stylisé dans l'UI (cartes métriques + badges de complétude).
- [ ] **Tâche 1.2 : Visualisations Univariées Interactives (Plotly)**
  - **Pour les variables numériques :**
    - Histogramme interactif avec estimation de la densité par noyau (KDE).
    - Boxplot (boîte à moustaches) complet affichant les quartiles, la médiane et les points aberrants.
    - Graphique Violon (*Violin plot*) pour apprécier la forme multimodale des données.
  - **Pour les variables catégorielles :**
    - Diagrammes en barres des fréquences absolues et relatives (pourcentages).
    - Donut chart épuré avec survol interactif.

### Phase 2 : Visualisations Bivariées & Analyse Multivariée
- [ ] **Tâche 2.1 : Matrice de Corrélation Interactive (Heatmap)**
  - Calculer les coefficients de corrélation de **Pearson** (relations linéaires) et **Spearman** (relations monotones/non-linéaires).
  - Générer une Heatmap Plotly haute fidélité avec échelle de couleurs divergente (ex: Coolwarm, RdBu), annotations des valeurs, et masquage optionnel du triangle supérieur pour plus de clarté.
- [ ] **Tâche 2.2 : Nuages de Points 2D/3D & Matrice de Dispersion (*Pairplot*)**
  - Scatter plot interactif permettant à l'utilisateur de choisir l'axe X, l'axe Y et une variable de couleur (cible $y$).
  - Vue 3D rotative (`plotly.graph_objects.Scatter3d`) pour les jeux de données à plusieurs dimensions clés.
  - Matrice de dispersion interactive filtrable.

### Phase 3 : Graphiques Diagnostiques pour Modèles de Machine Learning (Module 6)
En collaboration avec le **Membre 4**, vous générez les graphiques d'évaluation adaptés à chaque type de tâche :

#### A. Diagnostics de Régression (Simple, Multiple, Polynomiale)
- [ ] **Tâche 3.1 : Courbes d'Ajustement & Droite de Régression**
  - *Régression Simple & Polynomiale :* Tracer les points réels ($X, y$) et superposer la droite ou la courbe polynomiale ajustée ($\hat{y}$) avec intervalle de confiance.
  - *Régression Multiple :* Tracer le graphique Valeurs Prédites vs Valeurs Réelles avec la ligne idéale $y = x$.
- [ ] **Tâche 3.2 : Analyse des Résidus & Normalité des Erreurs**
  - Graphique des Résidus ($e_i = y_i - \hat{y}_i$) en fonction des valeurs prédites (vérification de l'homoscédasticité).
  - Histogramme de distribution des résidus pour vérifier l'hypothèse de normalité des erreurs.

#### B. Diagnostics de Classification (Logistique, SVM, Arbre, Forêt, Naïf Bayes, KNN)
- [ ] **Tâche 3.3 : Matrice de Confusion Interactive**
  - Heatmap avec compteurs bruts et pourcentages normalisés (Vrais Positifs, Faux Positifs, Vrais Négatifs, Faux Négatifs).
  - Support multi-classes (ex: 3 classes pour Iris).
- [ ] **Tâche 3.4 : Courbes ROC & Précision-Rappel (PR)**
  - Tracé de la courbe ROC (Taux de Vrais Positifs vs Taux de Faux Positifs) avec zone grisée et score AUC mis en évidence.
  - Courbe Précision-Rappel pour les jeux de données déséquilibrés.
- [ ] **Tâche 3.5 : Frontières de Décision en 2D (Projection PCA)**
  - Réduire l'espace des features à 2 composantes principales (PCA) et afficher la frontière de décision graphique montrant la séparation des classes par l'algorithme sélectionné (SVM, Arbre, KNN, etc.).

#### C. Diagnostics de Clustering Non Supervisé (K-Means)
- [ ] **Tâche 3.6 : Méthode du Coude (*Elbow Curve*) & Score de Silhouette**
  - Graphique de l'Inertie Intra-Cluster (WCSS) en fonction du nombre de clusters $k \in [2, 10]$ pour identifier le coude optimal.
  - Graphique des coefficients de silhouette individuels par cluster.
- [ ] **Tâche 3.7 : Visualisation 2D des Clusters & Centroïdes**
  - Projection des clusters formés sur les deux premiers axes PCA avec mise en évidence des centroïdes finaux.

#### D. Diagnostics Arbres, Forêts & Réseaux de Neurones
- [ ] **Tâche 3.8 : Importance des Variables (*Feature Importance*)**
  - Diagramme en barres horizontales ordonné des variables les plus contributrices (Random Forest, Decision Tree, et coefficients en valeur absolue pour la Régression Logistique).
- [ ] **Tâche 3.9 : Courbe de Convergence du Réseau de Neurones (MLP)**
  - Graphique de la décroissance de la fonction de perte (*Loss*) et de l'évolution du score de validation au fil des itérations d'entraînement.

### Phase 4 : Optimisation du Rendu Graphique & Dark Mode Styling
- [ ] **Tâche 4.1 : Harmonisation Visuelle avec le Thème Notion/Linear**
  - Personnaliser les templates Plotly (fond transparent `#0b0f19`, grille discrète `#1e293b`, police Inter, infobulles stylisées en Dark Mode).
- [ ] **Tâche 4.2 : Sérialisation JSON Ultra-Rapide**
  - Utiliser `plotly.io.to_json(fig)` côté Django et `Plotly.newPlot()` côté template pour éviter tout fichier image lourd et garantir une fluidité totale à 60 FPS.

---

## 🛠️ Architecture du Code & Contrat d'Interface

### Fabrique de Graphiques Plotly (`apps/eda/plot_factory.py`)

```python
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from typing import Dict, Any

# Thème sombre moderne inspiré de Linear / Notion
DARK_THEME_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.6)",
    font=dict(color="#f1f5f9", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
    margin=dict(l=40, r=40, t=50, b=40)
)

class EDAPlotFactory:
    @staticmethod
    def create_distribution_plot(df: pd.DataFrame, column: str) -> str:
        """Génère un histogramme avec KDE au format JSON Plotly."""
        fig = px.histogram(
            df, x=column, marginal="box", nbins=30,
            title=f"Distribution de la variable : {column}",
            color_discrete_sequence=["#6366f1"]
        )
        fig.update_layout(**DARK_THEME_LAYOUT)
        return pio.to_json(fig)

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> str:
        """Génère la matrice de corrélation interactive."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr(method=method).round(2)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="Viridis",
            zmin=-1, zmax=1,
            text=corr_matrix.values,
            texttemplate="%{text}",
            hoverongaps=False
        ))
        fig.update_layout(title=f"Matrice de Corrélation ({method.capitalize()})", **DARK_THEME_LAYOUT)
        return pio.to_json(fig)
```

### Fabrique de Graphiques Diagnostiques ML (`apps/model_viz/diagnostic_plots.py`)

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

class ModelDiagnosticPlots:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None) -> str:
        """Génère la matrice de confusion interactive."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"Prédit {l}" for l in (labels or range(len(cm)))],
            y=[f"Réel {l}" for l in (labels or range(len(cm)))],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hoverongaps=False
        ))
        fig.update_layout(title="Matrice de Confusion", **DARK_THEME_LAYOUT)
        return pio.to_json(fig)

    @staticmethod
    def plot_roc_curve(y_true, y_score) -> str:
        """Génère la courbe ROC avec calcul d'AUC."""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})', line=dict(color='#8b5cf6', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(dash='dash', color='#94a3b8')))
        fig.update_layout(title="Courbe ROC", xaxis_title="Taux Faux Positifs", yaxis_title="Taux Vrais Positifs", **DARK_THEME_LAYOUT)
        return pio.to_json(fig)

    @staticmethod
    def plot_residuals(y_true, y_pred) -> str:
        """Génère le graphique des résidus pour la régression."""
        residuals = np.array(y_true) - np.array(y_pred)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(color='#38bdf8', opacity=0.7)))
        fig.add_hline(y=0, line_dash="dash", line_color="#f43f5e")
        fig.update_layout(title="Graphique des Résidus", xaxis_title="Valeurs Prédites", yaxis_title="Résidus", **DARK_THEME_LAYOUT)
        return pio.to_json(fig)
```

---

## 📅 Planning & Jalons Personnels

* **Fin Semaine 1 :** Moteur de calcul des statistiques descriptives opérationnel, gabarits Plotly configurés avec le thème sombre.
* **Fin Semaine 2 :** Galerie complète des graphiques univariés et bivariés (histogrammes, boxplots, matrice de corrélation).
* **Fin Semaine 3 :** Graphiques diagnostiques de régression (droite, résidus) et de classification (ROC, matrice de confusion).
* **Fin Semaine 4 :** Graphiques de clustering K-Means (Elbow, clusters PCA), courbe de perte du réseau de neurones et feature importance.

---

## 🏆 Critères de Réussite & Check-list d'Évaluation
- [ ] Tous les graphiques sont fluides, réactifs au redimensionnement et s'intègrent dans le thème sombre.
- [ ] Les infobulles au survol affichent des valeurs précises et lisibles.
- [ ] La matrice de corrélation et la matrice de confusion sont intuitives et annotées.
- [ ] Le graphique K-Means Elbow et les courbes ROC permettent une réelle prise de décision pour l'utilisateur.

