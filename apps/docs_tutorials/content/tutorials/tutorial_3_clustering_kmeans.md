# Tutoriel Guidé 3 : Segmentation Non-Supervisée & Détection de Profils

Ce troisième tutoriel explore l'apprentissage **non supervisé** : nous disposons de données sans variable cible étiquetée, et notre objectif est de découvrir automatiquement des groupes d'individus homogènes.

---

## 🎯 Objectifs Pédagogiques
1. Utiliser le dataset `wine.csv` (178 échantillons de vins décrits par 13 caractéristiques physico-chimiques : alcool, acide malique, cendres, magnésium, flavonoïdes, etc.).
2. Comprendre l'impact critique de la normalisation (`StandardScaler`) sur le calcul des distances euclidiennes.
3. Déterminer le nombre optimal de clusters $K$ en traçant la **Courbe du Coude** (*Elbow Curve*) et le **Score de Silhouette**.
4. Visualiser les clusters formés en 2D via une **Analyse en Composantes Principales (PCA)**.

---

## 📋 Déroulé Pas-à-Pas dans le Studio

### Étape 1 : Ingestion du Dataset
1. Rendez-vous dans **1. Datasets** et chargez le jeu de données **Wine**.
2. Remarquez la disparité des échelles : la teneur en `proline` varie de 278 à 1680, alors que l'indice `nonflavanoid_phenols` varie entre 0.13 et 0.66.

### Étape 2 : Préparation pour le Clustering
1. Rendez-vous dans **3. Prétraitement**.
2. Indiquez **Aucune Cible (Tâche non supervisée)**.
3. Appliquez obligatoirement un **StandardScaler** : chaque descripteur aura une moyenne nulle ($\mu = 0$) et un écart-type unitaire ($\sigma = 1$). Sans cette étape, le descripteur `proline` pèserait pour plus de 90% dans le calcul de la distance !

### Étape 3 : Recherche du $K$ Optimal
1. Rendez-vous dans **4. Modélisation** et sélectionnez **K-Means Clustering**.
2. Lancez le balayage automatique pour $K \in [2, 8]$ :
   * **Courbe du coude :** Une cassure nette d'inertie intra-cluster apparaît distinctement à $K = 3$.
   * **Score de Silhouette :** Le coefficient moyen atteint son pic à $K = 3$ ($s \approx 0.28$ dans l'espace standardisé complet).

### Étape 4 : Visualisation Diagnostique (Projection PCA 2D)
1. Rendez-vous dans **5. Diagnostics**.
2. Observez la projection spatiale 2D générée par Plotly grâce aux deux premières composantes principales ($PC_1$ et $PC_2$).
3. Les 3 groupes de vins se séparent nettement dans l'espace factoriel avec des centroïdes bien espacés.

### Étape 5 : Export & Profils
1. Téléchargez la segmentation pour chaque échantillon.
2. Générez le rapport d'audit PDF récapitulant les coordonnées des barycentres de chaque cluster.

