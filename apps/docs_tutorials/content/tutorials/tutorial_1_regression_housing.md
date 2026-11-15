# Tutoriel Guidé 1 : Prédire le Prix Médian des Logements en Californie

Bienvenue dans ce premier tutoriel pratique ! Nous allons parcourir le pipeline complet de ML Studio pour résoudre une tâche de **régression continue** : estimer la valeur médiane des maisons californiennes (`MedHouseVal`).

---

## 🎯 Objectifs Pédagogiques
1. Charger le dataset benchmark `california_housing.csv` (20 640 observations, 8 variables numériques).
2. Explorer les corrélations avec la variable cible (`MedInc` - Revenu médian).
3. Construire un pipeline de prétraitement avec `StandardScaler` pour normaliser les échelles.
4. Comparer les performances de la **Régression Linéaire**, de la **Régression Polynomiale** ($d=2$) et de **Random Forest Regressor**.
5. Analyser le graphique des résidus et le score $R^2$.

---

## 📋 Déroulé Pas-à-Pas dans le Studio

### Étape 1 : Ingestion du Jeu de Données
1. Rendez-vous dans le menu **1. Datasets & Ingestion**.
2. Dans la section "Jeux de Données de Référence", cliquez sur **Charger California Housing**.
3. Observez le résumé : 20 640 lignes, 8 colonnes explicatives, aucune valeur manquante.

### Étape 2 : Analyse Exploratoire (EDA)
1. Cliquez sur **2. Exploration (EDA)** dans le stepper en haut.
2. Examinez la **Matrice de Corrélation Plotly** :
   * Notez la forte corrélation positive ($r \approx 0.69$) entre `MedInc` et la cible `MedHouseVal`.
   * Examinez l'histogramme de la cible : plafonnement artificiel des valeurs au-delà de \$500,000.

### Étape 3 : Configuration du Prétraitement
1. Cliquez sur **3. Prétraitement**.
2. Définissez `MedHouseVal` comme **Variable Cible**.
3. Choisissez le ratio de découpage : **80% Entraînement / 20% Test**.
4. Activez le **StandardScaler** pour centrer et réduire toutes les colonnes numériques.

### Étape 4 : Modélisation et Comparaison
1. Cliquez sur **4. Modélisation**.
2. Entraînez successivement :
   * **Régression Linéaire :** $R^2 \approx 0.60$, $RMSE \approx 0.72$.
   * **Régression Polynomiale ($d=2$) :** $R^2 \approx 0.68$ (meilleure capture des interactions géographiques `Latitude` $\times$ `Longitude`).
   * **Random Forest Regressor :** $R^2 \approx 0.81$ (excellent pouvoir prédictif non-linéaire).

### Étape 5 : Diagnostics & Résidus
1. Dans **5. Diagnostics**, examinez le nuage de points des **Résidus vs Valeurs Prédites**.
2. Vérifiez que les erreurs sont distribuées de manière aléatoire et symétrique autour de zéro (hypothèse d'homoscédasticité).

### Étape 6 : Inférence & Export
1. Dans **6. Inférence & Export**, saisissez un profil de quartier personnalisé (`MedInc = 4.5`, `HouseAge = 25`, `AveRooms = 5.8`).
2. Obtenez instantanément la valeur estimée de la propriété.
3. Téléchargez le pipeline sérialisé `.joblib` et le rapport récapitulatif PDF d'audit.

