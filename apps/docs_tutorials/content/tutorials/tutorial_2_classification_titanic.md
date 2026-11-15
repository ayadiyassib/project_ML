# Tutoriel Guidé 2 : Prédire la Survie des Passagers du Titanic

Dans ce deuxième cas pratique, nous abordons une problématique classique de **classification supervisée binaire** avec données hétérogènes (numériques et textuelles) et valeurs manquantes.

---

## 🎯 Objectifs Pédagogiques
1. Charger le dataset `titanic.csv` contenant des passagers avec leur classe, âge, sexe, tarif et port d'embarquement.
2. Traiter les données manquantes (imputation par la médiane pour l'âge).
3. Encoder les variables qualitatives (`Sex` en binaire, `Embarked` en One-Hot Encoding).
4. Entraîner et comparer :
   * **Régression Logistique** (modèle probabiliste de base)
   * **Support Vector Machine (SVC)** avec noyau RBF
   * **Random Forest Classifier** (forêt d'arbres)
5. Interpréter la **Matrice de Confusion**, la **Courbe ROC** et l'**AUC**.

---

## 📋 Déroulé Pas-à-Pas dans le Studio

### Étape 1 : Ingestion
1. Dans **1. Datasets**, sélectionnez le benchmark **Titanic**.
2. Identifiez les variables disponibles : `Survived` (cible binaire 0 ou 1), `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.

### Étape 2 : Exploration & Valeurs Manquantes
1. Rendez-vous dans **2. Exploration (EDA)**.
2. Le tableau des valeurs manquantes met en évidence :
   * Environ 20% de valeurs manquantes sur la colonne `Age`.
   * Quelques valeurs manquantes sur `Embarked`.
   * Forte disparité du taux de survie selon le sexe (taux de survie des femmes $\approx 74\%$ vs hommes $\approx 18\%$).

### Étape 3 : Prétraitement sans Data Leakage
1. Rendez-vous dans **3. Prétraitement** :
   * **Variable cible :** Sélectionner `Survived`.
   * **Colonnes numériques (`Age`, `Fare`) :** Imputation par la médiane + `StandardScaler`.
   * **Colonnes catégorielles (`Sex`, `Embarked`, `Pclass`) :** Imputation par la valeur la plus fréquente (mode) + `OneHotEncoder`.
2. Lancez le découpage **Train / Test** (75% / 25%) stratifié.

### Étape 4 : Entraînement des Modèles
1. Rendez-vous dans **4. Modélisation**.
2. Lancez la comparaison avec une validation croisée à 5 plis ($K=5$) :
   * **Régression Logistique :** Accuracy $\approx 80\%$, F1-score $\approx 0.74$.
   * **SVM (RBF Kernel) :** Accuracy $\approx 82\%$.
   * **Random Forest :** Accuracy $\approx 83.5\%$, AUC ROC $\approx 0.88$.

### Étape 5 : Diagnostics de Classification
1. Dans **5. Diagnostics**, consultez :
   * **La Matrice de Confusion :** Vérifiez le compromis entre Faux Positifs et Faux Négatifs.
   * **La Courbe ROC :** Observez la proximité avec le coin supérieur gauche ($AUC = 0.88$).

### Étape 6 : Prédiction Interactive
1. Dans **6. Inférence**, simulez un passager : `Femme, 1ère classe, 28 ans, tarif $80`.
2. Obtenez la probabilité prédite de survie (ex: $94.2\%$).

