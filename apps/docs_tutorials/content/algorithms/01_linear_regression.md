# Régression Linéaire (Simple & Multiple)

La **Régression Linéaire** est l'algorithme fondamental de l'apprentissage supervisé pour modéliser une variable quantitative continue $y \in \mathbb{R}$ en fonction d'un ensemble de variables explicatives $X = (X_1, X_2, \dots, X_p) \in \mathbb{R}^p$.

---

## 1. Formulation Mathématique

### Modèle Linéaire Théorique
Pour chaque observation $i \in \{1, \dots, n\}$, le modèle postule :

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \varepsilon_i$$

Sous forme matricielle compacte :
$$y = X \beta + \varepsilon$$

où :
* $y \in \mathbb{R}^{n \times 1}$ est le vecteur des cibles observées.
* $X \in \mathbb{R}^{n \times (p+1)}$ est la matrice de design (avec une première colonne de 1 pour l'intercept $\beta_0$).
* $\beta = (\beta_0, \beta_1, \dots, \beta_p)^T \in \mathbb{R}^{p+1}$ est le vecteur des paramètres inconnus à estimer.
* $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$ est le terme d'erreur résiduelle aléatoire.

---

## 2. Estimation par les Moindres Carrés Ordinaires (OLS)

L'objectif de l'estimateur des **Moindres Carrés Ordinaires** (Ordinary Least Squares - OLS) est de minimiser la somme des carrés des résidus (Residual Sum of Squares, $RSS$) :

$$RSS(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \|y - X\beta\|^2 = (y - X\beta)^T (y - X\beta)$$

En annulant le gradient par rapport à $\beta$ :
$$\nabla_\beta RSS(\beta) = -2 X^T (y - X\beta) = 0 \implies X^T X \beta = X^T y$$

Sous l'hypothèse que la matrice $X^T X$ est inversible (absence de multicolinéarité parfaite), la solution analytique exacte est :

$$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$$

---

## 3. Hypothèses Classiques de Gauss-Markov

Pour que l'estimateur OLS soit **BLUE** (*Best Linear Unbiased Estimator* - Meilleur estimateur linéaire sans biais), les hypothèses suivantes doivent être respectées :

1. **Linéarité :** La relation entre la variable cible $y$ et les variables explicatives $X$ est linéaire dans les paramètres.
2. **Exogénéité stricte :** L'espérance conditionnelle des erreurs est nulle : $\mathbb{E}[\varepsilon | X] = 0$.
3. **Homoscédasticité :** La variance des termes d'erreur est constante pour toutes les observations : $\text{Var}(\varepsilon_i | X) = \sigma^2, \forall i$.
4. **Non-autocorrélation :** $\text{Cov}(\varepsilon_i, \varepsilon_j | X) = 0 \text{ pour } i \neq j$.
5. **Absence de multicolinéarité parfaite :** Le rang de la matrice $X$ est égal à $p+1$ ($\text{det}(X^T X) \neq 0$).
6. **Normalité des résidus :** $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$ (nécessaire pour les tests de Student et de Fisher).

---

## 4. Métriques d'Évaluation en Régression

* **Erreur Quadratique Moyenne (MSE) :**
  $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Racine de l'Erreur Quadratique Moyenne (RMSE) :**
  $$RMSE = \sqrt{MSE}$$
* **Erreur Absolue Moyenne (MAE) :**
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **Coefficient de Détermination ($R^2$) :**
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} \in (-\infty, 1]$$

---

## 5. Implémentation Scikit-Learn dans ML Studio

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialisation du modèle
model = LinearRegression(fit_intercept=True)

# Entraînement
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)
```

