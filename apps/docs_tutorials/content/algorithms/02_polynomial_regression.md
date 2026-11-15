# Régression Polynomiale

La **Régression Polynomiale** est une extension de la régression linéaire permettant de capturer des relations non-linéaires entre les descripteurs et la variable cible tout en conservant la linéarité par rapport aux paramètres estimés $\beta$.

---

## 1. Formulation Mathématique

Pour une variable explicative unique $X$, le modèle polynomial d'ordre $d \ge 2$ s'écrit :

$$y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \dots + \beta_d X^d + \varepsilon$$

Pour $p$ variables explicatives $X = (X_1, \dots, X_p)$, le développement polynomial inclut tous les termes de degrés supérieurs et les termes d'interaction croisée (ex: $X_1 X_2, X_1^2 X_2$, etc.).

### Transformation de l'Espace des Descripteurs
On applique une fonction de transformation $\phi(X) : \mathbb{R}^p \to \mathbb{R}^D$ qui projette l'espace initial dans un espace vectoriel de dimension supérieure $D > p$.

La matrice de design transformée s'exprime :
$$\Phi = \begin{bmatrix}
1 & x_{1} & x_{1}^2 & \dots & x_{1}^d \\
1 & x_{2} & x_{2}^2 & \dots & x_{2}^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n} & x_{n}^2 & \dots & x_{n}^d
\end{bmatrix}$$

L'estimateur analytique s'obtient toujours par la formule OLS :
$$\hat{\beta} = (\Phi^T \Phi)^{-1} \Phi^T y$$

---

## 2. Compromis Biais - Variance & Surapprentissage (Overfitting)

Le choix du degré polynomial $d \in [2, 5]$ est crucial :

* **Degré trop faible ($d = 1$) : Sous-apprentissage (Underfitting)**
  * Fort biais théorique, incapacité à capturer la courbure réelle des données.
* **Degré optimal ($d \in [2, 3]$) : Équilibre optimal**
  * Minimisation de l'erreur généralisée sur le jeu de test.
* **Degré excessif ($d \ge 4$ ou $5$) : Surapprentissage (Overfitting)**
  * Forte variance statistique : le modèle apprend le bruit aléatoire du jeu d'entraînement, oscillant de manière instable (phénomène de Runge) et perdant tout pouvoir prédictif sur des données nouvelles.

---

## 3. Implémentation Scikit-Learn Pipeline

Dans ML Studio, la régression polynomiale est encapsulée dans un `Pipeline` pour éviter tout risque de fuite de données :

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

degree = 3  # Choix interactif dans l'interface

poly_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

poly_pipeline.fit(X_train, y_train)
y_pred = poly_pipeline.predict(X_test)
```

