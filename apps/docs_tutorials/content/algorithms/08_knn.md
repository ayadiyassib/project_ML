# K-Plus Proches Voisins (KNN Classifier & Regressor)

L'algorithme des **K-Plus Proches Voisins** (*K-Nearest Neighbors* - KNN) est une méthode d'apprentissage supervisé non-paramétrique dite **paresseuse** (*lazy learning*) : il n'y a pas de phase d'entraînement explicite ; le modèle stocke l'ensemble des données d'apprentissage et effectue l'intégralité de ses calculs géométriques au moment de l'inférence.

---

## 1. Métriques de Distance Géométrique

Pour mesurer la proximité entre une nouvelle observation $x$ et un point d'apprentissage $x_i \in \mathbb{R}^p$ :

### Distance de Minkowski (Généralisation)
$$D(x, x_i) = \left( \sum_{j=1}^{p} |x_j - x_{ij}|^q \right)^{\frac{1}{q}}$$

* Pour $q = 2$ : **Distance Euclidienne (L2)**
  $$d_{\text{Euclidienne}}(x, x_i) = \sqrt{\sum_{j=1}^{p} (x_j - x_{ij})^2}$$
* Pour $q = 1$ : **Distance de Manhattan (L1)**
  $$d_{\text{Manhattan}}(x, x_i) = \sum_{j=1}^{p} |x_j - x_{ij}|$$

---

## 2. Règle de Décision

Soit $\mathcal{N}_k(x)$ l'ensemble des $k$ observations d'apprentissage les plus proches de $x$ selon la métrique choisie :

* **Classification :** Vote majoritaire (uniforme ou pondéré par l'inverse de la distance $w_i = \frac{1}{d(x, x_i)}$) :
  $$\hat{y} = \arg\max_c \sum_{x_i \in \mathcal{N}_k(x)} w_i \mathbb{I}(y_i = c)$$
* **Régression :** Moyenne (simple ou pondérée) des cibles des $k$ voisins :
  $$\hat{y} = \frac{\sum_{x_i \in \mathcal{N}_k(x)} w_i y_i}{\sum_{x_i \in \mathcal{N}_k(x)} w_i}$$

---

## 3. Impact Crucial du Choix de $k$ et de la Mise à l'Échelle

* **$k = 1$ :** Frontières très bruitées, hypersensibilité aux données aberrantes (fort surapprentissage).
* **$k$ très grand ($k \to n$) :** Prédiction égale à la classe majoritaire ou à la moyenne globale (sous-apprentissage).

> **Avertissement Fondamental :**  
> KNN est extrêmement sensible à l'échelle des variables. Une variable mesurée en milliers (ex: salaire) écrasera totalement une variable mesurée entre 0 et 1 (ex: ratio). Une normalisation préalable (**StandardScaler** ou **MinMaxScaler**) est impérative.

---

## 4. Implémentation Scikit-Learn

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2))
])

knn_pipeline.fit(X_train, y_train)
y_pred = knn_pipeline.predict(X_test)
```

