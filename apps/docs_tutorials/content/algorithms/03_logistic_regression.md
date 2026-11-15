# Régression Logistique (Binaire & Multiclasse)

La **Régression Logistique** est l'algorithme probabiliste standard pour les problèmes de classification supervisée binaire ($y \in \{0, 1\}$) et multiclasse ($y \in \{1, \dots, K\}$).

---

## 1. Modèle Probabiliste & Fonction Sigmoïde

Contrairement à la régression linéaire, la variable cible étant qualitative, nous cherchons à modéliser la probabilité a posteriori :

$$p(X) = P(Y = 1 \mid X)$$

Pour contraindre la sortie dans l'intervalle strictement valide $[0, 1]$, on utilise la **fonction logistique (ou sigmoïde)** :

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

où $z = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p = \beta^T X$.

---

## 2. Le Rapport des Cotes (Odds Ratio) et Logit

Le rapport entre la probabilité de succès et la probabilité d'échec est appelé la **cote** (*odds*) :

$$\text{Odds} = \frac{P(Y = 1 \mid X)}{1 - P(Y = 1 \mid X)} = e^{\beta^T X}$$

En appliquant le logarithme népérien, on obtient la fonction **Logit**, linéaire dans ses paramètres :

$$\ln\left(\frac{P(Y = 1 \mid X)}{1 - P(Y = 1 \mid X)}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$$

La frontière de décision géométrique correspond à $P(Y=1 \mid X) = 0.5$, ce qui équivaut à l'hyperplan :
$$\beta^T X = 0$$

---

## 3. Fonction de Perte : Entropie Croisée Binaire (Log-Loss)

Les paramètres $\beta$ sont estimés par la méthode du **Maximum de Vraisemblance**.
La log-vraisemblance négative (ou log-loss) à minimiser est :

$$J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \frac{1}{2C} \|\beta\|_2^2$$

où $C > 0$ est l'inverse de la force de régularisation ($L_2$ ou $L_1$).
Cette fonction de perte étant convexe, elle admet un minimum global unique résolu par descente de gradient (algorithmes L-BFGS, Newton-CG, SAGA).

---

## 4. Extension Multiclasse : Softmax

Pour $K > 2$ classes, la fonction **Softmax** généralise la sigmoïde :

$$P(Y = k \mid X) = \frac{e^{\beta_k^T X}}{\sum_{j=1}^{K} e^{\beta_j^T X}}$$

---

## 5. Implémentation Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression

# Classification binaire ou multiclasse avec régularisation L2
clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=500,
    multi_class='auto'
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
```

