# Support Vector Machines (SVM - SVC & SVR)

Les **Machines à Vecteurs de Support** (Support Vector Machines - SVM) sont des algorithmes géométriques supervisés puissants, conçus pour la classification binaire/multiclasse (**SVC**) et la régression (**SVR**).

---

## 1. Principe Géométrique : Séparation à Marge Maximale

Pour un ensemble linéairement séparable $\{(x_i, y_i)\}_{i=1}^n$ avec $y_i \in \{-1, +1\}$, il existe une infinité d'hyperplans séparateurs $w^T x + b = 0$.

Le SVM sélectionne l'unique hyperplan qui maximise la **marge géométrique** $\gamma = \frac{2}{\|w\|_2}$ séparant les classes opposées.

### Formulation du Problème Primal avec Variables Ressorts (Marge Douce)
En pratique, les données réelles se chevauchent. On introduit des variables d'écart (*slack variables*) $\xi_i \ge 0$ et un paramètre de régularisation $C > 0$ :

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i$$

sous les contraintes :
$$y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i \in \{1, \dots, n\}$$

* **$C$ élevé :** Forte pénalisation des erreurs, marge étroite, risque de surapprentissage.
* **$C$ faible :** Marge plus large autorisant des erreurs de classification, meilleure capacité de généralisation.

Les points situés exactement sur la marge ou la violant sont appelés les **vecteurs de support** (*support vectors*). Ce sont les seuls points qui déterminent la position de l'hyperplan optimal.

---

## 2. L'Astuce du Noyau (Kernel Trick)

Lorsque la frontière de décision est non-linéaire, on projette les données dans un espace de Hilbert de dimension supérieure voire infinie via une fonction $\phi(x)$. Grâce au théorème de Mercer, il n'est pas nécessaire de calculer explicitement $\phi(x)$ : il suffit d'évaluer la fonction noyau $K(x, x') = \langle \phi(x), \phi(x') \rangle$.

### Principaux Noyaux Disponibles dans ML Studio :
1. **Noyau Linéaire :**
   $$K(x, x') = x^T x'$$
2. **Noyau RBF (Radial Basis Function / Gaussien) :**
   $$K(x, x') = \exp\left( -\gamma \|x - x'\|^2 \right)$$
   où $\gamma > 0$ définit la portée de l'influence d'un vecteur de support unique.
3. **Noyau Polynomial :**
   $$K(x, x') = (\gamma x^T x' + r)^d$$

---

## 3. Support Vector Regression (SVR)

En régression, le SVR construit un tube d'insensibilité $\varepsilon$ (*$\varepsilon$-insensitive tube*). Les erreurs de prédiction inférieures à $\varepsilon$ ne sont pas pénalisées, ce qui confère une robustesse exceptionnelle face aux valeurs aberrantes (outliers).

---

## 4. Implémentation Scikit-Learn

```python
from sklearn.svm import SVC, SVR

# Classification non linéaire avec noyau RBF
svc = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=True
)
svc.fit(X_train, y_train)

# Régression avec tube d'insensibilité
svr = SVR(
    C=1.0,
    kernel='rbf',
    epsilon=0.1
)
svr.fit(X_train, y_train)
```

