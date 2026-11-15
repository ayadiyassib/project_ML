# Arbres de Décision (Classification & Régression)

Les **Arbres de Décision** (CART - *Classification and Regression Trees*) sont des modèles non-paramétriques qui apprennent des règles de décision hiérarchiques sous forme de structure arborescente (nœud racine, nœuds internes, feuilles).

---

## 1. Critères de Séparation en Classification

À chaque nœud $m$ contenant une région $R_m$ avec $N_m$ observations, la proportion d'individus de la classe $k \in \{1, \dots, K\}$ est :
$$p_{mk} = \frac{1}{N_m} \sum_{x_i \in R_m} \mathbb{I}(y_i = k)$$

### Critère 1 : Impureté de Gini (Défaut CART)
L'impureté de Gini mesure la probabilité d'erreur de classification si une étiquette était attribuée aléatoirement selon la distribution empirique du nœud :

$$I_G(m) = 1 - \sum_{k=1}^{K} p_{mk}^2 \in \left[0, 1 - \frac{1}{K}\right]$$

Un nœud pur (toutes les observations sont d'une même classe) a une impureté $I_G = 0$.

### Critère 2 : Entropie de Shannon
Mesure de l'incertitude informationnelle contenue dans la distribution des classes :

$$H(m) = - \sum_{k=1}^{K} p_{mk} \log_2(p_{mk})$$

Le gain d'information résultant d'une scission en deux sous-nœuds gauche ($L$) et droit ($R$) est :
$$\Delta H = H(m) - \left( \frac{N_L}{N_m} H(L) + \frac{N_R}{N_m} H(R) \right)$$

---

## 2. Critère de Séparation en Régression

Pour un problème de régression, la valeur prédite dans chaque feuille terminale est la moyenne des cibles de cette région :
$$\hat{y}_m = \frac{1}{N_m} \sum_{i \in R_m} y_i$$

L'algorithme choisit la scission $(j, s)$ (variable $j$, seuil $s$) qui minimise la variance résiduelle intra-nœud :

$$MSE(m) = \frac{1}{N_m} \sum_{i \in R_m} (y_i - \hat{y}_m)^2$$

---

## 3. Contrôle du Surapprentissage (Élagage / Pruning)

Sans contrainte, un arbre de décision continuera de créer des branches jusqu'à ce que chaque feuille contienne une seule observation, menant à un surapprentissage sévère. Hyperparamètres essentiels :

* `max_depth` : Profondeur maximale de l'arbre.
* `min_samples_split` : Nombre minimal d'échantillons requis pour autoriser une scission.
* `min_samples_leaf` : Nombre minimal d'échantillons devant résider dans chaque feuille terminale.

---

## 4. Implémentation Scikit-Learn

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
tree_clf = DecisionTreeClassifier(
    criterion='gini',       # ou 'entropy'
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
tree_clf.fit(X_train, y_train)

# Régression
tree_reg = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=5
)
tree_reg.fit(X_train, y_train)
```

