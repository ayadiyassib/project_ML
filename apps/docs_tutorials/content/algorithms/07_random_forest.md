# Forêts Aléatoires (Random Forest Classifier & Regressor)

La **Forêt Aléatoire** (*Random Forest*) est un algorithme d'apprentissage d'ensemble (*Ensemble Learning*) basé sur la méthode du **Bagging** (*Bootstrap Aggregating*), inventé par Leo Breiman. Il combine des dizaines ou centaines d'arbres de décision décorrélés pour réduire drastiquement la variance sans augmenter le biais.

---

## 1. Mécanisme de Construction de l'Ensemble

Pour construire une forêt de $B$ arbres décisionnels :

### Étape 1 : Échantillonnage Bootstrap
Pour chaque arbre $b \in \{1, \dots, B\}$, on tire avec remise un sous-échantillon de taille $n$ à partir du jeu d'entraînement. En moyenne, chaque échantillon bootstrap contient environ $63.2\%$ des observations uniques. Les $36.8\%$ restants constituent les échantillons **Out-Of-Bag (OOB)**.

### Étape 2 : Décorrélation des Arbres (Random Subspace Method)
À chaque nœud de chaque arbre, au lieu de chercher la meilleure scission parmi l'intégralité des $p$ variables, l'algorithme sélectionne aléatoirement un sous-ensemble restreint de $m$ variables :
* **Classification :** $m = \lfloor \sqrt{p} \rfloor$
* **Régression :** $m = \lfloor p / 3 \rfloor$

Cette sélection aléatoire empêche les variables très fortes de dominer tous les arbres au même endroit, assurant une forte diversité statistique dans les prédictions.

---

## 2. Agrégation des Prédictions

* **Classification : Vote Majoritaire**
  $$\hat{C}_{RF}(x) = \arg\max_k \sum_{b=1}^{B} \mathbb{I}(T_b(x) = k)$$
* **Régression : Moyenne Arithmétique**
  $$\hat{y}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

---

## 3. Évaluation Out-Of-Bag (OOB)

L'erreur OOB fournit une estimation non biaisée de l'erreur de généralisation sans nécessiter de jeu de validation distinct : chaque observation est prédite uniquement par les arbres qui ne l'ont pas vue pendant leur phase d'apprentissage bootstrap.

---

## 4. Importance des Variables (Feature Importance)

L'importance de la variable $j$ est calculée par la diminution moyenne totale d'impureté (*Mean Decrease in Impurity* - MDI) induite par cette variable à travers l'ensemble des arbres de la forêt.

---

## 5. Implémentation Scikit-Learn

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    oob_score=True,
    random_state=42
)
rf_clf.fit(X_train, y_train)

# Régression
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    oob_score=True,
    random_state=42
)
rf_reg.fit(X_train, y_train)
```

