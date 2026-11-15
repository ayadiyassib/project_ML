# Classifieur Naïf Bayésien (GaussianNB & MultinomialNB)

Le **Classifieur Naïf Bayésien** (*Naive Bayes*) est un algorithme de classification probabiliste génératif reposant sur l'application stricte du **Théorème de Bayes**, combiné à l'hypothèse d'indépendance conditionnelle des descripteurs sachant la classe.

---

## 1. Fondement Théorique : Le Théorème de Bayes

Étant donné un vecteur d'observations $X = (x_1, x_2, \dots, x_p)$, la règle de Bayes donne la probabilité a posteriori de la classe $Y = k$ :

$$P(Y = k \mid X) = \frac{P(X \mid Y = k) P(Y = k)}{P(X)}$$

où :
* $P(Y = k)$ est la **probabilité a priori** de la classe $k$, estimée par la fréquence empirique : $\hat{P}(Y=k) = \frac{n_k}{n}$.
* $P(X \mid Y = k)$ est la **vraisemblance** (*likelihood*).
* $P(X) = \sum_{j} P(X \mid Y=j) P(Y=j)$ est la probabilité marginale d'évidence, constante pour toutes les classes.

---

## 2. L'Hypothèse "Naïve" d'Indépendance Conditionnelle

Pour éviter de devoir estimer la densité conjointe multidimensionnelle $P(x_1, \dots, x_p \mid Y=k)$ qui nécessiterait un volume de données exponentiel, l'algorithme formule l'hypothèse d'**indépendance conditionnelle mutuelle** entre les variables :

$$P(X \mid Y = k) = \prod_{j=1}^{p} P(x_j \mid Y = k)$$

La règle de classification optimale (*Maximum A Posteriori* - MAP) s'écrit :

$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} \left( \ln P(Y = k) + \sum_{j=1}^{p} \ln P(x_j \mid Y = k) \right)$$

---

## 3. Variantes Selon la Nature des Données

### A. Gaussian Naive Bayes (`GaussianNB`)
Adapté aux variables quantitatives continues. Chaque descripteur suit une loi normale conditionnellement à la classe :

$$P(x_j \mid Y = k) = \frac{1}{\sqrt{2\pi \sigma_{jk}^2}} \exp\left( - \frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2} \right)$$

où $\mu_{jk}$ et $\sigma_{jk}^2$ sont la moyenne et la variance empiriques de la variable $j$ pour la classe $k$.

### B. Multinomial Naive Bayes (`MultinomialNB`)
Adapté aux données discrètes ou de comptage (ex: fréquences de mots dans l'analyse textuelle / NLP).

---

## 4. Implémentation Scikit-Learn

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Pour variables quantitatives continues
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)
```

