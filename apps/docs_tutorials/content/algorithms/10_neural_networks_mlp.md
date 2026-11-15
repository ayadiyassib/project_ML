# Perceptron Multicouche (ANN / MLP Classifier & Regressor)

Le **Perceptron Multicouche** (*Multi-Layer Perceptron* - MLP) est le modèle d'entrée du Deep Learning : un réseau de neurones artificiels à propagation avant (*Feedforward Neural Network*) constitué d'une couche d'entrée, d'une ou plusieurs couches cachées et d'une couche de sortie.

---

## 1. Architecture & Propagation Avant (Forward Pass)

Pour un réseau comportant $L$ couches :
Soit $a^{[l-1]}$ le vecteur d'activation émis par la couche précédente (avec $a^{[0]} = X$) :

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}\left( z^{[l]} \right)$$

où :
* $W^{[l]}$ est la matrice des poids synaptiques de la couche $l$.
* $b^{[l]}$ est le vecteur de biais.
* $g^{[l]}$ est la fonction d'activation non-linéaire.

### Principales Fonctions d'Activation :
* **ReLU (Rectified Linear Unit) :** $g(z) = \max(0, z)$ (défaut recommandé, évite l'évanouissement du gradient).
* **Sigmoïde Logistique :** $g(z) = \frac{1}{1 + e^{-z}}$
* **Tangente Hyperbolique (Tanh) :** $g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

---

## 2. Rétropropagation du Gradient (Backpropagation)

L'apprentissage consiste à ajuster les poids $W$ et biais $b$ pour minimiser la fonction de perte $J(\theta)$ (Entropie Croisée ou MSE).

Par application de la règle de dérivation en chaîne (*Chain Rule*) :
$$\delta^{[l]} = \frac{\partial J}{\partial z^{[l]}} = \left( (W^{[l+1]})^T \delta^{[l+1]} \right) \odot (g^{[l]})'(z^{[l]})$$

$$\frac{\partial J}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T, \quad \frac{\partial J}{\partial b^{[l]}} = \delta^{[l]}$$

Mise à jour des poids avec le taux d'apprentissage $\eta$ :
$$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial J}{\partial W^{[l]}}$$

### Optimiseurs Supportés :
* **Adam :** Estimation adaptative des moments d'ordre 1 et 2 (standard industriel).
* **SGD :** Descente de gradient stochastique avec moment (*momentum*).
* **L-BFGS :** Méthode quasi-Newton idéale pour les petits datasets.

---

## 3. Implémentation Scikit-Learn

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Réseau à 2 couches cachées de 64 et 32 neurones
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,           # Régularisation L2
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,    # Arrêt précoce sur validation
    random_state=42
)

mlp_clf.fit(X_train, y_train)
y_pred = mlp_clf.predict(X_test)
loss_curve = mlp_clf.loss_curve_  # Historique de la perte par époque
```

