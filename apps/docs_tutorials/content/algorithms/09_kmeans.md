# K-Means Clustering (Partitionnement Non-Supervisé)

L'algorithme des **K-Moyennes** (*K-Means*) est le modèle de référence en apprentissage non-supervisé pour partitionner un ensemble de $n$ observations non étiquetées en $K$ sous-groupes homogènes distincts (clusters) $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$.

---

## 1. Fonction Objectif (Inertie Intra-Cluster)

K-Means cherche à partitionner les données de manière à minimiser la somme des carrés des distances entre chaque point et le centroïde $\mu_k$ de son cluster assigné :

$$J(\mathcal{C}, \mu) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

où le centroïde du cluster $k$ est le barycentre des observations affectées à $C_k$ :
$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

---

## 2. L'Algorithme de Lloyd (Alternance en 2 Étapes)

1. **Initialisation :** Sélection de $K$ centroïdes initiaux (méthode `k-means++` pour éviter les minima locaux sous-optimaux).
2. **Étape d'Assignation (Expectation) :** Chaque observation $x_i$ est affectée au cluster dont le centroïde est le plus proche :
   $$C_k^{(t)} = \left\{ x_i : \|x_i - \mu_k^{(t)}\|^2 \le \|x_i - \mu_j^{(t)}\|^2, \forall j \in \{1, \dots, K\} \right\}$$
3. **Étape de Mise à Jour (Maximization) :** Recalcul de la position des centroïdes comme moyenne des points nouvellement assignés :
   $$\mu_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{x_i \in C_k^{(t)}} x_i$$
4. **Convergence :** Répétition des étapes 2 et 3 jusqu'à ce que les centroïdes ne bougent plus ($\|\mu^{(t+1)} - \mu^{(t)}\| < \epsilon$).

---

## 3. Détermination du Nombre Optimal de Clusters $K$

### A. La Méthode du Coude (Elbow Method)
On trace l'inertie intra-cluster en fonction de $K \in [2, 10]$. Le nombre optimal de clusters correspond au "coude" de la courbe, c'est-à-dire le point d'inflexion où l'ajout d'un cluster supplémentaire n'apporte plus de réduction significative de variance.

### B. Le Score de Silhouette
Pour chaque point $i$, on compare la distance moyenne intra-cluster $a(i)$ à la distance moyenne avec le cluster voisin le plus proche $b(i)$ :

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$

* **Proche de $+1$ :** Point très bien séparé et parfaitement assigné.
* **Proche de $0$ :** Point situé sur la frontière entre deux clusters.
* **Négatif :** Point probablement mal assigné.

---

## 4. Implémentation Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)
score = silhouette_score(X_scaled, cluster_labels)
```

