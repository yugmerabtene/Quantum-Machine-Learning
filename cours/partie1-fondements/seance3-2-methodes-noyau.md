# Séance 3.2 — Méthodes à noyau classiques

## Support Vector Machine (SVM) et kernel trick

La machine à vecteurs de support (SVM) est un classifieur linéaire qui maximise la **marge** — la distance entre la frontière de décision et les échantillons les plus proches (vecteurs de support). Dans son espace primal, le SVM cherche un hyperplan $w \cdot x + b = 0$ tel que :

$$
y_i (w \cdot x_i + b) \geq 1, \quad \forall i
$$

La marge vaut $2 / \|w\|$. Maximiser la marge revient à minimiser $\|w\|^2$ sous contraintes. Le problème dual s'écrit :

$$
\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle
$$
$$
\text{s.c.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0
$$

La fonction de décision est alors :

$$
f(x) = \text{sign}\left( \sum_{i} \alpha_i y_i \langle x_i, x \rangle + b \right)
$$

### Kernel trick

Le **kernel trick** consiste à remplacer le produit scalaire $\langle x_i, x_j \rangle$ par un noyau $k(x_i, x_j)$ correspondant à un produit scalaire dans un espace de redescription $\mathcal{H}$ (RKHS) de dimension possiblement infinie :

$$
f(x) = \text{sign}\left( \sum_{i} \alpha_i y_i \, k(x_i, x) + b \right)
$$

Cette substitution transforme un classifieur linéaire en un classifieur non-linéaire sans augmenter le coût algorithmique, le calcul du noyau étant en $O(d)$ contre $O(\dim(\mathcal{H}))$ pour une explicitation de la feature map.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=200, n_features=20, random_state=42)

# SVM linéaire
svm_linear = SVC(kernel='linear')
acc_linear = cross_val_score(svm_linear, X, y, cv=5).mean()

# SVM avec noyau RBF
svm_rbf = SVC(kernel='rbf', gamma='scale')
acc_rbf = cross_val_score(svm_rbf, X, y, cv=5).mean()

print(f"Accuracy SVM linéaire : {acc_linear:.4f}")
print(f"Accuracy SVM RBF : {acc_rbf:.4f}")
```

## Noyaux fondamentaux

### Noyau RBF (gaussien)

$$
k_{\text{RBF}}(x, x') = \exp\left(-\gamma \|x - x'\|^2\right), \quad \gamma = \frac{1}{2\sigma^2}
$$

Le noyau RBF est universel : il peut approximer toute fonction continue sur un compact. Son RKHS est de dimension infinie.

### Noyau polynomial

$$
k_{\text{poly}}(x, x') = (\langle x, x' \rangle + c)^d
$$

Avec $c > 0$, il capture les interactions polynomiales entre features jusqu'au degré $d$.

### Noyau laplacien

$$
k_{\text{lap}}(x, x') = \exp\left(-\frac{\|x - x'\|_1}{\sigma}\right)
$$

Moins régulier que le RBF, il est adapté aux données à variations abruptes.

```python
from sklearn.metrics.pairwise import (
    rbf_kernel, laplacian_kernel, polynomial_kernel
)
import numpy as np

X = np.random.randn(10, 5)
K_rbf = rbf_kernel(X, X, gamma=0.5)
K_poly = polynomial_kernel(X, X, degree=3, coef0=1)
K_lap = laplacian_kernel(X, X, gamma=0.5)

print(f"Matrice RBF (condition) : {np.linalg.cond(K_rbf):.2f}")
print(f"Matrice polynomiale (condition) : {np.linalg.cond(K_poly):.2f}")
```

## Kernel alignment

Le **kernel alignment** mesure la similarité entre deux noyaux (ou entre un noyau et une cible idéale). L'**alignment** empirique est :

$$
\hat{A}(K_1, K_2) = \frac{\langle K_1, K_2 \rangle_F}{\sqrt{\langle K_1, K_1 \rangle_F \langle K_2, K_2 \rangle_F}}
$$

où $\langle A, B \rangle_F = \sum_{i,j} A_{ij} B_{ij}$ est le produit de Frobenius.

Le **Kernel Target Alignment** (KTA) aligne le noyau sur la matrice cible $Y = yy^T$ pour la classification :

$$
\hat{A}(K, yy^T) = \frac{\langle K, yy^T \rangle_F}{\sqrt{\langle K, K \rangle_F \langle yy^T, yy^T \rangle_F}}
$$

```python
def kernel_alignment(K1, K2):
    """Calcule l'alignement entre deux matrices de Gram."""
    K1 = K1 / np.linalg.norm(K1, 'fro')
    K2 = K2 / np.linalg.norm(K2, 'fro')
    return np.sum(K1 * K2)

# Alignement d'un noyau avec la cible idéale
y = np.array([1, 1, 1, -1, -1, -1])
Y_target = np.outer(y, y)
K = rbf_kernel(X[:6], X[:6], gamma=0.5)
alignment = kernel_alignment(K, Y_target)
print(f"Kernel target alignment : {alignment:.4f}")
```

Le KTA est utilisé pour **sélectionner les hyperparamètres** du noyau (ex : $\gamma$ du RBF) sans avoir à ré-entraîner le SVM à chaque fois :

$$
\gamma^* = \arg\max_{\gamma} \hat{A}(K_\gamma, yy^T)
```

## Théorie de la régularisation par noyau

La régularisation par noyau fournit un cadre unifié pour comprendre l'apprentissage dans les RKHS. Le problème général est :

$$
\min_{f \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \ell(y_i, f(x_i)) + \lambda \|f\|_\mathcal{H}^2
$$

Le **théorème de représentation** (Kimeldorf & Wahba, 1971) stipule que la solution optimale s'écrit comme une combinaison linéaire finie de noyaux évalués sur les données :

$$
f^*(x) = \sum_{i=1}^m \alpha_i \, k(x_i, x)
```

Cette représentation est fondamentale car elle ramène un problème d'optimisation en dimension infinie à un problème en dimension $m$ (taille de l'échantillon).

```python
def kernel_ridge_regression(X, y, kernel, lam=1.0):
    """Régression ridge par noyau."""
    m = X.shape[0]
    K = kernel(X, X)
    alpha = np.linalg.solve(K + lam * np.eye(m), y)
    return alpha

# Exemple sur données synthétiques
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(50)

alpha = kernel_ridge_regression(X, y, lambda A, B: rbf_kernel(A, B, gamma=0.5))
K_test = rbf_kernel(X, X, gamma=0.5)
y_pred = K_test @ alpha

import matplotlib.pyplot as plt
plt.plot(X, y, 'o', label='Données')
plt.plot(X, y_pred, '-', label='Prédiction ridge')
plt.legend()
```

### Analyse de la généralisation

Pour un noyau $k$ avec $k(x, x) \leq \kappa$, la complexité de Rademacher du RKHS est bornée par :

$$
\hat{\mathfrak{R}}_m(\mathcal{H}) \leq \kappa \sqrt{\frac{\text{Tr}(K)}{m^2}} \leq \kappa \sqrt{\frac{\lambda_{\max}(K)}{m}}
$$

Cette borne permet de contrôler l'erreur de généralisation des méthodes à noyau.

## Lien avec les méthodes quantiques

Les méthodes à noyau classiques sont le pont naturel vers les **quantum kernel methods** (séance 7.1). Les parallèles sont directs :

- Un noyau quantique est $k_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$, un cas particulier de noyau de Mercer
- Le *quantum kernel alignment* (séance 8.2) est l'analogue quantique du KTA classique
- La concentration exponentielle des kernels quantiques [Agl26] est un problème de régularisation : l'espace de Hilbert $2^n$ est si vaste que les noyaux convergent vers une constante

Le SVM quantique (QSVM, séance 7.2) remplace $k$ par $k_Q$, mais le cadre théorique — dualité, marge, vecteurs de support — reste identique, ce qui permet une analyse comparative rigoureuse.

## Références

- [MRT18] Mohri, M., Rostamizadeh, A. & Talwalkar, A. *Foundations of Machine Learning.* 2nd ed., MIT Press, 2018. Ch. 6–7.
- [Bis06] Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 6.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. Ch. 6.
- [Hav19] Havlíček, V. et al. « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019).
- [Agl26] Agliardi, G. et al. « Mitigating exponential concentration in covariant quantum kernels for subspace and real-world data. » *npj Quantum Information* 12, 12 (2026).
