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

### Exemple complet : SVM avec noyau RBF sur le dataset Iris

L'exemple suivant illustre un pipeline complet de classification par SVM à noyau sur un jeu de données réel, avec sélection d'hyperparamètres par validation croisée et visualisation.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Chargement et préparation des données ---
iris = datasets.load_iris()
X = iris.data[:, :2]  # Seules les 2 premières features pour visualisation
y = iris.target

# Sélection des classes 0 et 1 (Setosa vs Versicolor) pour classification binaire
mask = y < 2
X_binary = X[mask]
y_binary = y[mask]

# --- 2. Pipeline avec standardisation et SVM ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])

# --- 3. Recherche d'hyperparamètres ---
param_grid = {
    'svm__C': [0.1, 1.0, 10.0, 100.0],
    'svm__gamma': [0.01, 0.1, 0.5, 1.0, 5.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid_search.fit(X_binary, y_binary)

print(f"Meilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleure accuracy (CV) : {grid_search.best_score_:.4f}")

# --- 4. Évaluation détaillée ---
y_pred = grid_search.predict(X_binary)
print("\nRapport de classification :")
print(classification_report(y_binary, y_pred, target_names=iris.target_names[:2]))

# --- 5. Visualisation de la frontière de décision ---
def plot_decision_boundary(clf, X, y, title="Frontière de décision SVM"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k', s=50)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig('svm_iris_decision_boundary.png', dpi=150)
    plt.show()

plot_decision_boundary(grid_search.best_estimator_, X_binary, y_binary,
                       title=f"SVM RBF (C={grid_search.best_params_['svm__C']}, "
                             f"γ={grid_search.best_params_['svm__gamma']})")

# --- 6. Étude de l'effet de gamma sur la frontière ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for idx, gamma_val in enumerate([0.01, 0.1, 1.0, 10.0]):
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma=gamma_val))])
    pipe.fit(X_binary, y_binary)
    
    h = 0.02
    x_min, x_max = X_binary[:, 0].min() - 1, X_binary[:, 0].max() + 1
    y_min, y_max = X_binary[:, 1].min() - 1, X_binary[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    axes[idx].scatter(X_binary[:, 0], X_binary[:, 1], c=y_binary, cmap=plt.cm.RdBu, s=30)
    acc = cross_val_score(pipe, X_binary, y_binary, cv=5).mean()
    axes[idx].set_title(f'γ={gamma_val}, acc={acc:.3f}')
    axes[idx].set_xlabel('Longueur sépale')
    
plt.suptitle('Effet du paramètre γ sur la frontière de décision')
plt.tight_layout()
plt.savefig('svm_gamma_effect.png', dpi=150)
plt.show()
```

Cet exemple montre clairement l'effet de $\gamma$ : une valeur faible donne une frontière lisse (sous-apprentissage), une valeur élevée donne une frontière complexe qui épouse les données (sur-apprentissage).

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

### Kernel PCA : réduction de dimension non linéaire

La **Kernel PCA** (KPCA) est une extension non linéaire de l'ACP qui opère dans l'espace de redescription $\mathcal{H}$. Au lieu de diagonaliser la matrice de covariance $\frac{1}{m}XX^T$, on diagonalise la matrice de Gram centrée :

$$
\tilde{K} = K - \mathbf{1}_m K - K \mathbf{1}_m + \mathbf{1}_m K \mathbf{1}_m
$$

où $\mathbf{1}_m = \frac{1}{m} \mathbf{1}\mathbf{1}^T$ est la matrice de moyenne. Les composantes principales sont les vecteurs propres de $\tilde{K}$, et la projection d'un point $x$ sur la $k$-ième composante est :

$$
\phi_k(x) = \sum_{i=1}^m \alpha_i^{(k)} k(x_i, x)
$$

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Données non linéairement séparables (cercles concentriques)
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# KPCA avec noyau RBF
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=5.0)
X_kpca_rbf = kpca_rbf.fit_transform(X_circles)

# KPCA avec noyau polynomial
kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3, coef0=1)
X_kpca_poly = kpca_poly.fit_transform(X_circles)

# KPCA avec noyau linéaire (ACP classique)
kpca_linear = KernelPCA(n_components=2, kernel='linear')
X_kpca_linear = kpca_linear.fit_transform(X_circles)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='bwr', s=20)
axes[0].set_title('Données originales')

axes[1].scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=y_circles, cmap='bwr', s=20)
axes[1].set_title('ACP linéaire')

axes[2].scatter(X_kpca_poly[:, 0], X_kpca_poly[:, 1], c=y_circles, cmap='bwr', s=20)
axes[2].set_title('KPCA (polynomial d=3)')

axes[3].scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=y_circles, cmap='bwr', s=20)
axes[3].set_title('KPCA (RBF γ=5)')

plt.tight_layout()
plt.savefig('kpca_comparison.png', dpi=150)
plt.show()
```

### Kernel alignment : sélection d'hyperparamètres et comparaison de noyaux

Le kernel alignment peut être utilisé pour sélectionner automatiquement les hyperparamètres du noyau et pour comparer la qualité de différents noyaux sur un jeu de données donné.

```python
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel
from sklearn.datasets import make_moons

def kernel_target_alignment(K, y):
    """
    Calcule le Kernel Target Alignment (KTA) entre la matrice de Gram K
    et la matrice cible Y = yy^T.
    """
    Y_target = np.outer(y, y)
    num = np.sum(K * Y_target)
    den = np.sqrt(np.sum(K * K) * np.sum(Y_target * Y_target))
    return num / den

def optimize_gamma(X, y, gamma_range):
    """Seleccione le gamma optimal du noyau RBF par KTA."""
    best_gamma = gamma_range[0]
    best_alignment = -1
    for gamma in gamma_range:
        K = rbf_kernel(X, X, gamma=gamma)
        alignment = kernel_target_alignment(K, y)
        if alignment > best_alignment:
            best_alignment = alignment
            best_gamma = gamma
    return best_gamma, best_alignment

# Données : moons (non linéairement séparables)
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
y_moons = 2 * y_moons - 1

# Optimisation de gamma par KTA
gamma_range = np.logspace(-3, 3, 50)
best_gamma, best_alignment = optimize_gamma(X_moons, y_moons, gamma_range)
print(f"Gamma optimal (KTA) : {best_gamma:.4f}")
print(f"KTA optimal : {best_alignment:.4f}")

# Comparaison des noyaux
kernels_dict = {
    'RBF (optimal)': rbf_kernel(X_moons, X_moons, gamma=best_gamma),
    'RBF (γ=0.5)': rbf_kernel(X_moons, X_moons, gamma=0.5),
    'Polynomial (d=3)': polynomial_kernel(X_moons, X_moons, degree=3, coef0=1),
    'Laplacian (γ=1)': laplacian_kernel(X_moons, X_moons, gamma=1.0),
    'Linéaire': np.dot(X_moons, X_moons.T),
}

print("\nKernel Target Alignment comparé :")
for name, K in kernels_dict.items():
    alignment = kernel_target_alignment(K, y_moons)
    cond_number = np.linalg.cond(K)
    eigvals = np.linalg.eigvalsh(K)
    print(f"  {name:20s} : KTA={alignment:.4f}, cond={cond_number:.1f}, "
          f"eig_min={eigvals.min():.2e}, eig_max={eigvals.max():.2e}")
```

Le KTA est utilisé pour **sélectionner les hyperparamètres** du noyau (ex : $\gamma$ du RBF) sans avoir à ré-entraîner le SVM à chaque fois :

$$
\gamma^* = \arg\max_{\gamma} \hat{A}(K_\gamma, yy^T)
$$

## Théorie de la régularisation par noyau

La régularisation par noyau fournit un cadre unifié pour comprendre l'apprentissage dans les RKHS. Le problème général est :

$$
\min_{f \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \ell(y_i, f(x_i)) + \lambda \|f\|_\mathcal{H}^2
$$

Le **théorème de représentation** (Kimeldorf & Wahba, 1971) stipule que la solution optimale s'écrit comme une combinaison linéaire finie de noyaux évalués sur les données :

$$
f^*(x) = \sum_{i=1}^m \alpha_i \, k(x_i, x)
$$

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

### Correspondances détaillées entre noyaux classiques et quantiques

La transition des noyaux classiques aux noyaux quantiques s'opère en remplaçant la feature map classique $\Phi : \mathcal{X} \to \mathcal{H}$ par une feature map quantique $U_\phi : \mathcal{X} \to \mathcal{H}_Q$. Les correspondances clés sont :

| Concept classique | Analogue quantique | Description |
|---|---|---|
| Feature map $\Phi(x)$ | Circuit d'encodage $U_\phi(x)$ | Transformation de la donnée classique en vecteur (classique) ou état quantique (quantique) |
| Espace de redescription $\mathcal{H}$ | Espace de Hilbert $\mathcal{H}_Q = (\mathbb{C}^2)^{\otimes n}$ | Espace où le produit scalaire est calculé ; de dimension $\dim(\mathcal{H})$ (classique) ou $2^n$ (quantique) |
| Noyau $k(x, x') = \langle \Phi(x), \Phi(x')\rangle$ | Fidélité $k_Q(x, x') = |\langle \psi(x)|\psi(x')\rangle|^2$ | Produit scalaire dans l'espace de redescription ; classiquement calculé par kernel trick, quantiquement par mesure |
| Matrice de Gram $K_{ij} = k(x_i, x_j)$ | Matrice de fidélité quantique | Matrice $m \times m$ ; classiquement $O(m^2 d)$, quantiquement $O(m^2 \cdot n_{\text{shots}})$ |
| Kernel trick | Estimation de fidélité | Évite l'explicitation de $\Phi$ ; classiquement gratuit, quantiquement coûteux en mesures |
| Kernel alignment (KTA) | Quantum kernel alignment | Mesure d'adéquation entre noyau et cible ; mêmes formules, coût d'évaluation différent |
| Kernel PCA | Quantum PCA | Réduction de dimension non linéaire ; classique $O(m^3)$, quantique potentiellement $O(\text{polylog}(m))$ |
| Kernel ridge regression | Quantum kernel ridge | Régularisation dans le RKHS ; même formulation, évaluation du noyau différente |

```python
import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler

# --- Comparaison : noyau classique RBF vs noyau quantique ---

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

def quantum_feature_map(x, wires):
    """Encodage ZZ + entanglement."""
    for i in range(len(wires)):
        qml.Hadamard(wires=wires[i])
        qml.RZ(x[i % len(x)], wires=wires[i])
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])
        qml.RZ((x[i % len(x)] - x[(i+1) % len(x)])**2, wires=wires[i+1])
        qml.CNOT(wires=[wires[i], wires[i+1]])

@qml.qnode(dev)
def quantum_kernel_entry(x1, x2):
    """Calcul d'une entrée du noyau quantique via test de swap (estimation de fidélité)."""
    quantum_feature_map(x1, wires=range(n_qubits))
    qml.adjoint(quantum_feature_map)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

def compute_quantum_kernel(X1, X2):
    """Calcule la matrice de Gram quantique."""
    m1, m2 = X1.shape[0], X2.shape[0]
    K = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            probs = quantum_kernel_entry(X1[i], X2[j])
            K[i, j] = probs[0]  # |<psi(x_i)|psi(x_j)>|^2 = prob(|00...0>)
    return K

# --- Expérience comparative ---
X_moons, y_moons = make_moons(n_samples=50, noise=0.3, random_state=42)
X_moons = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_moons)

# Noyau classique RBF
from sklearn.metrics.pairwise import rbf_kernel
K_classical = rbf_kernel(X_moons, X_moons, gamma=0.5)

# Noyau quantique
K_quantum = compute_quantum_kernel(X_moons, X_moons)

# SVM pré-calculé avec chaque noyau
svm_classical = SVC(kernel='precomputed')
svm_classical.fit(K_classical, y_moons)
acc_classical = accuracy_score(y_moons, svm_classical.predict(K_classical))

svm_quantum = SVC(kernel='precomputed')
svm_quantum.fit(K_quantum, y_moons)
acc_quantum = accuracy_score(y_moons, svm_quantum.predict(K_quantum))

# Kernel Target Alignment
def kta(K, y):
    Y = np.outer(y, y)
    return np.sum(K * Y) / np.sqrt(np.sum(K * K) * np.sum(Y * Y))

print(f"Accuracy SVM RBF classique : {acc_classical:.4f}")
print(f"Accuracy SVM noyau quantique : {acc_quantum:.4f}")
print(f"KTA noyau classique : {kta(K_classical, y_moons):.4f}")
print(f"KTA noyau quantique : {kta(K_quantum, y_moons):.4f}")
```

### Problématiques spécifiques du passage classique → quantique

1. **Concentration du noyau** : Pour $n$ qubits avec un circuit expressif, $k_Q(x, x') \to 1/2^n$ uniformément, rendant la matrice de Gram indistinguable d'une matrice constante. Les noyaux classiques n'ont pas ce problème car leur espace de feature est mieux calibré.

2. **Coût d'évaluation** : Classiquement, $k(x, x')$ se calcule en $O(d)$. Quantiquement, chaque entrée coûte $O(1/\epsilon^2)$ mesures pour une précision $\epsilon$, soit un coût total $O(m^2/\epsilon^2)$ pour la matrice de Gram.

3. **Choix de la feature map** : Le noyau classique RBF a un unique hyperparamètre ($\gamma$). Le noyau quantique a de multiples degrés de liberté : type d'encodage, profondeur du circuit, portes d'intrication. Ce choix détermine la géométrie de l'espace de feature et la qualité du noyau.

4. **Régularisation** : Le paramètre $C$ du SVM contrôle la régularisation de la même manière dans les deux cas, mais l'effet de la *structure* du noyau est différent. Un noyau quantique mal structuré peut nécessiter une régularisation plus agressive.

### Tableau comparatif des noyaux

| Noyau | Formule | Dimension de $\mathcal{H}$ | Paramètres | Propriétés | Avantage quantique potentiel |
|---|---|---|---|---|---|
| Linéaire | $k(x,x') = x \cdot x'$ | $d$ | Aucun | Simple, interprétable | Faible (équivalent à un encodage sans intrication) |
| Polynomial | $k(x,x') = (x \cdot x' + c)^d$ | $\binom{d+p}{p}$ | $d, c$ | Capture interactions limitées | Modéré (feature map exponentielle possible) |
| RBF (gaussien) | $k(x,x') = \exp(-\gamma\|x-x'\|^2)$ | $\infty$ | $\gamma$ | Universel, lisse | Modéré (hard to beat sur données lisses) |
| Laplacien | $k(x,x') = \exp(-\gamma\|x-x'\|_1)$ | $\infty$ | $\gamma$ | Moins lisse, discontinu | Faible |
| Exponentiel | $k(x,x') = \exp(-\gamma\|x-x'\|)$ | $\infin$ | $\gamma$ | Adapté aux données ponctuelles | Faible |
| Matérn | $k(x,x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu}\frac{r}{\ell}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{r}{\ell}\right)$ | $\infty$ | $\nu, \ell, \sigma$ | Régularité paramétrable | Faible (très flexible classiquement) |
| Cosinus | $k(x,x') = \frac{x \cdot x'}{\|x\|\|x'\|}$ | $d$ | Aucun | Normalisé, invariant en norme | Modéré (similaire à encodage angle) |
| **Quantique (fidelity)** | $k_Q(x,x') = |\langle\psi(x)|\psi(x')\rangle|^2$ | $2^n$ | Circuit $U_\phi$, shots | Dépend du circuit | **Élevé** (si le circuit est bien choisi) |

## Références

- [MRT18] Mohri, M., Rostamizadeh, A. & Talwalkar, A. *Foundations of Machine Learning.* 2nd ed., MIT Press, 2018. Ch. 6–7.
- [Bis06] Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 6.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. Ch. 6.
- [Hav19] Havlíček, V. et al. « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019).
- [Agl26] Agliardi, G. et al. « Mitigating exponential concentration in covariant quantum kernels for subspace and real-world data. » *npj Quantum Information* 12, 12 (2026).
- [SS02] Schölkopf, B. & Smola, A. J. *Learning with Kernels.* MIT Press, 2002. Ch. 2–4.
- [CV95] Cortes, C. & Vapnik, V. « Support-vector networks. » *Machine Learning* 20(3), 273–297 (1995).
- [SSC00] Schölkopf, B., Smola, A. J. & Müller, K.-R. « Nonlinear component analysis as a kernel eigenvalue problem. » *Neural Computation* 10(5), 1299–1319 (1998).
