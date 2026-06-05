# Séance 2.1 — Algèbre linéaire complexe et espaces de Hilbert à noyau reproduisant

## Espaces vectoriels complexes et notation de Dirac

L'apprentissage automatique quantique repose sur le formalisme des espaces de Hilbert complexes. Introduisons la **notation de Dirac** (bra-ket), standard en mécanique quantique et désormais centrale en théorie des noyaux.

Un **ket** $|\psi\rangle$ est un vecteur dans un espace de Hilbert $\mathcal{H}$ (espace vectoriel complet muni d'un produit scalaire). Le **bra** $\langle \phi|$ est la forme linéaire duale associée, soit le conjugate transposé : $\langle \phi| = |\phi\rangle^\dagger$. Le produit scalaire s'écrit :

$$
\langle \phi | \psi \rangle \in \mathbb{C}
$$

avec les propriétés de sesquilinéarité : $\langle \phi | a\psi + b\varphi \rangle = a\langle \phi|\psi\rangle + b\langle \phi|\varphi\rangle$ et $\langle a\phi + b\varphi | \psi \rangle = \bar{a}\langle\phi|\psi\rangle + \bar{b}\langle\varphi|\psi\rangle$.

La norme induite est $||\psi|| = \sqrt{\langle \psi | \psi \rangle}$.

## Produit tensoriel

Le **produit tensoriel** est l'opération fondamentale pour construire des espaces de grande dimension à partir de composantes plus petites. Soit $\mathcal{H}_A$ et $\mathcal{H}_B$ deux espaces de Hilbert. Leur produit tensoriel $\mathcal{H}_A \otimes \mathcal{H}_B$ est engendré par les états produits $|\psi\rangle_A \otimes |\phi\rangle_B$, noté $|\psi\rangle_A |\phi\rangle_B$ ou $|\psi \phi\rangle$.

Propriété clé : $\dim(\mathcal{H}_A \otimes \mathcal{H}_B) = \dim(\mathcal{H}_A) \times \dim(\mathcal{H}_B)$. Pour $n$ systèmes à 2 niveaux (qubits), $\dim(\mathcal{H}) = 2^n$.

En notation coordonnée, si $|\psi\rangle = \sum_i \alpha_i |i\rangle$ et $|\phi\rangle = \sum_j \beta_j |j\rangle$, alors :

$$
|\psi\rangle \otimes |\phi\rangle = \sum_{i,j} \alpha_i \beta_j |i\rangle \otimes |j\rangle
$$

```python
import numpy as np

def tensor_product(v1, v2):
    """Produit tensoriel de deux vecteurs."""
    return np.kron(v1, v2)

# Exemple : produit de deux qubits
zero = np.array([1, 0])   # |0⟩
one = np.array([0, 1])    # |1⟩
bell = tensor_product(zero, zero) + tensor_product(one, one)
bell = bell / np.linalg.norm(bell)
print(f"État de Bell (|00⟩ + |11⟩)/√2 : {bell}")
```

## RKHS — Reproducing Kernel Hilbert Space

Un **espace de Hilbert à noyau reproduisant** (RKHS) est un espace de Hilbert $\mathcal{H}$ de fonctions $f : \mathcal{X} \to \mathbb{R}$ tel que pour tout $x \in \mathcal{X}$, la fonctionnelle d'évaluation $\delta_x : f \mapsto f(x)$ est continue. Par le théorème de représentation de Riesz, il existe une fonction $k_x \in \mathcal{H}$ telle que :

$$
f(x) = \langle f, k_x \rangle_\mathcal{H}
$$

On définit alors le **noyau reproduisant** $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ par :

$$
k(x, x') = \langle k_x, k_{x'} \rangle_\mathcal{H}
$$

Propriété fondamentale : $k(x, \cdot) \in \mathcal{H}$ pour tout $x$, et $\langle f, k(x, \cdot) \rangle = f(x)$ — c'est la *propriété de reproduction*.

### Théorème de Mercer

Le théorème de Mercer établit le lien entre noyaux semi-définis positifs et RKHS. Une fonction symétrique $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ est un **noyau de Mercer** si pour tout ensemble fini $\{x_i\}$, la matrice de Gram $K_{ij} = k(x_i, x_j)$ est semi-définie positive. Alors il existe une base orthonormée $\{\phi_j\}$ de $L^2(\mathcal{X})$ et des valeurs $\lambda_j \geq 0$ telles que :

$$
k(x, x') = \sum_{j=1}^\infty \lambda_j \phi_j(x) \phi_j(x')
$$

et le RKHS associé est :

$$
\mathcal{H} = \left\{ f(x) = \sum_{j=1}^\infty a_j \sqrt{\lambda_j} \phi_j(x) \;:\; \sum_{j} a_j^2 < \infty \right\}
$$

#### Démonstration (schéma)

Soit $k$ un noyau de Mercer continu sur $\mathcal{X} \times \mathcal{X}$ (compact). L'opérateur intégral $T_k : L^2(\mathcal{X}) \to L^2(\mathcal{X})$ est défini par :

$$
(T_k f)(x) = \int_\mathcal{X} k(x, x') f(x') \, dx'
$$

Puisque $k$ est semi-défini positif, $T_k$ est un opérateur auto-adjoint positif et compact. Par le théorème spectral pour opérateurs compacts :

1. **Existence des valeurs propres** : Il existe une suite $(\lambda_j)_{j \geq 1}$ de valeurs propres positives décroissantes, avec $\lambda_j \to 0$, et une base orthonormée de fonctions propres $(\phi_j)$.
2. **Décomposition** : $k(x, x') = \sum_{j=1}^\infty \lambda_j \phi_j(x) \phi_j(x')$, la convergence étant uniforme et absolue sur $\mathcal{X} \times \mathcal{X}$.
3. **RKHS** : L'espace $\mathcal{H} = \{f = \sum_j a_j \sqrt{\lambda_j} \phi_j : \sum_j a_j^2 < \infty\}$ muni du produit scalaire $\langle f, g \rangle_\mathcal{H} = \sum_j a_j b_j$ (où $f = \sum_j a_j\sqrt{\lambda_j}\phi_j$, $g = \sum_j b_j\sqrt{\lambda_j}\phi_j$) est un RKHS dont le noyau reproduisant est $k$.

#### Exemple numérique : noyau RBF sur un intervalle

Pour $k(x, x') = e^{-|x - x'|^2 / (2\sigma^2)}$ sur $\mathcal{X} = [0, 1]$, les fonctions et valeurs propres de Mercer sont connues analytiquement (arrondies de la décomposition en série de Fourier) :

$$
\lambda_j = \sqrt{\frac{2}{\pi}} \cdot \frac{\sigma}{1 + (\sigma^2 \omega_j^2)} \cdot \exp\left(-\frac{\sigma^2 \omega_j^2}{2}\right)
$$

où $\omega_j$ sont les fréquences propres. En pratique, on vérifie numériquement que la matrice de Gram est bien SDP :

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def verify_mercer(X, sigma=1.0):
    """
    Vérifie les propriétés du noyau de Mercer :
    1. Symétrie de la matrice de Gram
    2. Semi-définie positivité
    3. Approximation de la fonction par valeurs propres
    """
    gamma = 1.0 / (2 * sigma**2)
    K = rbf_kernel(X.reshape(-1, 1), X.reshape(-1, 1), gamma=gamma)
    
    # 1. Symétrie
    is_symmetric = np.allclose(K, K.T)
    print(f"Symétrie : {is_symmetric}")
    
    # 2. Semi-définie positivité (valeurs propres >= 0)
    eigenvalues = np.linalg.eigvalsh(K)
    is_sdp = np.all(eigenvalues >= -1e-10)
    print(f"Semi-définie positive : {is_sdp}")
    print(f"Valeurs propres (top 5) : {eigenvalues[-5:][::-1]}")
    print(f"Valeurs propres (min) : {eigenvalues.min():.6e}")
    
    # 3. Décomposition de Mercer : k(x,x') = sum_j lambda_j phi_j(x) phi_j(x')
    eigenvalues_m, eigenfunctions = np.linalg.eigh(K)
    idx = eigenvalues_m.argsort()[::-1]
    eigenvalues_m = eigenvalues_m[idx]
    eigenfunctions = eigenfunctions[:, idx]
    
    # Approximation tronquée
    for n_components in [1, 3, 5, 10]:
        K_approx = (eigenfunctions[:, :n_components] * eigenvalues_m[:n_components]) @ eigenfunctions[:, :n_components].T
        error = np.linalg.norm(K - K_approx, 'fro') / np.linalg.norm(K, 'fro')
        print(f"Erreur relative approximation Mercer (n={n_components}) : {error:.4f}")
    
    return eigenvalues_m, eigenfunctions

# Vérification sur [0, 1] avec 50 points
X = np.linspace(0, 1, 50)
eigenvalues, eigenfunctions = verify_mercer(X, sigma=0.5)
```

### Feature map implicite

Un noyau $k$ définit implicitement une **feature map** $\Phi : \mathcal{X} \to \mathcal{H}$ telle que :

$$
k(x, x') = \langle \Phi(x), \Phi(x') \rangle_\mathcal{H}
$$

Cette feature map peut être de dimension infinie (ex : noyau RBF) ou finie (ex : noyau polynomial). L'avantage du *kernel trick* est qu'on peut calculer le produit scalaire dans $\mathcal{H}$ sans jamais expliciter $\Phi(x)$, en utilisant directement $k$.

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

X = np.array([[0.0], [1.0], [2.0]])
K = rbf_kernel(X, X, gamma=0.5)
print("Matrice de Gram (RBF) :")
print(K)
# Vérification semi-définie positivité
eigvals = np.linalg.eigvalsh(K)
print(f"Valeurs propres : {eigvals}")
```

### Visualisation de la feature map du noyau RBF

Le noyau RBF correspond à une feature map de dimension infinie. En utilisant la décomposition de Mercer, on peut approximer cette feature map par un nombre fini de composantes. Le code suivant visualise la feature map explicite :

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

def rbf_feature_map_approx(x, sigma, n_components=20):
    """
    Approximation de la feature map du noyau RBF via décomposition de Mercer.
    
    Pour le noyau RBF sur R, la feature map est :
    phi(x) = (sqrt(lambda_j) * phi_j(x))_j
    où lambda_j et phi_j sont les valeurs/fonctions propres de Mercer.
    
    On utilise l'approximation par série de Fourier sur [-L, L].
    """
    L = 3 * sigma  # Domaine tronqué
    freqs = np.arange(n_components) * np.pi / (2 * L)
    # Coefficients de la feature map
    coeffs = np.sqrt(np.sqrt(2 * np.pi) * sigma * 
                     np.exp(-2 * np.pi * sigma**2 * freqs**2))
    return coeffs * np.cos(freqs * (x + L))

def compute_and_visualize_kernel(sigma=1.0, n_points=100):
    """Calcule et visualise le noyau RBF et sa feature map."""
    x = np.linspace(-3, 3, n_points).reshape(-1, 1)
    
    # Matrice de Gram
    K = rbf_kernel(x, x, gamma=1/(2*sigma**2))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Visualisation de la matrice de Gram
    im = axes[0].imshow(K, extent=[-3, 3, -3, 3], cmap='viridis')
    axes[0].set_title(f'Matrice de Gram RBF (σ={sigma})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel("x'")
    plt.colorbar(im, ax=axes[0])
    
    # 2. Feature map approximée
    n_comp = 10
    features = np.array([rbf_feature_map_approx(xi[0], sigma, n_comp) for xi in x])
    for j in range(min(5, n_comp)):
        axes[1].plot(x, features[:, j], label=f'$\\phi_{j}$')
    axes[1].set_title('Composantes de la feature map')
    axes[1].set_xlabel('x')
    axes[1].legend()
    
    # 3. Valeurs propres de la décomposition de Mercer
    eigenvalues = np.linalg.eigvalsh(K)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    axes[2].semilogy(range(len(eigenvalues)), eigenvalues, 'bo-')
    axes[2].set_title('Valeurs propres (échelle log)')
    axes[2].set_xlabel('Indice j')
    axes[2].set_ylabel(r'$\lambda_j$')
    
    plt.tight_layout()
    plt.savefig('rbf_kernel_analysis.png', dpi=150)
    plt.show()
    
    # Vérification : k(x, x') = <phi(x), phi(x')>
    x_test = np.array([0.0, 1.0, 2.0]).reshape(-1, 1)
    K_true = rbf_kernel(x_test, x_test, gamma=1/(2*sigma**2))
    phi_test = np.array([rbf_feature_map_approx(xi[0], sigma, 20) for xi in x_test])
    K_approx = phi_test @ phi_test.T
    print(f"Erreur relative ||K_true - K_approx||/||K_true|| = {np.linalg.norm(K_true - K_approx) / np.linalg.norm(K_true):.4f}")
    
    return K, eigenvalues

K, eig = compute_and_visualize_kernel(sigma=1.0)
```

## Noyaux classiques fondamentaux

### Noyau RBF (Radial Basis Function)

Le noyau RBF (ou gaussien) est le plus utilisé :

$$
k_{\text{RBF}}(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)
$$

où $\sigma > 0$ est la largeur de bande. Il correspond à une feature map de dimension infinie. Son RKHS contient toutes les fonctions lisses (indéfiniment différentiables).

### Noyau polynomial

$$
k_{\text{poly}}(x, x') = (\langle x, x' \rangle + c)^d
$$

où $d \in \mathbb{N}$ est le degré et $c \geq 0$. La feature map contient tous les monômes de degré $\leq d$.

### Noyau laplacien

$$
k_{\text{lap}}(x, x') = \exp\left(-\frac{\|x - x'\|_1}{\sigma}\right)
$$

Moins lisse que le RBF, il est adapté aux données avec des discontinuités.

```python
def laplacian_kernel(X, Y, sigma=1.0):
    """Noyau laplacien."""
    from scipy.spatial.distance import cdist
    K = cdist(X, Y, metric='cityblock')
    return np.exp(-K / sigma)

X = np.array([[0.0], [1.0], [2.0]])
K_lap = laplacian_kernel(X, X, sigma=1.0)
print("Matrice de Gram (Laplacien) :")
print(K_lap)
```

## Lien avec les méthodes quantiques

La connexion avec le calcul quantique est directe : l'espace de Hilbert d'un système de $n$ qubits est $\mathcal{H}_Q = (\mathbb{C}^2)^{\otimes n}$, un espace de dimension $2^n$. Une feature map quantique $x \mapsto |\psi(x)\rangle$ plonge les données dans $\mathcal{H}_Q$, et le noyau quantique correspondant est :

$$
k_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2
$$

C'est un cas particulier de RKHS où $\mathcal{H}$ est l'espace des états quantiques. La différence cruciale avec les RKHS classiques est que $\mathcal{H}_Q$ est de dimension finie ($2^n$) mais exponentiellement grande, offrant un espace de représentation potentiellement très riche.

La théorie unifiée des RKHS et des noyaux quantiques sera approfondie en séance 7.1, où nous verrons que le *fidelity quantum kernel* peut être vu comme un noyau de Mercer particulier.

### Feature map quantique et RKHS

La connexion entre RKHS classique et espace de Hilbert quantique est profonde et mérite d'être détaillée. Considérons une feature map quantique :

$$
\Phi_Q : \mathcal{X} \to \mathcal{H}_Q, \quad x \mapsto |\psi(x)\rangle \in (\mathbb{C}^2)^{\otimes n}
$$

où $\mathcal{H}_Q$ est l'espace de Hilbert de $n$ qubits, de dimension $2^n$. Le noyau quantique associé est :

$$
k_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2
$$

**Propriétés clés de la correspondance RKHS classique ↔ quantique :**

1. **Dimension de l'espace de feature** : Un RKHS classique peut être de dimension infinie (RBF) ou finie (polynomial). L'espace quantique $\mathcal{H}_Q$ est de dimension finie $2^n$, mais exponentiellement grande.

2. **Noyau de Mercer** : $k_Q$ est bien un noyau de Mercer puisque la matrice de Gram $K_{ij} = |\langle \psi(x_i)|\psi(x_j)\rangle|^2$ est semi-définie positive (c'est la matrice de fidélité des états quantiques). Le théorème de Mercer s'applique donc.

3. **Feature map explicite vs implicite** :
   - Classiquement, le *kernel trick* évite d'expliciter $\Phi(x)$ car le calcul direct coûterait $O(\dim(\mathcal{H}))$.
   - Quantiquement, la feature map $\Phi_Q$ est *naturellement* accessible : l'état $|\psi(x)\rangle$ est préparé par le circuit quantique, et le produit scalaire est obtenu par mesure (estimation de fidélité).

4. **Encodage des données** : L'analogue quantique de la feature map classique est le circuit d'encodage. Par exemple, avec l'encodage par angles (*AngleEmbedding*) :

$$
|\psi(x)\rangle = \bigotimes_{j=1}^{n} R_Y(x_j) |0\rangle^{\otimes n}
$$

qui plonge $x \in \mathbb{R}^n$ dans $\mathbb{C}^{2^n}$. Avec des couches d'intrication supplémentaires, la feature map devient :

$$
|\psi(x)\rangle = U_{\text{ent}} \left[\bigotimes_{j=1}^{n} R_Y(x_j)\right] |0\rangle^{\otimes n}
$$

5. **Comparaison des dimensions de feature** :

| Noyau / Feature map | Dimension de $\mathcal{H}$ | Calcul de $k(x, x')$ |
|---|---|---|
| Linéaire | $d$ | $O(d)$ |
| Polynomial degré $p$ | $\binom{d+p}{p}$ | $O(d)$ (kernel trick) |
| RBF | $\infty$ | $O(d)$ (kernel trick) |
| Quantum ($n$ qubits) | $2^n$ | $O(n)$ qubits + mesures |

```python
import pennylane as qml
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    """Calcul du fidelity quantum kernel k_Q(x, x') = |<psi(x)|psi(x')>|^2."""
    # Encodage du premier état
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.HadamardLayers(wires=range(n_qubits))
    # Encodage inverse du second état (conjugé)
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    # Mesure de la probabilité de |0...0>
    return qml.probs(wires=range(n_qubits))

def quantum_kernel_matrix(X):
    """Calcule la matrice de Gram quantique."""
    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            probs = quantum_kernel(X[i], X[j])
            K[i, j] = probs[0]  # Probabilité de |0...0> = |<psi(x_i)|psi(x_j)>|^2
    return K

# Exemple sur données synthétiques
X = np.random.uniform(0, np.pi, size=(5, n_qubits))
K_Q = quantum_kernel_matrix(X)

print("Matrice de Gram quantique :")
print(np.round(K_Q, 4))

# Vérification : matrice SDP et diagonale = 1
print(f"\nDiagonale : {np.diag(K_Q)}")
eigenvalues = np.linalg.eigvalsh(K_Q)
print(f"Valeurs propres min : {eigenvalues.min():.6f}")
print(f"noyau de Mercer valide : {eigenvalues.min() >= -1e-10}")
```

## Exercices

### Exercice 1 : Produit tensoriel et état de GHZ

Démontrer que l'état de GHZ $|\text{GHZ}\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ ne peut pas être factorisé en un produit tensoriel d'états individuels (i.e., il est intriqué). Calculer la matrice de densité réduite $\rho_A = \text{Tr}_{BC}(|\text{GHZ}\rangle\langle\text{GHZ}|)$ et vérifier qu'elle est mixte.

### Exercice 2 : Vérification du théorème de Mercer

Soit le noyau $k(x, x') = (1 + x \cdot x')^2$ sur $\mathcal{X} = \mathbb{R}^2$.

1. Montrer que $k$ est un noyau de Mercer en explicitant la feature map $\Phi : \mathbb{R}^2 \to \mathbb{R}^6$.
2. Calculer la matrice de Gram pour les points $x_1 = (0, 0)$, $x_2 = (1, 0)$, $x_3 = (1, 1)$ et vérifier qu'elle est SDP.

### Exercice 3 : Noyau quantique et RKHS

On considère l'encodage $|\psi(x)\rangle = \cos(\frac{x}{2})|0\rangle + \sin(\frac{x}{2})|1\rangle$ pour un qubit.

1. Calculer le noyau quantique $k_Q(x, x') = |\langle\psi(x)|\psi(x')\rangle|^2$ et montrer que $k_Q(x, x') = \cos^2(\frac{x - x'}{2})$.
2. Vérifier que $k_Q$ est un noyau de Mercer en calculant la feature map explicite.
3. Comparer avec le noyau RBF : pour quelles valeurs de $\sigma$ les deux noyaux sont-ils similaires ?

### Exercice 4 : Complexité de Rademacher d'un RKHS

Soit $\mathcal{H}$ le RKHS associé au noyau $k$ avec $k(x, x) \leq \kappa^2$ pour tout $x$.

1. Montrer que la complexité de Rademacher empirique d'une classe de fonctions bornées dans $\mathcal{H}$ satisfait $\hat{\mathfrak{R}}_m(\mathcal{H}) \leq \frac{\kappa}{m}\sqrt{\text{Tr}(K)}$.
2. En déduire une borne de généralisation pour un SVM avec noyau RBF.

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. §2.1–2.3.
- [MRT18] Mohri, M., Rostamizadeh, A. & Talwalkar, A. *Foundations of Machine Learning.* 2nd ed., MIT Press, 2018. Ch. 6.
- [Bis06] Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 6.
- [NC00] Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.* Cambridge University Press, 2000. §2.
- [SS02] Schölkopf, B. & Smola, A. J. *Learning with Kernels.* MIT Press, 2002. Ch. 4.
- [PRW16] Petersen, L. R., Rabus, C. & Weickert, J. « Mercer kernels for computer vision. » *International Journal of Computer Vision* 117(2), 127–149 (2016).
- [Hav19] Havlíček, V. et al. « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019).
