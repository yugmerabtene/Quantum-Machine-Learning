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

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. §2.1–2.3.
- [MRT18] Mohri, M., Rostamizadeh, A. & Talwalkar, A. *Foundations of Machine Learning.* 2nd ed., MIT Press, 2018. Ch. 6.
- [Bis06] Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 6.
- [NC00] Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.* Cambridge University Press, 2000. §2.
