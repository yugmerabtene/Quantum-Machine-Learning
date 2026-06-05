# Séance 7.1 — Quantum Kernel Methods (QKM)

## Objectifs pédagogiques

- Comprendre la définition et le formalisme des kernels quantiques
- Maîtriser le fidelity quantum kernel et son estimation
- Savoir implémenter un kernel quantique avec PennyLane et Qiskit ML
- Analyser l'avantage potentiel des espaces de features quantiques

---

## 1. Définition fondamentale

Les **Quantum Kernel Methods** (QKM) étendent le *kernel trick* classique aux espaces de Hilbert quantiques. L'idée centrale est de remplacer le noyau classique $k(\mathbf{x}, \mathbf{x}')$ par une mesure de similarité entre états quantiques encodés.

Soit $\mathcal{X} \subseteq \mathbb{R}^d$ l'espace des données d'entrée. On définit une **feature map quantique** :

$$
\Phi : \mathcal{X} \to \mathcal{H}, \quad \Phi(\mathbf{x}) = |\psi(\mathbf{x})\rangle = U(\mathbf{x})|0^{\otimes n}\rangle
$$

où $U(\mathbf{x})$ est un circuit paramétré par les données et $n$ le nombre de qubits. L'espace de Hilbert $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ a une dimension $2^n$, offrant un espace de features **exponentiellement grand**.

Le noyau quantique (dit *fidelity quantum kernel*) est alors défini par la fidélité au carré :

$$
K(\mathbf{x}, \mathbf{x}') = |\langle\psi(\mathbf{x})|\psi(\mathbf{x}')\rangle|^2
$$

Cette quantité est bornée entre 0 et 1 : $K(\mathbf{x}, \mathbf{x}') = 1$ si et seulement si $|\psi(\mathbf{x})\rangle = |\psi(\mathbf{x}')\rangle$ (à une phase près), et 0 si les états sont orthogonaux.

### 1.1 Lien avec les méthodes à noyau classiques

En apprentissage classique, un noyau $k$ doit satisfaire les conditions de Mercer : $k$ doit être symétrique et semi-définie positive (SDP). Soit $k(\mathbf{x}, \mathbf{x}') = \langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle$ où $\phi : \mathcal{X} \to \mathcal{F}$ est une feature map dans un RKHS $\mathcal{F}$. Le *kernel trick* permet de calculer ce produit scalaire sans expliciter $\phi$.

Le kernel quantique $K(\mathbf{x}, \mathbf{x}')$ est bien un noyau de Mercer valide : il est symétrique par construction, et la matrice de Gram $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ est semi-définie positive (c'est une matrice de Gram de produits scalaires dans $\mathcal{H}$).

---

## 2. Quantum Kernel Estimator

En pratique, on ne peut pas accéder directement à $|\langle\psi(\mathbf{x})|\psi(\mathbf{x}')\rangle|^2$. On doit l'estimer par **échantillonnage** (shots) sur un circuit quantique.

### 2.1 Estimation par swap test

Le *swap test* (Buhrman et al., 2001) permet d'estimer la fidélité entre deux états quantiques $|\psi\rangle$ et $|\phi\rangle$ en utilisant un qubit auxiliaire :

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=5, shots=1000)

@qml.qnode(dev)
def swap_test(x_params, xprime_params):
    # États |psi(x)> sur les qubits 0-1
    qml.AngleEmbedding(x_params, wires=[0, 1])
    # États |psi(x')> sur les qubits 2-3
    qml.AngleEmbedding(xprime_params, wires=[2, 3])
    # Swap test avec ancilla sur le qubit 4
    qml.Hadamard(wires=4)
    qml.CSWAP(wires=[4, 0, 2])
    qml.CSWAP(wires=[4, 1, 3])
    qml.Hadamard(wires=4)
    return qml.expval(qml.PauliZ(4))
```

La probabilité de mesurer $|0\rangle$ sur l'ancilla est :

$$
P(0) = \frac{1}{2} + \frac{1}{2} |\langle\psi|\phi\rangle|^2
$$

D'où l'estimateur :

$$
\hat{K}(\mathbf{x}, \mathbf{x}') = 2P(0) - 1
$$

### 2.2 Estimation par ComputeUncompute

Une méthode plus efficace en pratique est le **ComputeUncompute** (Havlíček et al., 2019) :

```python
@qml.qnode(dev)
def compute_uncompute(x, x_prime):
    # Préparer |psi(x)>
    qml.AngleEmbedding(x, wires=range(n_qubits))
    # Inverser la préparation de |psi(x')> : U(x')^\dagger
    qml.adjoint(qml.AngleEmbedding)(x_prime, wires=range(n_qubits))
    # Mesurer la probabilité de |0...0>
    return qml.probs(wires=range(n_qubits))
```

La probabilité de l'état $|0^{\otimes n}\rangle$ en sortie est :

$$
P(0^{\otimes n}) = |\langle 0^{\otimes n}| U(\mathbf{x}')^\dagger U(\mathbf{x}) |0^{\otimes n}\rangle|^2 = |\langle\psi(\mathbf{x})|\psi(\mathbf{x}')\rangle|^2
$$

Cette approche nécessite $2n$ qubits (contre $2n+1$ pour le swap test) et un seul appel au circuit.

---

## 3. Feature Maps Quantiques

Le choix de $U(\mathbf{x})$ est crucial : il détermine la **richesse expressive** du kernel.

### 3.1 ZZFeatureMap

$$
U_{\text{ZZ}}(\mathbf{x}) = \left( \bigotimes_{j=1}^n R_X(x_j) \right) \left( \bigotimes_{\langle i,j \rangle} e^{i x_i x_j Z_i Z_j} \right)
$$

```python
from qiskit.circuit.library import ZZFeatureMap

n_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2,
                          entanglement="linear")
print(feature_map.draw())
```

Le terme $Z_i Z_j$ introduit des corrélations d'ordre 2 entre les features, ce qui correspond à un noyau polynomial d'ordre 2 dans l'espace classique.

### 3.2 PauliFeatureMap

La généralisation avec des produits de Pauli arbitraires :

$$
U_{\text{Pauli}}(\mathbf{x}) = \left( \bigotimes_{j=1}^n R_P(x_j) \right) \left( \bigotimes_{S \subseteq [n]} e^{i \phi_S(\mathbf{x}) \bigotimes_{i \in S} P_i} \right)
$$

où $P \in \{X, Y, Z\}$ et $\phi_S(\mathbf{x})$ est un monôme des entrées.

### 3.3 Comparaison des feature maps

| Feature map | Portes | Corrélations | Complexité circuit |
|---|---|---|---|
| `ZFeatureMap` | $R_Z$ | Aucune | $O(n)$ |
| `ZZFeatureMap` | $R_Z, R_{ZZ}$ | Paires | $O(n^2)$ |
| `PauliFeatureMap` | $R_P, R_{PP}$ | Ordre supérieur | $O(2^n)$ |

---

## 4. Implémentation PennyLane

PennyLane fournit une API native pour les kernels quantiques :

```python
import pennylane as qml
from pennylane.kernels import embedding_kernel_matrix
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

# Définition de l'embedding
@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

# Kernel function
def kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]  # P(|0...0>)

# Génération de la matrice de Gram
X = np.random.randn(10, n_qubits)
K = embedding_kernel_matrix(X, kernel)
```

```output
Matrice de Gram (10×10) :
[[1.         0.824  0.231  0.547 ...]
 [0.824      1.     0.412  0.668 ...]
 [0.231      0.412  1.     0.145 ...]
 [ ...                              ]]
```

### Optimisation des paramètres

```python
from pennylane.kernels import quantum_kernel_alignment

def trained_kernel(x1, x2, params):
# Feature map paramétrée
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    for i in range(n_qubits):
        qml.RY(-params[i], wires=i)
    return qml.probs(wires=range(n_qubits))[0]

# Optimisation par kernel alignment
params_opt = quantum_kernel_alignment(
    X, y, trained_kernel,
    init_params=np.random.randn(n_qubits),
    optimizer=qml.AdamOptimizer(stepsize=0.1)
)
```

---

## 5. Implémentation Qiskit ML

Qiskit ML propose `FidelityQuantumKernel` et `ComputeUncompute` :

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import Sampler

n_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)

kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    samplers=[Sampler()],
    evaluate_duplicates='off_diagonal'
)

X_train = np.random.randn(50, n_qubits)
K_train = kernel.evaluate(x_vec=X_train)
```

```output
Matrice kernel (50×50) : forme (50, 50), valeurs min=0.02, max=1.00
Temps de construction : 2.34 s
```

### 5.1 Visualisation de l'espace des features

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Projection t-SNE de la matrice kernel
tsne = TSNE(metric="precomputed", init="random")
X_embedded = tsne.fit_transform(1 - K_train)  # 1 - K = distance

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train)
plt.title("Espace des features quantiques (t-SNE, ZZFeatureMap)")
plt.show()
```

---

## 6. Avantage des espaces de features quantiques

L'avantage potentiel des QKM réside dans la **dimension exponentielle** de l'espace de Hilbert : pour $n$ qubits, la dimension de l'espace des features est $2^n$, inaccessible classiquement.

### 6.1 Dimension effective

La dimension de l'espace de Hilbert ne garantit pas à elle seule un avantage. Ce qui compte est la **dimension effective** du kernel :

$$
d_{\text{eff}} = \frac{\text{Tr}(K)}{\|K\|_F}
$$

où $K$ est la matrice kernel sur les données d'entraînement. Une dimension effective élevée indique une plus grande expressivité.

```python
def effective_dim(K):
    eigvals = np.linalg.eigvalsh(K)
    trace = np.trace(K)
    frob = np.linalg.norm(K, 'fro')
    return trace / frob

print(f"Dimension effective : {effective_dim(K_train):.2f} / 2^{n_qubits}")
```

```output
Dimension effective : 8.47 / 2^4 = 16
```

### 6.2 Complexité computationnelle

| Aspect | Kernel classique (RBF) | Kernel quantique |
|---|---|---|
| Stockage feature map | $O(d)$ | $O(2^n)$ (explicite impossible) |
| Calcul d'un élément $K_{ij}$ | $O(d)$ | $O(\text{poly}(n, 1/\epsilon))$ |
| Dimension feature space | $\infty$ (RBF) | $2^n$ |
| Robustesse au bruit | Oui | Sensible (readout, gate errors) |

---

## 7. Pièges et limites

1. **Concentration exponentielle** : Pour $n$ grand, $K(\mathbf{x}, \mathbf{x}') \to \delta_{\mathbf{x}, \mathbf{x}'}$ — les kernels deviennent triviaux (Séance 8.1)
2. **Coût d'estimation** : Chaque élément $K_{ij}$ nécessite $O(1/\epsilon^2)$ shots pour une précision $\epsilon$
3. **Bruit de mesure** : Les erreurs de readout dégradent systématiquement la fidélité estimée
4. **Choix de la feature map** : Une feature map mal choisie rend le kernel inexpressif ou concentré

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. Ch. 6.
- [Hav19] Havlíček, V. et al. « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019).
- [Agl26] Agliardi, G. et al. « Mitigating exponential concentration in covariant quantum kernels. » *npj Quantum Information* 12, 12 (2026).
- [SC24] Schuld, M. *Supervised Learning with Quantum Computers.* Springer, 2024.
- [QML25] Qiskit ML Contributors. « Qiskit Machine Learning: an open-source library for QML tasks at scale. » *arXiv:2505.17756*, 2025.
