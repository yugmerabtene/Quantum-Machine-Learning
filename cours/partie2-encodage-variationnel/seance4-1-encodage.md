# Séance 4.1 — Encodage de données classiques en états quantiques

## 1. Introduction : pourquoi encoder ?

Un ordinateur quantique traite l'information dans l'espace de Hilbert $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$. Pour qu'un algorithme d'apprentissage automatique quantique (QML) puisse traiter des données classiques $\mathcal{X} \subseteq \mathbb{R}^d$, il faut une **feature map quantique** :

$$
\phi : \mathcal{X} \to \mathcal{H}, \quad \mathbf{x} \mapsto |\phi(\mathbf{x})\rangle
$$

Cette application transforme chaque point de donnée $\mathbf{x} \in \mathbb{R}^d$ en un état quantique $|\phi(\mathbf{x})\rangle$. Le choix de $\phi$ détermine :
- L'**expressivité** du modèle : quelle classe de fonctions peut-on représenter ?
- Les **ressources** nécessaires : nombre de qubits $n$, profondeur du circuit $D$
- La **trainabilité** : la variance du gradient et la concentration des kernels

---

## 2. Angle encoding

### 2.1 Principe

Chaque feature $x_i$ est encodée comme un angle de rotation d'un qubit. On applique des portes de rotation paramétrées :

$$
| \phi(\mathbf{x}) \rangle = \bigotimes_{i=1}^{d} R(x_i) |0\rangle^{\otimes d}
$$

où $R \in \{R_X, R_Y, R_Z\}$. Par exemple, avec $R_Y$ :

$$
R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}
$$

### 2.2 Implémentation PennyLane

```python
import pennylane as qml
import numpy as np

def angle_encoding_circuit(x):
    """Angle encoding avec R_Y."""
    for i, xi in enumerate(x):
        qml.RY(xi, wires=i)
    return qml.state()

dev = qml.device("default.qubit", wires=4)
circ = qml.QNode(angle_encoding_circuit, dev)

x = np.array([0.5, 1.2, -0.3, 2.1])
print(circ(x))  # état encodé
```

PennyLane fournit aussi `qml.AngleEmbedding` :

```python
qml.AngleEmbedding(x, wires=range(4), rotation="Y")
```

### 2.3 Propriétés

| Propriété | Valeur |
|-----------|--------|
| Qubits nécessaires | $n = d$ |
| Profondeur du circuit | $O(1)$ (parallélisable) |
| Espace des phases | $[0, 2\pi)^d$ |
| Non-linéarité | Introduite par la périodicité des rotations |

**Limitation** : l'espace des états accessibles est un tore $d$-dimensionnel dans $\mathcal{H}$, ce qui limite l'expressivité.

---

## 3. Amplitude encoding

### 3.1 Principe

On encode directement un vecteur $\mathbf{x} \in \mathbb{R}^N$ (normalisé) dans les amplitudes d'un état quantique à $n = \lceil \log_2 N \rceil$ qubits :

$$
|\phi(\mathbf{x})\rangle = \frac{1}{\|\mathbf{x}\|} \sum_{i=1}^{N} x_i |i\rangle
$$

où $|i\rangle$ est l'état de base $|b_1 b_2 \dots b_n\rangle$ correspondant à la représentation binaire de $i$.

### 3.2 Implémentation PennyLane

```python
def amplitude_encoding_circuit(x):
    qml.AmplitudeEmbedding(x, wires=range(3), normalize=True)
    return qml.state()

dev = qml.device("default.qubit", wires=3)
circ = qml.QNode(amplitude_encoding_circuit, dev)

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
print(circ(x))
```

### 3.3 Propriétés

| Propriété | Valeur |
|-----------|--------|
| Qubits nécessaires | $n = \lceil \log_2 d \rceil$ |
| Profondeur du circuit | $O(2^n)$ (général) |
| Avantage exponentiel | $2^n$ amplitudes avec $n$ qubits |
| Contrainte | Normalisation : $\|\mathbf{x}\| = 1$ |

**Avantage majeur** : nombre de qubits logarithmique en la dimension des données. C'est le seul encodage qui offre un véritable avantage en termes de compression.

**Inconvénient** : la préparation d'états arbitraires nécessite $O(2^n)$ portes, ce qui rend l'encodage coûteux pour des données de grande dimension.

---

## 4. Basis encoding

### 4.1 Principe

On représente chaque feature par sa représentation binaire. Pour une donnée $\mathbf{x} \in \mathbb{R}^d$, on convertit chaque $x_i$ en un mot binaire de $m$ bits de précision :

$$
|x_i\rangle = |b_{i,1} b_{i,2} \dots b_{i,m}\rangle
$$

L'état complet est :

$$
|\phi(\mathbf{x})\rangle = \bigotimes_{i=1}^{d} |x_i\rangle
$$

### 4.2 Implémentation Qiskit

```python
from qiskit import QuantumCircuit
import numpy as np

def basis_encoding(x, precision=4):
    n_qubits = len(x) * precision
    qc = QuantumCircuit(n_qubits)
    
    for i, xi in enumerate(x):
        # Conversion en entier sur `precision` bits
        xi_int = int((xi + 1) * (2**(precision-1)))
        binary = f"{xi_int:0{precision}b}"
        for j, bit in enumerate(binary):
            if bit == "1":
                qc.x(i * precision + j)
    return qc
```

### 4.3 Propriétés

| Propriété | Valeur |
|-----------|--------|
| Qubits nécessaires | $n = d \times m$ |
| Profondeur | $O(1)$ (portes X parallèles) |
| Précision | Limitée par $m$ |
| Type | Computationnel : états de base uniquement |

**Remarque** : cet encodage n'exploite pas la superposition — les états sont des états de base orthogonaux. Il est utile pour les algorithmes qui utilisent des requêtes oracle (Grover, etc.).

---

## 5. IQP encoding

### 5.1 Principe

L'**Instantaneous Quantum Polynomial** (IQP) encoding utilise des circuits de la classe IQP : portes diagonales dans la base computationnelle entrelacées de portes Hadamard.

$$
|\phi(\mathbf{x})\rangle = H^{\otimes n} \left( \prod_{i,j} e^{i x_i x_j Z_i Z_j} \right) H^{\otimes n} |0\rangle^{\otimes n}
$$

Cette construction a été proposée par Havlíček et al. (2019) pour les *quantum-enhanced feature spaces*.

### 5.2 Implémentation PennyLane

```python
def iqp_encoding_circuit(x):
    qml.IQPEmbedding(x, wires=range(4), n_repeats=2)
    return qml.state()

dev = qml.device("default.qubit", wires=4)
circ = qml.QNode(iqp_encoding_circuit, dev)

x = np.array([0.5, 1.2, -0.3, 2.1])
print(circ(x))
```

### 5.3 Implémentation Qiskit : ZZFeatureMap et PauliFeatureMap

```python
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap

n_qubits = 4
x = [0.5, 1.2, -0.3, 2.1]

# ZZFeatureMap : interactions à 2 corps Z⊗Z
zz_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
zz_circuit = zz_map.assign_parameters(x)

# PauliFeatureMap : interactions personnalisables
pauli_map = PauliFeatureMap(
    feature_dimension=n_qubits,
    paulis=["Z", "ZZ", "YY"],
    reps=2
)
```

### 5.4 Propriétés

| Propriété | Valeur |
|-----------|--------|
| Qubits | $n = d$ |
| Profondeur | $O(\text{reps} \times n^2)$ |
| Expressivité | Haute (interactions non-linéaires entre features) |
| Hardness | Difficulté classique à simuler (conjecture IQP) |

L'IQP encoding est au cœur des *quantum kernel methods* [Hav19] : la feature map génère un espace de Hilbert de grande dimension dans lequel le produit scalaire (fidélité) est difficile à calculer classiquement.

---

## 6. Hamiltonian encoding

### 6.1 Principe

On encode les données comme paramètres d'une évolution temporelle hamiltonienne :

$$
|\phi(\mathbf{x})\rangle = e^{-i H(\mathbf{x}) t} |\psi_0\rangle
$$

où $H(\mathbf{x}) = \sum_i x_i H_i$ avec $H_i$ des opérateurs hermitiens (généralement des produits de Pauli).

### 6.2 Exemple : évolution sous Hamiltonien

```python
def hamiltonian_encoding_circuit(x, time=1.0):
    # Hamiltonien : H = x0*Z0 + x1*Z1 + x0*x1*Z0Z1
    coeffs = [x[0], x[1], x[0] * x[1]]
    obs = [qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(1)]
    H = qml.Hamiltonian(coeffs, obs)
    qml.TrotterProduct(H, time=time, n=2)
    return qml.state()
```

### 6.3 Propriétés

| Propriété | Valeur |
|-----------|--------|
| Qubits | $n$ quelconque |
| Profondeur | Dépend de $H$ et de la méthode de Trotter |
| Expressivité | Très haute ($e^{-iH}$ explore tout $\mathcal{H}$) |
| Coût | Trotterisation $O(\text{poly}(n) \times \text{reps})$ |

---

## 7. Comparaison des méthodes d'encodage

### 7.1 Tableau récapitulatif

| Méthode | Qubits | Profondeur | Expressivité | Normalisation | Applications |
|---------|--------|------------|-------------|---------------|--------------|
| **Angle** | $d$ | $O(1)$ | Faible-moyenne | Non | VQC, classifieurs |
| **Amplitude** | $\lceil \log_2 d \rceil$ | $O(2^n)$ | Haute | Oui ($\|x\|=1$) | QSVM, VQE |
| **Basis** | $d \times m$ | $O(1)$ | Faible | Non | Oracles, algos exacts |
| **IQP** | $d$ | $O(\text{reps}\cdot n^2)$ | Haute | Non | Kernels quantiques |
| **Hamiltonian** | $d$ (ou moins) | $O(\text{Trotter})$ | Très haute | Non | Simulations, ADAPT |

### 7.2 Compromis fondamental

Il existe un compromis entre **économie de qubits** et **profondeur du circuit** :

$$
n_{\text{qubits}} \times D_{\text{profondeur}} \approx \text{ressources totales}
$$

L'amplitude encoding minimise $n$ mais maximise $D$, tandis que l'angle encoding fait l'inverse. Le choix dépend de la plateforme NISQ : les dispositifs actuels ont peu de qubits (50–156) mais des profondeurs limitées ($D \lesssim 100$).

### 7.3 Expressivité et capacité de généralisation

L'expressivité d'un encodage peut être mesurée par le **killing number** [SP21] :

$$
\mathcal{E}(\phi) = \dim(\text{span}\{ |\phi(\mathbf{x})\rangle : \mathbf{x} \in \mathcal{X} \})
$$

Pour l'angle encoding, $\mathcal{E} \leq 2^d$ (tore dans $\mathcal{H}$). Pour l'amplitude encoding, tout état de $\mathcal{H}$ est atteignable : $\mathcal{E} = 2^n$.

Un encodage trop expressif peut nuire à la généralisation (concentration exponentielle des kernels, voir séance 8.1).

---

## 8. Feature maps dans les frameworks

### 8.1 Qiskit

```python
from qiskit.circuit.library import (
    ZFeatureMap,    # RZ uniquement, pas d'interactions
    ZZFeatureMap,   # RZ + RZZ (interactions 2 corps)
    PauliFeatureMap # interactions personnalisables
)

# ZFeatureMap : encodage linéaire de base
zf = ZFeatureMap(feature_dimension=4, reps=1)

# ZZFeatureMap : encodage avec interactions (Havlíček 2019)
zzf = ZZFeatureMap(feature_dimension=4, reps=2)

# PauliFeatureMap : généralisation
pmf = PauliFeatureMap(
    feature_dimension=4,
    paulis=["Z", "ZZ", "ZXZ"],  # chaînes de Pauli
    reps=2
)
```

### 8.2 PennyLane

```python
# AngleEmbedding : angle encoding (équivalent ZFeatureMap)
qml.AngleEmbedding(x, wires=range(4), rotation="Z")

# AmplitudeEmbedding : amplitude encoding
qml.AmplitudeEmbedding(x, wires=range(3), normalize=True)

# IQPEmbedding : IQP encoding (équivalent ZZFeatureMap)
qml.IQPEmbedding(x, wires=range(4), n_repeats=2)
```

---

## 9. Exercices

1. **Comparaison Angle vs Amplitude** : Pour un vecteur $\mathbf{x} \in \mathbb{R}^{16}$, calculez le nombre de qubits et la profondeur minimale pour chaque méthode.
2. **IQP encoding** : Implémentez manuellement un circuit IQP à 2 qubits avec $x = [0.3, 0.7]$ et comparez avec `qml.IQPEmbedding`.
3. **Angle encoding périodique** : Montrez que l'angle encoding avec $R_Y$ ne peut pas distinguer $x$ et $x + 2\pi$. Quelle conséquence pour l'apprentissage ?

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. — Chapitre 4.
- [Hav19] Havlíček, V. et al. « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019).
- [NC00] Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.* Cambridge University Press, 2000.
- [QML25] Qiskit ML Contributors. « Qiskit Machine Learning: an open-source library for QML tasks at scale. » *arXiv:2505.17756*, 2025.
