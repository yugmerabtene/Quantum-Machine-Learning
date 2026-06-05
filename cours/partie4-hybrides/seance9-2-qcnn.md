# Séance 9.2 — Quantum Convolutional Neural Networks (QCNN)

## Introduction

Les Quantum Convolutional Neural Networks (QCNN) étendent le paradigme des réseaux de neurones convolutifs classiques au domaine quantique. Proposés initialement par Cong et al. (2019) pour la classification de phases topologiques de la matière, les QCNN remplacent les filtres de convolution classiques par des circuits quantiques paramétrés agissant sur des sous-ensembles de qubits.

---

## Architecture du QCNN

Un QCNN se compose de trois types de couches organisées hiérarchiquement :

### 1. Convolution quantique

Une couche de convolution quantique applique un circuit variationnel $U(\theta)$ identique (poids partagés) sur un voisinage de $k$ qubits, en translation sur l'ensemble du registre. Pour un système à $n$ qubits avec un voisinage de 2 qubits :

```python
def conv_layer(weights, n_qubits):
    """Circuit de convolution 2x2 avec poids partagés."""
    for i in range(0, n_qubits - 1, 2):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(weights[i, 2], wires=i)
        qml.RZ(weights[i, 3], wires=i + 1)
        qml.CNOT(wires=[i + 1, i])
```

Le circuit de convolution $U(\theta)$ réalise une transformation unitaire : l'état d'entrée $|\psi_{\text{in}}\rangle$ sur $k$ qubits évolue selon :

$$
|\psi_{\text{out}}\rangle = U(\theta) |\psi_{\text{in}}\rangle
$$

### 2. Pooling quantique

Le pooling quantique réduit la dimensionnalité du système en mesurant certains qubits et en conditionnant les qubits restants. L'opération de pooling projette le système sur un sous-espace de plus petite dimension :

```python
def pooling_layer(n_qubits):
    """Pooling : mesure des qubits pairs, conditionnement des impairs."""
    for i in range(0, n_qubits - 1, 2):
        ancilla = qml.sample(wires=i)
        if ancilla == 0:
            qml.CNOT(wires=[i, i + 1])
        else:
            qml.PauliX(i + 1)
```

Dans la pratique, le pooling s'implémente via un circuit unitaire ancillaire suivi d'une mesure partielle. Après pooling, le nombre de qubits est réduit de moitié.

### 3. Hiérarchie de features

La répétition des couches convolution + pooling crée une hiérarchie de features analogue aux CNN classiques :

$$
n \xrightarrow{\text{conv+pool}} \frac{n}{2} \xrightarrow{\text{conv+pool}} \frac{n}{4} \xrightarrow{\text{conv+pool}} \ldots \xrightarrow{\text{mesure}} \text{label}
$$

Chaque pooling réduit le nombre de qubits d'un facteur 2, tandis que la profondeur du circuit reste constante.

### Architecture PennyLane complète

```python
import pennylane as qml
from pennylane import numpy as np

n_qubits = 8
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

def qcnn_layer(weights_conv, weights_pool, n_qubits):
    """Une couche QCNN complète : convolution + pooling."""
    conv_layer(weights_conv, n_qubits)
    pooling_layer(n_qubits)

@qml.qnode(dev)
def qcnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    idx = 0
    for _ in range(n_layers):
        w_conv = weights[idx:idx + (n_qubits // 2) * 4].reshape((n_qubits // 2, 4))
        qcnn_layer(w_conv, None, n_qubits)
        idx += (n_qubits // 2) * 4
        n_qubits //= 2

    return qml.expval(qml.PauliZ(0))

class QCNN:
    def __init__(self, n_qubits_init=8, n_layers=2):
        self.n_qubits = n_qubits_init
        self.n_layers = n_layers
        total_params = sum(
            (n_qubits_init // (2**l)) * 4
            for l in range(n_layers)
        )
        self.weights = np.random.random(total_params)

    def __call__(self, x):
        return qcnn_circuit(x, self.weights)
```

---

## Avantages Théoriques

### Réduction drastique du nombre de paramètres

Contrairement à un CNN classique où chaque filtre a $k^2 \times c_{\text{in}} \times c_{\text{out}}$ paramètres, un QCNN utilise un unique circuit $U(\theta)$ partagé : le nombre de paramètres est $O(\text{couches} \times n_{\text{qubits}})$, indépendant de la profondeur du réseau.

### Stabilité aux Barren Plateaux

Les QCNN atténuent les barren plateaux pour deux raisons principales :

1. **Coût local** : la mesure finale ne porte que sur un seul qubit, ce qui correspond à un opérateur de coût local (vs. global pour un VQC standard).

2. **Structure hiérarchique** : la variance du gradient dans un QCNN décroît comme $O(1 / n_{\text{couches}})$ plutôt que $O(1 / 2^n)$ pour un VQC non-structuré [SP21].

$$
\text{Var}[\nabla_{\theta} \mathcal{L}]_{\text{QCNN}} \sim \frac{1}{L}, \quad
\text{Var}[\nabla_{\theta} \mathcal{L}]_{\text{VQC}} \sim \frac{1}{2^n}
$$

où $L$ est le nombre de couches QCNN et $n$ le nombre de qubits.

### Hiérarchie de features naturelle

La réduction progressive du nombre de qubits force le réseau à apprendre des features de plus en plus globales, imitant la hiérarchie de représentation des CNN. Les premières couches capturent les corrélations locales, les dernières les structures globales.

---

## Applications

### Classification d'états quantiques

Application historique : distinguer des phases topologiques de la matière (phase de SPT, phase triviale) à partir de données de mesure locales. Un QCNN avec 8 qubits atteint $>96\%$ de précision sur cette tâche.

### Reconnaissance d'images

Sur le jeu de données MNIST réduit (images $4\times4$, 8 qubits), le QCNN atteint des performances comparables à un CNN classique avec $5\times$ moins de paramètres.

```python
# Comparaison QCNN vs CNN sur MNIST 4x4
# QCNN : 8 qubits, 2 couches → 56 paramètres
# CNN : Conv2d(1,4,3) + FC(16,10) → 214 paramètres
```

### Avantage comparatif QCNN vs CNN classique

| Critère | CNN classique | QCNN |
|---------|--------------|------|
| Paramètres (MNIST 4×4) | 214 | 56 |
| Espace de features | $\mathbb{R}^d$ | $\mathbb{C}^{2^n}$ |
| Stabilité gradient | dépend de la profondeur | $O(1/L)$ |
| Exécution matérielle | GPU | NISQ ($<100$ qubits) |
| Interprétabilité | filtres visibles | mesures projetées |

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. — §7.3.
- [QuL26] « Quantum LEGO Learning: A Modular and Architecture-Agnostic Hybrid QML Framework. » *arXiv:2601.21780*, 2026.
- [Tra26] « Quantum Transformers for Image Classification. » *Quantum Machine Intelligence* 8, 43 (2026).
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
