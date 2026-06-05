# Séance 13.2 — Déploiement cloud et multi-plateforme pour QML

## Cloud quantique : les trois géants

L'accès au matériel quantique se fait principalement via trois fournisseurs cloud, chacun offrant des backends et des niveaux d'abstraction différents :

| Fournisseur | SDK principal | Backends disponibles | Paiement |
|-------------|--------------|---------------------|----------|
| **IBM Quantum** | Qiskit | IBM Condor, Heron, Flamingo | Crédits (∼0.1–1 $/shot) |
| **Amazon Braket** | Braket SDK | IonQ, Rigetti, QuEra, IQM | Temps d'exécution (∼0.3 $/s) |
| **Azure Quantum** | Q# / Cirq | IonQ, Quantinuum, Rigetti | Crédits |

Chaque plateforme expose un jeu de portes natif différent, ce qui rend la portabilité d'un circuit QML non triviale.

## Framework-agnostic QNN : abstraction multi-backend

Le travail fondateur de [Fra26] propose une architecture **framework-agnostic** pour les QNN (Quantum Neural Networks), où le même modèle QML peut être exécuté sur n'importe quel backend sans modification du code de haut niveau :

```
┌──────────────────────────────────────────────────┐
│              QNN Layer (abstraction)              │
├──────────────────────────────────────────────────┤
│  qml.RX(x[0]) → Forward → expval(PauliZ(0))      │
└──────────────────────┬───────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   Qiskit Backend  Braket Backend  Cirq Backend
   (IBM Heron)     (IonQ Forte-1)  (Simulateur)
```

Le noyau de l'architecture définit une **spécification de circuit intermédiaire** (IR) indépendante de tout SDK :

```python
class QNNLayer(torch.nn.Module):
    """Couche QNN framework-agnostic [Fra26]."""

    def __init__(self, n_qubits: int, n_layers: int, backend: str = "qiskit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits))
        self.backend = backend
        self.device = self._init_device(backend)

    def _init_device(self, backend: str):
        if backend == "pennylane":
            return qml.device("default.qubit", wires=self.n_qubits)
        elif backend == "qiskit":
            from qiskit_aer import AerSimulator
            return AerSimulator()
        elif backend == "braket":
            from braket.devices import LocalSimulator
            return LocalSimulator()
        # ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "pennylane":
            return self._forward_pennylane(x)
        elif self.backend == "qiskit":
            return self._forward_qiskit(x)
        # ...

    def _forward_pennylane(self, x):
        @qml.qnode(self.device)
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return torch.tensor(circuit(x, self.weights.detach().numpy()))
```

## Export ONNX pour circuits quantiques

La conversion entre frameworks quantiques est standardisée via **ONNX** (Open Neural Network Exchange), initialement conçu pour les réseaux de neurones classiques mais étendu aux circuits quantiques par [Fra26].

### Round-trip fidelity

La *round-trip fidelity* mesure la précision de la conversion quand un circuit subit une séquence complète : format A → format B → format A. Les résultats montrent une fidélité $> 0.9999$ :

| Conversion | Fidélité round-trip | Temps de transpilation |
|-----------|---------------------|-----------------------|
| Qiskit → Cirq → Qiskit | 0.99997 | 23 ms |
| PennyLane → Qiskit → PennyLane | 0.99993 | 18 ms |
| PennyLane → Braket → PennyLane | 0.99995 | 31 ms |
| Cirq → Braket → Cirq | 0.99991 | 27 ms |

### Implémentation ONNX-Q

```python
# OnnxQuantumCircuit : représentation intermédiaire universelle
from onnx_quantum import OnnxQuantumCircuit, convert

# Construction en représentation ONNX
circ = OnnxQuantumCircuit(n_qubits=4)
circ.rx(0, "x0")
circ.rx(1, "x1")
circ.cnot(0, 1)
circ.ry(2, "theta0")
circ.ry(3, "theta1")

# Export vers Qiskit
qiskit_circ = convert(circ, to="qiskit")
# Export vers Cirq
cirq_circ = convert(circ, to="cirq")
# Export vers PennyLane
pl_circ = convert(circ, to="pennylane")
# Export vers Braket (IonQ)
braket_circ = convert(circ, to="braket", target="ionq")

# Vérification unitaire : la sortie doit correspondre
from qiskit.quantum_info import Statevector
# ...
```

## Benchmark cross-platform

Le benchmark [Fra26] a exécuté le même circuit VQC sur 4 plateformes matérielles différentes et comparé les gradients obtenus :

| Plateforme | Technologie | Qubits utilisés | Gradient MAE vs. simulateur |
|------------|-------------|-----------------|----------------------------|
| IBM Heron (qiskit) | Supraconducteur | 6 | 0.0057 ± 0.0008 |
| Rigetti Aspen M3 (braket) | Supraconducteur | 6 | 0.0059 ± 0.0011 |
| IQM Garnet (braket) | Supraconducteur | 4 | 0.0060 ± 0.0013 |
| IonQ Forte-1 (braket) | Ions piégés | 6 | 0.0043 ± 0.0007 |

Le **gradient MAE ≤ 0.006** (Mean Absolute Error) par rapport au simulateur sans bruit confirme que les gradients sont reproductibles à travers les plateformes — un prérequis essentiel pour l'entraînement distribué de VQC.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def compute_gradient_mae(grad_platform, grad_simulator):
    """Mean Absolute Error entre gradients."""
    grad_p = np.array(grad_platform).flatten()
    grad_s = np.array(grad_simulator).flatten()
    return np.mean(np.abs(grad_p - grad_s))

# Données du benchmark [Fra26]
mae_scores = {
    "IBM Heron": 0.0057,
    "Rigetti Aspen": 0.0059,
    "IQM Garnet": 0.0060,
    "IonQ Forte-1": 0.0043,
}
print(f"MAE moyen : {np.mean(list(mae_scores.values())):.4f}")
```

```
MAE moyen : 0.0055
```

### Facteurs de variation

La variation entre plateformes s'explique par trois facteurs principaux :

1. **Bruit de mesure** : matrice de confusion $M$ différente par backend
2. **Fidélité des portes** : les portes natives diffèrent, induisant des chemins de transpilation distincts
3. **Taux d'échantillonnage** : les limites de shots diffèrent (Braket 100k, IBM 1M)

## Transpilation automatique

Le problème central du déploiement multi-plateforme est la **transpilation** : un circuit universel doit être converti dans le jeu de portes natif de chaque backend.

```python
# Transpilation automatique Qiskit
from qiskit import transpile

circuit_qml = QuantumCircuit(4)
circuit_qml.h(0)
circuit_qml.cx(0, 1)
circuit_qml.rz(0.5, 1)
circuit_qml.cx(1, 2)
circuit_qml.cx(2, 3)

# Transpilation pour IBM Heron (jeu : CX, ECR, RZ, X, SX)
transpiled_ibm = transpile(
    circuit_qml,
    backend=ibm_heron_backend,
    optimization_level=3,
    basis_gates=["cx", "ecr", "rz", "x", "sx"]
)

# Transpilation pour IonQ (jeu : MS, GPi, GPi2)
from braket.circuits import Circuit as BraketCircuit
braket_circ = BraketCircuit().h(0).cnot(0, 1).rz(1, 0.5).cnot(1, 2).cnot(2, 3)
# Braket transpile automatiquement pour le backend cible
```

L' **optimization_level=3** (le plus agressif) applique des règles de réécriture : commutation de portes, fusion de rotations, annihilation de portes inverses. Le nombre de portes peut être réduit de 30–50 % par rapport à une transpilation naïve.

## Conclusion pratique

Le déploiement multi-plateforme est désormais mature pour le QML : la fidélité round-trip $> 0.9999$ et le gradient MAE $\leq 0.006$ démontrent qu'un même VQC peut être entraîné sur simulateur et déployé sur plusieurs matériels sans perte significative de performance. L'abstraction framework-agnostic ouvre la voie à des *cloud quantum orchestrators* où la charge de travail QML est routée dynamiquement vers le backend le plus adapté.

## Références

- [Fra26] « A Framework-Agnostic Quantum Neural Network Architecture. » *arXiv:2604.04414*, Avril 2026.
- [QML25] Qiskit ML Contributors. « Qiskit Machine Learning: an open-source library for QML tasks at scale. » *arXiv:2505.17756*, 2025.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [Tra25] « Quantum transfer learning for image classification. » *arXiv:2603.16973*, 2025–2026.
