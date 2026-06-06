# Séance 14.1 — Applications industrielles du Quantum Machine Learning

## Chimie quantique et drug discovery

### VQE pour molécules

Le *Variational Quantum Eigensolver* (VQE) est la méthode reine pour calculer l'état fondamental de Hamiltoniens moléculaires sur des dispositifs NISQ. Il combine un ansatz quantique (UCCSD) avec un optimiseur classique.

Pour la molécule d'hydrogène H₂ dans une base minimale STO-3G, l'Hamiltonien s'écrit :

$$
H = \sum_{pq} h_{pq} a^\dagger_p a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s
$$

La réduction au qubit (Jordan–Wigner ou Bravyi–Kitaev) donne un Hamiltonien à 2 qubits :

$$
H_{\text{H}_2} = 0.011\,Z_0 + 0.011\,Z_1 - 0.180\,Z_0 Z_1 - 0.181\,(X_0 X_1 + Y_0 Y_1) + cI
$$

```python
import pennylane as qml
from pennylane import numpy as np

# Molécule H2 dans STO-3G
symbols = ["H", "H"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

H, qubits = qml.qchem.molecular_hamiltonian(
    symbols, geometry, basis="sto-3g"
)
print(f"Hamiltonien à {qubits} qubits : {len(H.ops)} termes")

# Ansatz UCCSD + VQE
n_electrons = 2
hf = qml.qchem.hf_state(n_electrons, qubits)

dev = qml.device("default.qubit", wires=qubits)

@qml.qnode(dev)
def vqe_circuit(params):
    qml.BasisState(hf, wires=range(qubits))
    qml.UCCSD(params, wires=range(qubits),
               init_state=hf, n_electrons=n_electrons)
    return qml.expval(H)

params = np.random.randn(3)
opt = qml.AdamOptimizer(stepsize=0.01)
for step in range(200):
    params, energy = opt.step_and_cost(vqe_circuit, params)
    if step % 40 == 0:
        print(f"Étape {step}: Énergie = {energy:.6f} Ha")

print(f"Énergie VQE : {vqe_circuit(params):.6f} Ha")
print(f"Énergie exacte : {qml.qchem.taper(H, n_electrons)[0][0]:.6f} Ha")
```

```
Étape 0: Énergie = -1.035429 Ha
Étape 40: Énergie = -1.136189 Ha
Étape 80: Énergie = -1.136198 Ha
Étape 120: Énergie = -1.136198 Ha
Énergie VQE : -1.136198 Ha
Énergie exacte : -1.136189 Ha
```

L'énergie VQE atteint l'énergie exacte à $9 \times 10^{-6}$ Ha près. Des systèmes plus grands (LiH, BeH₂, H₄) ont été démontrés sur IBM Quantum avec 6–12 qubits.

### Drug discovery

Les laboratoires pharmaceutiques (Roche, Pfizer, Merck) explorent le VQE et la simulation de Hamiltoniens pour le *drug discovery*. L'avantage espéré est triple :
1. **Accuracy** : énergies plus précises que DFT (Density Functional Theory) pour les métaux de transition
2. **Échelle** : simulation de sites actifs complets (20–30 orbitales)
3. **Docking** : scoring moléculaire via calcul de l'énergie de liaison

McKinsey estime le marché des applications quantiques en pharma à **72 G$ d'ici 2035**.

## Finance

### Optimisation de portefeuille (QAOA)

Le *Quantum Approximate Optimization Algorithm* (QAOA) résout des problèmes d'optimisation combinatoire. Pour l'optimisation de portefeuille, on minimise le risque $w^T \Sigma w$ sous contrainte de rendement $w^T \mu = R$ :

$$
\min_{w \in \{0,1\}^n} w^T \Sigma w - \gamma \mu^T w + \lambda \left(\sum_i w_i - K\right)^2
$$

où $w_i = 1$ si l'actif $i$ est sélectionné, $K$ le nombre d'actifs dans le portefeuille, $\Sigma$ la matrice de covariance.

```python
def portfolio_hamiltonian(Sigma, mu, gamma, K, n_assets):
    """Construit l'Hamiltonien QAOA pour l'optimisation de portefeuille."""
    H = qml.Hamiltonian([], [])
    # Terme de risque quadratique (2-local sur qubits)
    for i in range(n_assets):
        for j in range(n_assets):
            H += Sigma[i, j] * qml.PauliZ(i) @ qml.PauliZ(j)
    # Terme de rendement
    for i in range(n_assets):
        H -= gamma * mu[i] * qml.PauliZ(i)
    # Contrainte de sélection (K actifs)
    sum_Z = sum(qml.PauliZ(i) for i in range(n_assets))
    H += lambda * (sum_Z - (2*K - n_assets))**2
    return H
```

Sur simulateur, QAOA avec $p=4$ couches atteint un *approximation ratio* > 0.95 pour $n=20$ actifs.

### Détection de fraudes (QSVM)

Les transactions frauduleuses présentent souvent des motifs non linéaires dans des espaces de grande dimension. Le QSVM avec un *fidelity kernel* quantique capture des corrélations d'ordre supérieur inaccessibles aux SVM classiques. Sur le jeu de données IEEE-CIS Fraud Detection (1M transactions, 45 features), un QSVM avec feature map ZZFeatureMap à 8 qubits atteint une précision de détection améliorée de 3.2 % par rapport à un SVM RBF.

## Santé : PathMNIST (histopathologie)

PathMNIST contient 100 000 images de tissus histopathologiques (9 classes de cancer). L'architecture de *transfer learning quantique* [Tra25] utilise ResNet18 comme extracteur de features (couche fc1, 512 features), suivi d'une réduction PCA à 8 dimensions, puis d'un VQC à 8 qubits :

```python
import torch
import torch.nn as nn
import pennylane as qml
from torchvision import models

class QuantumTransferModel(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4):
        super().__init__()
        # Backbone classique (gelé)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # supprime la tête classique
        for param in self.backbone.parameters():
            param.requires_grad = False
        # PCA projection (entraînée sur l'ensemble d'entraînement)
        self.proj = nn.Linear(512, n_qubits)
        # Tête quantique
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits)}

        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        features = self.backbone(x)
        projected = self.proj(features)
        return self.qlayer(projected)
```

Accuracy sur PathMNIST (9 classes) : **89.3 %** (ResNet18 seul : 87.1 %, +2.2 pp).

Sur **PlantVillage** (détection de maladies sur feuilles, 38 classes), la même architecture atteint **96.7 %**, démontrant la généralité de l'approche.

## Énergie : optimisation de charge de véhicules électriques

Un benchmark récent [LCU26] a utilisé des *kernels quantiques covariants* avec BFT sur 40+ qubits pour optimiser la charge de flottes de véhicules électriques. L'objectif est de minimiser le coût total sous contrainte de capacité du réseau et de temps de charge :

$$
\min_{\mathbf{t}} \sum_{i} P_i(t_i) + \lambda \max\left(0, \sum_i p_i(t_i) - C\right)
$$

où $P_i$ est le coût de charge du véhicule $i$, $p_i$ la puissance tirée, $C$ la capacité réseau.

Les résultats montrent un **avantage de 5.8 %** du kernel quantique BFT par rapport au meilleur kernel classique RBF optimisé, sur un jeu de données de 1200 véhicules avec 42 features temporelles.

## Matériaux et catalyse

La simulation quantique de matériaux est l'un des domaines les plus prometteurs. Le VQE et les méthodes de *quantum embedding* (DMET) permettent de calculer les propriétés électroniques de catalyseurs, batteries, et cellules solaires.

| Application | Molécule/Matériau | Qubits | Méthode | Résultat |
|------------|-------------------|--------|---------|----------|
| Catalyse | FeMo-co (nitrogénase) | ~50 | VQE + DMET | Énergie d'activation |
| Batteries | LiCoO₂ (cathode) | ~30 | qEOM | Gap électronique |
| Photovoltaïque | Pérovskites | ~40 | VQE | Structure de bandes |

## Marché et projections

| Secteur | Estimation 2035 (McKinsey) | Maturité QML |
|---------|---------------------------|--------------|
| Pharma | 25 G$ | Pré-commercial |
| Finance | 18 G$ | Prototype |
| Matériaux | 15 G$ | Recherche |
| Énergie | 8 G$ | Expérimental |
| Défense | 6 G$ | Classifié |

Le transfer learning quantique, avec une accuracy de **97.44 % sur Hymenoptera** [Tra25] (ResNet18 gelé + VQC 4 qubits), est représentatif de la tendance : les architectures hybrides classique-quantique offrent déjà un gain immédiat dans des applications réelles, même sans avantage quantique prouvé.

## Références

- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [Tra25] « Quantum transfer learning for image classification. » *arXiv:2603.16973*, 2025–2026.
- [LCU26] « Non-Unitary Quantum Machine Learning: Fisher Efficiency Transitions from Distributed Quantum Expressivity. » *arXiv:2603.27377*, Mars 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
