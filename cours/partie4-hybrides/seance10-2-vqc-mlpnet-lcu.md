# Séance 10.2 — VQC-MLPNet et Architectures Non-Unitaires

## VQC-MLPNet : VQC Générant les Poids d'un MLP

### Principe

Proposé dans [VQC26], VQC-MLPNet introduit une architecture hybride unconventionalle où un circuit quantique variationnel génère les **poids d'un MLP classique**. Contrairement aux approches hybrides standard (où le VQC est une couche de décision), ici le VQC joue le rôle de *générateur de paramètres*.

**Architecture :**

$$
W_{\text{MLP}} = \mathcal{T}\left( \text{Tr}_{\text{partial}}\left[ |\psi(\theta)\rangle\langle\psi(\theta)| \cdot O \right] \right)
$$

où $|\psi(\theta)\rangle$ est l'état produit par le VQC, $O$ un observable, $\text{Tr}_{\text{partial}}$ la trace partielle et $\mathcal{T}$ un post-traitement (reshaping + normalisation).

### Implémentation

```python
import pennylane as qml
import torch

n_qubits = 6  # 2^n_qubits = 64 paramètres
n_params = 64  # poids du MLP (4×16)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def generate_mlp_weights(theta):
    """Génère les poids du MLP via amplitude encoding."""
    qml.AngleEmbedding(theta[:n_qubits], wires=range(n_qubits))
    qml.BasicEntanglerLayers(
        theta[n_qubits:].reshape((2, n_qubits)),
        wires=range(n_qubits)
    )
    # Mesure dans la base de Pauli : 2^n_qubits valeurs
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class VQC_MLPNet(torch.nn.Module):
    def __init__(self, vqc_params=6, mlp_hidden=16):
        super().__init__()
        self.vqc_params = torch.nn.Parameter(torch.randn(vqc_params + 2*vqc_params))
        self.fc1 = torch.nn.Linear(4, mlp_hidden, bias=False)
        self.fc2 = torch.nn.Linear(mlp_hidden, 2, bias=False)

    def forward(self, x):
        # Le VQC génère les poids de fc1 en une seule passe
        weights_flat = generate_mlp_weights(self.vqc_params)
        W = weights_flat.reshape(4, 16).to(x.device)
        h = torch.relu(torch.mm(x, W))
        return self.fc2(h)
```

Les poids $W$ sont donc *échantillonnés* depuis l'espace de Hilbert à chaque forward, plutôt qu'appris directement par rétropropagation.

---

## Analyse NTK (Neural Tangent Kernel)

### Bornes supérieures d'erreur

L'analyse via le NTK [VQC26] établit une borne supérieure sur l'erreur de généralisation de VQC-MLPNet :

$$
\mathcal{R}(f_{\text{VQC-MLPNet}}) \leq \sqrt{\frac{\mathcal{Y}^T (K_{\text{NTK}})^{-1} \mathcal{Y}}{n}} + O\left(\frac{1}{\sqrt{n}}\right)
$$

où $K_{\text{NTK}}$ est le Neural Tangent Kernel du modèle hybride, et $\mathcal{Y}$ le vecteur des labels.

**Résultat clé :** le conditionnement de $K_{\text{NTK}}$ pour VQC-MLPNet est strictement meilleur que celui d'un MLP standard ou d'un VQC seul :

$$
\kappa(K_{\text{NTK}}^{(hybride)}) \leq \min\left(\kappa(K_{\text{NTK}}^{(MLP)}), \kappa(K_{\text{NTK}}^{(VQC)})\right)
$$

Cette amélioration provient de la **combinaison non-triviale** des espaces de représentation classique et quantique.

---

## Non-Unitaire QML via LCU

### Principe de la Linear Combination of Unitaries

Les circuits quantiques standards sont unitaires ($U^\dagger U = I$), ce qui limite l'expressivité des modèles. La méthode **Linear Combination of Unitaries (LCU)** permet d'implémenter des opérations non-unitaires :

$$
U_{\text{LCU}} = \sum_{i=1}^k \alpha_i U_i, \quad \sum_i \alpha_i = 1
$$

où $U_i$ sont des portes unitaires et $\alpha_i$ des coefficients réels.

### Implémentation avec ancilla et post-sélection

L'implémentation de LCU nécessite un registre d'ancilla et une post-sélection :

```python
def lcu_layer(alpha, unitaries, n_data_qubits):
    """Couche LCU : combinaison linéaire de 2 unitaires."""
    n_ancilla = 1
    total_qubits = n_data_qubits + n_ancilla

    # Préparation de l'ancilla dans l'état |+>
    qml.Hadamard(wires=total_qubits - 1)

    # Unitaires contrôlés
    for i, U in enumerate(unitaries):
        qml.ctrl(U, control=total_qubits - 1)(
            wires=range(n_data_qubits),
            control_values=[i]
        )

    # Post-sélection : mesurer l'ancilla et ne garder que |0>
    result = qml.sample(wires=total_qubits - 1)
    # Le reste du circuit n'est valide que si result == 0
```

**Formalisme :** L'opération non-unitaire effective est obtenue par :

$$
\rho_{\text{out}} = \frac{\text{Tr}_{\text{ancilla}}\left[ P_0 U_{\text{LCU}} (\rho_{\text{in}} \otimes |0\rangle\langle0|) U_{\text{LCU}}^\dagger P_0 \right]}{\text{Pr}(\text{succès})}
$$

où $P_0 = |0\rangle\langle0|_{\text{ancilla}}$ est le projecteur sur l'état $|0\rangle$ de l'ancilla, et la probabilité de succès est :

$$
p_{\text{succès}} = \text{Tr}\left[ P_0 U_{\text{LCU}} (\rho_{\text{in}} \otimes |0\rangle\langle0|) U_{\text{LCU}}^\dagger \right]
$$

---

## Fisher Efficiency Transition

### Résultats [LCU26]

L'étude systématique [LCU26] sur plus de 570 expériences révèle une **transition de phase** dans l'efficacité de Fisher des modèles LCU :

$$
F_{\text{LCU}}(n) \begin{cases}
\sim O(1) & \text{pour } n < 10 \\
\sim O(2^{n}) & \text{pour } n \geq 12
\end{cases}
$$

Cette transition à **10–12 qubits** indique un seuil à partir duquel les modèles non-unitaires deviennent exponentiellement plus expressifs que leurs contreparties unitaires.

### Gains expérimentaux

| Domaine | Jeu de données | Gain LCU vs. unitaire |
|---------|---------------|---------------------|
| Vision | MNIST | +2.1% |
| Agriculture | PlantVillage | +5.8% |
| Chimie | QM9 | +0.2% |
| Médecine | PathMNIST | +3.4% |

```python
# Comparaison LCU vs. unitaire sur MNIST
resultats = {
    "VQC unitaire (8 qubits)": 0.934,
    "VQC-MLPNet (8 qubits)": 0.948,
    "LCU-VQC (8 qubits, α=[0.7,0.3])": 0.955,
}
```

### Interprétation théorique

La transition Fisher s'explique par la **distributed quantum expressivity** : au-delà d'un seuil de qubits, les combinaisons linéaires d'unitaires explorent des régions de l'espace de Hilbert inaccessibles aux circuits unitaires seuls. L'information de Fisher quantique $F_Q$ mesure cette capacité :

$$
F_Q(\rho_\theta) = \text{Tr}\left[ \rho_\theta L_\theta^2 \right], \quad
\nabla_\theta \rho_\theta = \frac{1}{2} (\rho_\theta L_\theta + L_\theta \rho_\theta)
$$

où $L_\theta$ est le log derivative symétrique.

---

## Références

- [VQC26] « VQC-MLPNet: Unconventional Hybrid Quantum-Classical Architecture. » *arXiv:2506.10275*, 2026.
- [LCU26] « Non-Unitary Quantum Machine Learning: Fisher Efficiency Transitions from Distributed Quantum Expressivity. » *arXiv:2603.27377*, Mars 2026.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
