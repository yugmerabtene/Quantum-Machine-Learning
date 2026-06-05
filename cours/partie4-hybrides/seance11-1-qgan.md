# Séance 11.1 — Quantum Generative Models

## Introduction

Les modèles génératifs quantiques étendent au domaine quantique la capacité d'apprendre des distributions de probabilité sous-jacentes à des données. Ils exploitent la mesure de circuits quantiques paramétrés (PQC) pour modéliser, échantillonner et générer des données. Cette séance couvre trois familles principales : les qGAN, les Born Machines, et les Quantum Boltzmann Machines.

---

## Quantum Generative Adversarial Networks (qGAN)

### Architecture

Un qGAN [SP21, QML25] remplace le générateur classique par un circuit quantique paramétré (PQC) :

$$
G_\theta(z) = \text{mesure}\left( U_\theta |\psi_0\rangle \right), \quad z \sim p_{\text{latent}}(z)
$$

Le discriminateur $D_\phi$ reste classique (réseau de neurones). La fonction objectif est la même que pour un GAN classique :

$$
\min_\theta \max_\phi \mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p_{\text{latent}}}[\log(1 - D_\phi(G_\theta(z)))]
$$

```python
import pennylane as qml
import torch

n_qubits = 4
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_generator(weights):
    """Générateur quantique : PQC transformant |0...0> en échantillon."""
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    # Mesure : échantillon de 4 bits (1 bit par qubit)
    return [qml.sample(wires=i) for i in range(n_qubits)]

class QuantumGenerator(torch.nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

    def forward(self, batch_size):
        samples = []
        for _ in range(batch_size):
            s = quantum_generator(self.weights)
            samples.append(s)
        return torch.tensor(samples, dtype=torch.float32)

class ClassicalDiscriminator(torch.nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

### Entraînement

L'entraînement alterne la mise à jour du discriminateur et du générateur :

```python
def train_qgan(generator, discriminator, real_data, epochs=100):
    opt_G = torch.optim.Adam(generator.parameters(), lr=0.01)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        # Entraînement du discriminateur
        real = real_data
        fake = generator(real.shape[0])
        d_loss = loss_fn(discriminator(real), torch.ones_like(real[:,0:1]))
        d_loss += loss_fn(discriminator(fake.detach()), torch.zeros_like(fake[:,0:1]))
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # Entraînement du générateur
        fake = generator(real.shape[0])
        g_loss = loss_fn(discriminator(fake), torch.ones_like(fake[:,0:1]))
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()
```

---

## Circuit Born Machine

### Principe

Une **Circuit Born Machine** est un modèle génératif quantique où la distribution de probabilité sur les chaînes de bits $x \in \{0,1\}^n$ est donnée par la règle de Born :

$$
p_\theta(x) = |\langle x | U_\theta | 0^n \rangle|^2
$$

L'objectif est d'approcher une distribution cible $p_{\text{data}}(x)$ en minimisant une divergence (KL, Jensen-Shannon) :

$$
\theta^* = \arg\min_\theta D_{\text{KL}}(p_{\text{data}} \parallel p_\theta)
$$

```python
def born_machine_circuit(theta, n_qubits):
    """Circuit paramétré produisant une distribution de probabilité."""
    for i in range(n_qubits):
        qml.RY(theta[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RZ(theta[n_qubits + i], wires=i)
    return qml.probs(wires=range(n_qubits))

# Distribution cible : nombres pairs en binaire
target = torch.tensor([0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0])

@qml.qnode(dev)
def born_loss(theta):
    probs = born_machine_circuit(theta, 3)
    return -torch.sum(target * torch.log(probs))  # KL divergence
```

---

## Quantum Boltzmann Machines

### Distribution thermique

Une **Quantum Boltzmann Machine** (QBM) modélise une distribution à partir de l'Hamiltonien quantique du système :

$$
p_\theta(x) = \frac{\langle x | e^{-\beta H_\theta} | x \rangle}{\text{Tr}[e^{-\beta H_\theta}]}
$$

où $H_\theta = \sum_i \theta_i h_i$ est un Hamiltonien paramétré et $\beta = 1/(k_B T)$ la température inverse.

L'Hamiltonien typique inclut des termes locaux et d'interaction :

$$
H_\theta = \sum_{i} a_i \sigma_i^z + \sum_{i<j} b_{ij} \sigma_i^z \sigma_j^z + \sum_{i} c_i \sigma_i^x
```

L'entraînement consiste à minimiser la log-vraisemblance négative :

$$
\mathcal{L}(\theta) = -\sum_{x \in \mathcal{D}} \log p_\theta(x)
$$

dont le gradient est :

$$
\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{p_{\text{data}}}[\nabla_\theta E_\theta(x)] - \mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x)]
$$

où $E_\theta(x) = \langle x | H_\theta | x \rangle$ est l'énergie du système.

---

## Comparaison qGAN vs. GAN Classique

| Critère | GAN classique | qGAN |
|---------|--------------|------|
| Générateur | MLP / CNN | PQC sur $n$ qubits |
| Espace latent | $\mathbb{R}^d$ (continu) | $\mathbb{C}^{2^n}$ (Hilbert) |
| Échantillonnage | forward pass | mesure projective |
| Expressivité | limitée par la profondeur | $2^n$ amplitudes |
| Entraînement | backprop standard | parameter-shift |
| Stabilité | mode collapse | atténué ? (rech. en cours) |
| Matériel | GPU | NISQ + classique |

### Applications

1. **Génération de distributions** : modéliser des distributions de probabilité complexes (finance, physique) avec un coût en qubits $O(\log N)$ pour $N$ états.

2. **Data augmentation** : générer des échantillons synthétiques pour enrichir des jeux de données médicaux ou industriels limités.

3. **Quantum chemistry** : échantillonner des configurations moléculaires à partir de la distribution thermique d'un Hamiltonien.

---

## Implémentation Qiskit ML

```python
from qiskit_machine_learning.algorithms import NumPyDiscriminator
from qiskit_machine_learning.algorithms import QGAN

# Configuration du qGAN avec Qiskit ML
qgan = QGAN(
    n_qubits=4,
    batch_size=100,
    num_epochs=50,
    discriminator=NumPyDiscriminator(n_features=4),
    generator=generator_circuit
)
qgan.run(quantum_instance)

# Échantillonnage après entraînement
samples = qgan.generator.sample(100)
```

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. — §7.5.
- [QML25] Qiskit ML Contributors. « Qiskit Machine Learning: an open-source library for QML tasks at scale. » *arXiv:2505.17756*, 2025.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [LCU26] « Non-Unitary Quantum Machine Learning: Fisher Efficiency Transitions. » *arXiv:2603.27377*, 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
