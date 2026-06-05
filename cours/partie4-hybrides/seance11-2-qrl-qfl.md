# Séance 11.2 — Quantum Reinforcement Learning et Quantum Federated Learning

## Quantum Reinforcement Learning (QRL)

Le Quantum Reinforcement Learning combine l'apprentissage par renforcement classique avec des circuits quantiques variationnels pour représenter la politique ou la fonction de valeur. L'objectif est d'exploiter l'espace de Hilbert exponentiel pour capturer des structures de décision complexes.

---

### Policy Gradient Quantique

#### Principe

Dans le policy gradient classique (REINFORCE, PPO), la politique $\pi_\theta(a|s)$ est paramétrée par un réseau de neurones. En QRL, cette politique est remplacée par un VQC :

$$
\pi_\theta(a|s) = |\langle a | U_\theta(s) | 0^n \rangle|^2
$$

où $U_\theta(s)$ est un circuit encodant l'état $s$ et paramétré par $\theta$.

```python
import pennylane as qml
import torch
import numpy as np

n_qubits = 4
n_actions = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_policy(state, weights):
    """Policy quantique : état → distribution sur les actions."""
    qml.AngleEmbedding(state, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_actions)]

class QuantumPolicy(torch.nn.Module):
    def __init__(self, state_dim=4, n_actions=2):
        super().__init__()
        self.n_actions = n_actions
        self.fc = torch.nn.Linear(state_dim, n_qubits)
        self.weights = torch.nn.Parameter(
            torch.randn(2, n_qubits, 3) * 0.1
        )

    def forward(self, state):
        encoding = torch.tanh(self.fc(state))
        probs = quantum_policy(encoding, self.weights)
        return torch.softmax(probs, dim=-1)
```

La mise à jour des paramètres suit le gradient de la politique :

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \, R_t \right]
$$

où $R_t = \sum_{k=t}^T \gamma^{k-t} r_k$ est le retour cumulé à horizon $T$ et $\gamma$ le facteur d'escompte.

---

### Deep Q-Network Quantique

#### Principe

Le Deep Q-Network (DQN) classique utilise un réseau pour estimer la fonction Q :

$$
Q(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q(s', a')]
$$

La version quantique remplace le réseau Q par un VQC :

```python
@qml.qnode(dev)
def quantum_q_network(state, action, weights):
    """Estimation de Q(s,a) par circuit quantique."""
    qml.AngleEmbedding(state, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class QuantumDQN(torch.nn.Module):
    def __init__(self, state_dim=4, n_actions=2):
        super().__init__()
        self.n_actions = n_actions
        self.encoder = torch.nn.Linear(state_dim, n_qubits)
        self.weights = torch.nn.Parameter(
            torch.randn(3, n_qubits, 3) * 0.1
        )

    def forward(self, x):
        x = torch.tanh(self.encoder(x))
        q_values = torch.stack([
            quantum_q_network(x, a, self.weights)
            for a in range(self.n_actions)
        ])
        return q_values
```

**Algorithme :** Les transitions $(s, a, r, s')$ sont stockées dans un *replay buffer* et échantillonnées aléatoirement pour briser les corrélations temporelles. La perte est :

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
$$

où $\theta^-$ est une copie lentement mise à jour (target network).

---

## Quantum Federated Learning (QFL)

### Principe

Le Quantum Federated Learning [Rev26] étend le paradigme du federated learning (McMahan et al., 2017) aux modèles QML. Des clients distribués entraînent localement des VQC sur leurs données privées, et un serveur central aggrège les paramètres sans accéder aux données brutes.

**Algorithme FedAvg quantique :**

```python
def quantum_fed_avg(client_weights, client_sizes):
    """Agrégation FedAvg des paramètres VQC."""
    total = sum(client_sizes)
    aggregated = sum(
        w * (size / total)
        for w, size in zip(client_weights, client_sizes)
    )
    return aggregated

def quantum_fed_prox(local_weights, global_weights, mu=0.01):
    """FedProx : régularisation proximale pour l'hétérogénéité."""
    proximal_term = (mu / 2) * sum(
        (lw - gw)**2 for lw, gw in zip(local_weights, global_weights)
    )
    return proximal_term
```

### Défis spécifiques au QFL

1. **Communication** : Les paramètres $\theta$ des VQC sont de dimension $O(L \times n)$, bien plus compacts que les poids de réseaux profonds classiques ($\sim 10^6$). Cependant, la communication multi-tours reste coûteuse.

2. **Hétérogénéité des clients** : Les clients peuvent avoir des capacités quantiques différentes (simulateur vs. matériel réel, nombre de qubits variable). Le framework Quantum LEGO [QuL26] propose une solution via des blocs adaptatifs.

3. **Sécurité et confidentialité** : Bien que les données ne soient pas partagées, les gradients $\nabla_\theta \mathcal{L}$ peuvent fuiter de l'information. Des techniques de *differential privacy* quantique sont en développement :

$$
\tilde{\nabla}_\theta \mathcal{L} = \nabla_\theta \mathcal{L} + \mathcal{N}(0, \sigma^2 I), \quad
\sigma \propto \frac{2 \log(1.25 / \delta)}{\epsilon}
$$

```python
def add_quantum_noise(gradient, epsilon=1.0, delta=1e-5):
    """Differential privacy pour les gradients VQC."""
    sensitivity = 2 * np.pi  # borne de la parameter-shift rule
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = torch.randn_like(gradient) * sigma
    return gradient + noise
```

---

## Applications

| Domaine | Application | Modèle QRL/QFL |
|---------|------------|---------------|
| Robotique | Contrôle de bras articulé, navigation | Quantum Policy Gradient |
| Finance | Optimisation de portefeuille multi-période | Quantum DQN |
| Jeux | Cartpole, FrozenLake, Atari | Quantum DQN / Policy |
| Santé | Apprentissage fédéré sur données hospitalières | QFL + FedAvg |
| Télécoms | Allocation de ressources distribuée | QFL avec clients hétérogènes |

### Résultats expérimentaux

Des expériences préliminaires [Rev26] montrent :

- **Cartpole** : un QRL avec 4 qubits atteint la même performance qu'un MLP à 2 couches cachées de 64 neurones, avec $10\times$ moins de paramètres.
- **FedAvg quantique** sur 10 clients : convergence en 50 rounds, accuracy $91.2\%$ vs. $92.8\%$ pour le FL classique (écart dû au bruit NISQ).

---

## Défis et Perspectives

1. **Passage à l'échelle** : Les environnements RL ont des espaces d'état et d'action élevés. L'angle encoding requiert $O(\dim(S))$ qubits, ce qui dépasse les capacités NISQ actuelles.

2. **Bruit matériel** : En QFL, le bruit des processeurs NISQ dégrade l'agrégation. Des techniques de *noise-resilient aggregation* sont en développement.

3. **Hétérogénéité** : Clients avec différents nombres de qubits, profondeurs de circuit, ou même frameworks. Le transfert de paramètres entre architectures hétérogènes reste un problème ouvert.

4. **Sécurité** : Les attaques par inference de gradient existent aussi en QFL. La *differential privacy* quantique, bien qu'analogue au cas classique, doit tenir compte de la structure non-linéaire des VQC.

---

## Références

- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [QuL26] « Quantum LEGO Learning: A Modular and Architecture-Agnostic Hybrid QML Framework. » *arXiv:2601.21780*, 2026.
- [VQC26] « VQC-MLPNet: Unconventional Hybrid Quantum-Classical Architecture. » *arXiv:2506.10275*, 2026.
- [LCU26] « Non-Unitary Quantum Machine Learning: Fisher Efficiency Transitions. » *arXiv:2603.27377*, 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
