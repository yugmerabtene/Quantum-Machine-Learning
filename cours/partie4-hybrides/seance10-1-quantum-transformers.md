# Séance 10.1 — Quantum Transformers

## Introduction

Les Quantum Transformers [Tra26] intègrent des circuits quantiques variationnels (VQC) dans l'architecture des Transformers classiques (Vaswani et al., 2017). L'objectif est de remplacer les mécanismes coûteux d'attention par des opérations quantiques exploitant l'espace de Hilbert, tout en conservant la puissance représentationnelle des Transformers.

---

## Mécanisme d'Attention Quantique

### Attention multi-têtes classique : rappel

L'attention multi-têtes classique calcule pour chaque tête :

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

où $Q = XW_Q$, $K = XW_K$, $V = XW_V$ sont les matrices Query, Key, Value, et $d_k$ la dimension de projection.

### Analogues quantiques de Q, K, V

Dans le Quantum Transformer, les vecteurs Q, K, V sont encodés dans des états quantiques via des circuits d'encoding paramétrés :

$$
|q_i\rangle = U_{\text{enc}}(x_i ; \theta_Q), \quad
|k_j\rangle = U_{\text{enc}}(x_j ; \theta_K), \quad
|v_j\rangle = U_{\text{enc}}(x_j ; \theta_V)
$$

L'attention entre la position $i$ et $j$ est mesurée par la **fidélité** (SWAP test) entre $|q_i\rangle$ et $|k_j\rangle$ :

$$
A_{ij} = \frac{|\langle q_i | k_j \rangle|^2}{\sum_j |\langle q_i | k_j \rangle|^2}
$$

```python
@qml.qnode(dev)
def quantum_attention(q_params, k_params, n_qubits=4):
    """Calcule la matrice d'attention via fidelity SWAP test."""
    # Encodage des queries
    for i in range(n_qubits):
        qml.RY(q_params[i], wires=i)
        qml.RZ(q_params[i + n_qubits], wires=i)

    # SWAP test : fidélité |<q|k>|^2
    qml.Hadamard(wires=n_qubits)  # ancilla
    for i in range(n_qubits):
        qml.CSWAP(wires=[n_qubits, i, i + n_qubits + 1])
    qml.Hadamard(wires=n_qubits)

    return qml.expval(qml.PauliZ(n_qubits))
```

### Avantage de l'attention quantique

L'attention quantique permet d'explorer un espace de similarités exponentiellement plus riche que l'attention classique :

$$
\text{dim}(\mathcal{H}_{\text{classique}}) = d_k, \quad
\text{dim}(\mathcal{H}_{\text{quantique}}) = 2^{n_{\text{qubits}}}
$$

---

## Quantum Wavelet Kolmogorov-Arnold Networks (QWKAN)

### Principe

Les QWKAN [Tra26] combinent les transformées en ondelettes quantiques avec les Kolmogorov-Arnold Networks (KAN). Au lieu d'utiliser des fonctions d'activation fixes, les KAN apprennent des fonctions univariées paramétrées par splines ou ondelettes.

**Représentation KAN :**

$$
f(x_1, \ldots, x_d) = \sum_{q=1}^{2d+1} \Phi_q\left(\sum_{p=1}^d \phi_{q,p}(x_p)\right)
$$

où $\phi_{q,p}$ sont des fonctions univariées apprises, et $\Phi_q$ la fonction de sortie.

**Extension quantique :** les fonctions $\phi_{q,p}$ sont remplacées par des circuits quantiques paramétrés dont la transformée en ondelettes est encodée dans le circuit :

```python
def qwkan_layer(x, weights, n_qubits=4):
    """Couche QWKAN : ondelettes quantiques + KAN."""
    qml.AngleEmbedding(x, wires=range(n_qubits))

    for layer in range(2):
        qml.RY(weights[layer, 0], wires=0)
        qml.RZ(weights[layer, 1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[layer, 2], wires=2)
        qml.RZ(weights[layer, 3], wires=3)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[1, 2])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

---

## Architecture Complète

L'architecture Quantum Transformer complète [Tra26] s'articule en quatre étapes :

### 1. Encodage des patches

L'image est divisée en patches, chaque patch est encodé par un CNN léger pour produire un vecteur de caractéristiques :

$$
x_{\text{patch}} = \text{CNN}_{\text{encoder}}(I_{\text{patch}}) \in \mathbb{R}^{d}
$$

### 2. Multi-Head Quantum Attention (MHQA)

Chaque tête d'attention utilise un VQC indépendant pour les projections Q, K, V. Les têtes sont concaténées :

$$
\text{MHQA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

### 3. QWKAN Feed-Forward

La sortie de l'attention est passée dans un bloc QWKAN qui remplace le MLP classique du Transformer.

### 4. Classification

Un VQC final agrège les représentations et produit la classification.

```python
class QuantumTransformer(torch.nn.Module):
    def __init__(self, n_qubits=4, n_heads=4, n_layers=6):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(3, 64, kernel_size=4, stride=4)
        self.attention_heads = nn.ModuleList([
            QuantumAttentionHead(n_qubits) for _ in range(n_heads)
        ])
        self.qwkan = QWKANBlock(n_qubits)
        self.norm = torch.nn.LayerNorm(64)
        self.classifier = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        for _ in range(n_layers):
            attn = torch.cat([h(x) for h in self.attention_heads], dim=-1)
            x = self.norm(x + attn)
            x = self.norm(x + self.qwkan(x))
        return self.classifier(x.mean(dim=1))
```

---

## Résultats Expérimentaux

Les expériences [Tra26] sur les jeux de données standard montrent des performances compétitives :

| Jeu de données | Quantum Transformer | Transformer classique | Delta |
|---------------|-------------------|---------------------|-------|
| Fashion MNIST | **94.42%** | 93.85% | +0.57% |
| CIFAR-10      | **90.57%** | 89.34% | +1.23% |
| MNIST         | **99.12%** | 99.04% | +0.08% |

### Scalabilité linéaire

Un résultat clé [Tra26] est la scalabilité $O(d)$ de l'attention quantique avec la dimension d'entrée $d$, contre $O(d^2)$ pour l'attention classique :

$$
T_{\text{quantique}} = O(d \cdot n_{\text{qubits}}), \quad
T_{\text{classique}} = O(d^2)
$$

Cette scalabilité provient de l'encodage parallèle des $d$ dimensions dans $n_{\text{qubits}}$ qubits via l'angle embedding.

---

## Comparaison Quantum Transformer vs. Transformer classique

| Aspect | Transformer classique | Quantum Transformer |
|--------|---------------------|-------------------|
| Attention | $O(d^2)$ | $O(d)$ |
| Paramètres | $4d^2$ par tête | $O(n_{\text{qubits}}^2)$ par tête |
| Expressivité | $\mathbb{R}^{d}$ | $\mathbb{C}^{2^{n_{\text{qubits}}}}$ |
| Backend | GPU | Simulateur / NISQ |
| Performance | baseline | +0.5–1.2% |

---

## Références

- [Tra26] « Quantum Transformers for Image Classification: Integrating VQC and Quantum Wavelet KAN. » *Quantum Machine Intelligence* 8, 43 (2026).
- [QuL26] « Quantum LEGO Learning: A Modular and Architecture-Agnostic Hybrid QML Framework. » *arXiv:2601.21780*, 2026.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
- [GBC16] Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning.* MIT Press, 2016.
