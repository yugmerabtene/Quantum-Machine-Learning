# Séance 9.1 — Architectures hybrides classique-quantique

## Transfer Learning Quantique

### Principe général

Le transfer learning quantique consiste à utiliser un réseau de neurones classique pré-entraîné comme extracteur de features, dont les poids sont gelés (*frozen backbone*), et à substituer la tête de classification classique par un circuit quantique variationnel (VQC). Cette approche permet de bénéficier de la puissance représentationnelle des réseaux profonds classiques tout en exploitant l'espace de Hilbert quantique pour la décision finale.

**Architecture :**

$$
y(x) = f_{\text{VQC}}(f_{\text{CNN}}(x ; \theta^*_{\text{CNN}}) ; \theta_{\text{VQC}})
$$

où $\theta^*_{\text{CNN}}$ sont les poids pré-entraînés et gelés du backbone, et $\theta_{\text{VQC}}$ les paramètres variationnels du circuit.

### Implémentation PennyLane + PyTorch

```python
import torch
import pennylane as qml
import torchvision.models as models

n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])
        for p in self.features.parameters():
            p.requires_grad = False
        self.fc = torch.nn.Linear(512, n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.output = torch.nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.features(x).flatten(1)
        x = torch.tanh(self.fc(x))
        x = self.qlayer(x)
        return self.output(x)
```

La tête quantique remplace avantageusement la tête fully-connected classique : elle offre un espace de caractéristiques de dimension $2^{n_{\text{qubits}}}$ (au lieu de $n_{\text{qubits}}$ linéairement) et une expressivité non-linéaire encodée dans le circuit.

---

## Quantum LEGO Learning

### Modularité et assemblage de blocs

Proposé dans [QuL26], le **Quantum LEGO Learning** est un cadre modulaire et agnostique à l'architecture pour la construction de modèles QML hybrides. L'idée centrale est d'assembler des *blocs gelés* (pré-entraînés et figés) avec des *VQC adaptatifs* (entraînables) selon un graphe de computation flexible.

**Formalisme :** Un modèle hybride est défini par un DAG de modules :

$$
\mathcal{M} = (B_{\text{frozen}}, B_{\text{trainable}}, \mathcal{E})
$$

où $B_{\text{frozen}}$ sont les blocs classiques ou quantiques gelés, $B_{\text{trainable}}$ les VQC adaptatifs, et $\mathcal{E}$ les arêtes spécifiant le flux de données.

```python
class QuantumLEGO(torch.nn.Module):
    def __init__(self, blocks, edges):
        super().__init__()
        self.blocks = torch.nn.ModuleDict(blocks)
        self.edges = edges

    def forward(self, x):
        activations = {"input": x}
        for src, dst, transform in self.edges:
            out = self.blocks[src](activations[src])
            activations[dst] = transform(out) if transform else out
        return activations["output"]
```

---

## Analyse Théorique de la Généralisation

### Décomposition de l'erreur

L'erreur de généralisation d'un modèle hybride se décompose en trois termes [QuL26] :

$$
\mathcal{R}(f) \leq \mathcal{R}_{\text{emp}}(f) + \underbrace{\mathcal{E}_{\text{approx}}}_{\text{approximation}} + \underbrace{\mathcal{E}_{\text{estim}}}_{\text{estimation}} + \underbrace{\mathcal{E}_{\text{opt}}}_{\text{optimisation}}
$$

1. **Erreur d'approximation** $\mathcal{E}_{\text{approx}}$ : écart entre la classe de fonctions réalisables par le modèle hybride et la fonction cible. Elle décroît avec la complexité du circuit (nombre de couches, de qubits).

2. **Erreur d'estimation** $\mathcal{E}_{\text{estim}}$ : écart entre le risque empirique (minimisé sur l'échantillon d'entraînement) et le risque réel. Elle est contrôlée par la complexité de Rademacher de la classe :

$$
\mathcal{E}_{\text{estim}} \leq 2\, \mathcal{R}_n(\mathcal{F}_{\text{hybride}}) + \sqrt{\frac{\log(1/\delta)}{2n}}
$$

3. **Erreur d'optimisation** $\mathcal{E}_{\text{opt}}$ : écart entre le minimum atteint par l'optimiseur et le minimum global de la fonction de coût.

### Bornes de généralisation par blocs

Un résultat clé de [QuL26] est que la complexité de Rademacher du modèle hybride se factorise :

$$
\mathcal{R}_n(\mathcal{F}_{\text{hybride}}) \leq \sum_{b \in B_{\text{trainable}}} \mathcal{R}_n(\mathcal{F}_b) + \sum_{b \in B_{\text{frozen}}} \mathcal{R}_n(\mathcal{F}_b^*)
$$

où $\mathcal{R}_n(\mathcal{F}_b^*)$ est la complexité du bloc gelé (nulle si les poids sont fixes, la fonction étant déterministe). En pratique, seuls les blocs entraînables contribuent à la borne.

---

## Résultats Expérimentaux

### Classification sur boîtes quantiques

Les benchmarks [QuL26] sur la classification de boîtes quantiques montrent une accuracy de $92.3\%$ avec un backbone ResNet18 + VQC 4 qubits, contre $88.7\%$ pour la tête classique fully-connected équivalente.

### Transcription Factor Binding Sites (TFBS)

Sur le jeu de données génomique TFBS, l'architecture hybride atteint une accuracy de $94.1\%$ et un AUC de $0.972$, surpassant les modèles purement classiques de $+2.3\%$.

### Exécution sur matériel IBM

Les modèles Quantum LEGO ont été déployés sur les processeurs *ibm_brisbane* et *ibm_torino* (127 qubits). La dégradation de performance due au bruit matériel est limitée à $< 3\%$ grâce à la faible profondeur des VQC adaptatifs (2-3 couches).

### Avantages des architectures hybrides

1. **Optimisation stable** : les poids gelés du backbone réduisent la variance du gradient, atténuant les barren plateaux.
2. **Sensibilité réduite au nombre de qubits** : contrairement aux VQC seuls, la performance ne chute pas avec l'augmentation du nombre de qubits.
3. **Passage à l'échelle** : le backbone classique traite les hautes dimensions, le VQC se concentre sur la décision dans l'espace de Hilbert.

---

## Références

- [QuL26] « Quantum LEGO Learning: A Modular and Architecture-Agnostic Hybrid QML Framework. » *arXiv:2601.21780*, 2026.
- [Tra25] « Quantum transfer learning for image classification. » *arXiv:2603.16973*, 2025–2026.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
