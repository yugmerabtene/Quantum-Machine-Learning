# Devoir 4 — Architecture hybride et projet préparatoire

**Cours** : Quantum Machine Learning (Master/PhD)
**Semaine** : 11 | **À rendre** : Semaine 13
**Poids** : 7,5 % de la note finale

---

## 1. Problèmes théoriques

### Problème 1 — Quantum LEGO Learning

#### Question 1.1
Expliquer le principe du **Quantum LEGO Learning** (QuL26). En quoi consiste l'approche « bloc gelé + VQC adaptatif » ?

**Indice** : voir arXiv:2601.21780 (2026). Le framework décompose un pipeline QML en briques (LEGO blocks) qui peuvent être gelées ou entraînées. Certaines parties (feature extraction classique) sont pré-entraînées et figées, tandis que le module quantique variationnel est adapté à la tâche cible.

#### Question 1.2
Quels sont les avantages pour la généralisation ?
1. Réduction du nombre de paramètres à entraîner
2. Réutilisation de features pré-appris
3. Adaptation à faible échantillon (few-shot)
4. Robustesse aux barren plateaux

Justifier chacun de ces points.

#### Question 1.3
Comparez cette approche avec le transfer learning classique (fine-tuning d'un CNN complet). Le Quantum LEGO Learning offre-t-il un avantage fondamental ou simplement pratique ?

---

### Problème 2 — Quantum Transformer

#### Question 2.1
Décrire l'architecture d'un **Quantum Transformer** (Tra26). Comment les matrices Q, K, V (Query, Key, Value) du mécanisme d'attention sont-elles implémentées quantiquement ?

**Indice** : voir *Quantum Machine Intelligence* 8, 43 (2026). L'attention multi-têtes classique utilise des projections linéaires. Dans le Quantum Transformer, ces projections sont remplacées par des VQC paramétrés qui produisent des états quantiques représentant Q, K, V.

#### Question 2.2
Le score d'attention quantique est :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Comment la mesure et la post-sélection modifient-elles ce calcul dans le cas quantique ?

#### Question 2.3
Quels sont les résultats rapportés sur Fashion MNIST (94,42 %) et CIFAR-10 (90,57 %) ? Comparez avec les Transformers classiques de taille équivalente.

---

### Problème 3 — Non-unitaire QML via LCU

#### Question 3.1
Expliquer le principe de la **Linear Combination of Unitaries** (LCU). Comment permet-elle d'implémenter des opérations non-unitaires dans un circuit quantique ?

$$U_{\text{non-unitaire}} = \sum_{i} \alpha_i U_i \quad \text{avec} \quad \sum_i |\alpha_i| = 1$$

**Indice** : utilisez un registre d'ancilla et la préparation d'état $$\sum_i \sqrt{\alpha_i} |i\rangle$$, puis les unitaires contrôlés $$U_i$$, et enfin une post-sélection.

#### Question 3.2
Qu'est-ce que la **Fisher efficiency transition** ? Pourquoi se produit-elle à 10–12 qubits selon [LCU26] ?

**Indice** : l'information de Fisher mesure la sensibilité des paramètres du circuit aux données. La transition correspond au point où la convergence des VQC non-unitaires devient meilleure que celle des VQC unitaires. Voir arXiv:2603.27377.

---

### Problème 4 — VQE vs. QAOA

#### Question 4.1
Comparer **VQE** (Variational Quantum Eigensolver) et **QAOA** (Quantum Approximate Optimization Algorithm) selon les critères suivants :

| Critère | VQE | QAOA |
|---------|-----|------|
| Objectif | | |
| Type d'ansatz | | |
| Fonction de coût | | |
| Type de problème | | |
| Garantie de convergence | | |

#### Question 4.2
Donner un exemple de problème résolu par chacun :
- VQE : calcul de l'état fondamental de H₂
- QAOA : MaxCut sur un graphe 3-régulier

Décrire les similitudes et différences dans la boucle classique-quantique.

---

## 2. Problèmes d'implémentation — Projet préparatoire

### Problème 5 — Transfer Learning hybride : ResNet18 + VQC (PennyLane)

Ce problème constitue un **projet préparatoire** pour le projet final. L'objectif est d'implémenter une architecture hybride complète et de comparer sérieusement la tête quantique vs. la tête classique.

#### Question 5.1 — Préparation du dataset synthétique ou Iris transformé

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

**Option A — Dataset synthétique** : générer des images 3×32×32 à partir d'Iris en dupliquant les features et ajoutant du bruit :

```python
def iris_to_images(X, img_size=32):
    """Transforme chaque point d'Iris en une image synthétique."""
    n_samples = X.shape[0]
    images = np.zeros((n_samples, 3, img_size, img_size))
    for i in range(n_samples):
        for c in range(3):
            images[i, c] = X[i, c] * np.ones((img_size, img_size))
            images[i, c] += np.random.normal(0, 0.1, (img_size, img_size))
    return images
```

**Option B** : utiliser un vrai sous-ensemble d'images (CIFAR-10 binaire, ou un dataset médical réduit).

#### Question 5.2 — Feature extractor gelé (ResNet18)

```python
from torchvision import models

# Charger ResNet18 pré-entraîné, sans la tête fully-connected
resnet = models.resnet18(pretrained=True)
modules = list(resnet.children())[:-1]  # Enlever le FC final
feature_extractor = nn.Sequential(*modules)

# Geler tous les paramètres
for param in feature_extractor.parameters():
    param.requires_grad = False

# Output : 512 features (ResNet18 avant le FC)
def extract_features(images):
    with torch.no_grad():
        features = feature_extractor(images)
    return features.view(features.size(0), -1)  # [batch, 512]
```

1. Extraire les features des images synthétiques.
2. Normaliser les features pour l'Angle Encoding (dimension réduite par PCA à 4 features).

#### Question 5.3 — Tête quantique (VQC 4-qubits)

```python
n_qubits = 4
n_layers = 3

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def vqc_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

Convertir en couche PyTorch :

```python
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)
        self.post_process = nn.Linear(n_qubits, 3)  # 3 classes Iris
        
    def forward(self, x):
        x = self.qlayer(x)
        return self.post_process(x)
```

#### Question 5.4 — Tête classique (baseline)

```python
class ClassicalHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
```

#### Question 5.5 — Comparaison complète

Entraîner les deux modèles (tête quantique et tête classique) sur les mêmes données :

```python
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    history = {"loss": [], "acc": []}
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        history["loss"].append(running_loss / len(train_loader))
        history["acc"].append(correct / total)
    return history
```

Produire un tableau de comparaison :

| Métrique | Tête quantique (4 qubits) | Tête classique (FC) |
|----------|--------------------------|---------------------|
| Accuracy finale (test) | | |
| F1-score | | |
| Nombre de paramètres | | |
| Temps d'entraînement/epoch | | |
| Convergence (epochs) | | |

#### Question 5.6 — Analyse de l'apport quantique

Discuter :
1. L'apport quantique est-il significatif par rapport à la tête classique ?
2. Tester avec $$n_{\text{qubits}} \in \{2, 4, 6, 8\}$$ : l'accuracy augmente-t-elle avec le nombre de qubits ?
3. Quel est le rôle de la réduction de dimensionnalité (PCA avant l'encodage quantique) ?
4. La tête quantique est-elle plus ou moins sujette au sur-apprentissage que la tête classique ?

---

### Problème 6 — Analyse approfondie (optionnel)

#### Question 6.1
Répéter l'expérience avec un dataset d'images réel (Hymenoptera, PlantVillage, ou CIFAR-10 binaire). Les conclusions sont-elles les mêmes ?

#### Question 6.2
Implémenter une variante avec **Quantum LEGO Learning** : au lieu de geler tout le ResNet18, geler seulement les premières couches et laisser les dernières entraînables avec le VQC. Comparer les performances.

---

## 3. Pistes et ressources

- **Quantum LEGO Learning** : arXiv:2601.21780 (2026)
- **Quantum Transformers** : *Quantum Machine Intelligence* 8, 43 (2026)
- **LCU non-unitaire QML** : arXiv:2603.27377 (2026)
- **Quantum transfer learning** : arXiv:2603.16973 (2025–2026)
- **PennyLane TorchLayer** : https://docs.pennylane.ai/en/stable/code/api/pennylane.qnn.TorchLayer.html
- **PyTorch ResNet** : https://pytorch.org/vision/main/models/resnet.html
- **Schuld & Petruccione (2021)** : Chapitre 7 — Hybrid quantum-classical learning
