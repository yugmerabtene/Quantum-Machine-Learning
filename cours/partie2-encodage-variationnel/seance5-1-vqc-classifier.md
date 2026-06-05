# Séance 5.1 — Variational Quantum Classifier (VQC)

## 1. Architecture complète d'un VQC

Le **Variational Quantum Classifier** (VQC) est l'analogue quantique d'un réseau de neurones pour la classification supervisée. Il suit la structure :

$$
\mathbf{x} \xrightarrow{\text{Encodage}} |\phi(\mathbf{x})\rangle \xrightarrow{V(\boldsymbol{\theta})} |\psi(\mathbf{x}, \boldsymbol{\theta})\rangle \xrightarrow{\text{Mesure}} f(\mathbf{x}; \boldsymbol{\theta})
$$

### 1.1 Pipeline complet

```
Données brutes → Normalisation → Encodage quantique → Circuit variationnel → Mesure → Sortie → Perte → Optimisation
```

Chaque composant est différentiable, permettant la rétropropagation à travers le circuit quantique.

### 1.2 Classification binaire

Pour un problème à 2 classes $y \in \{0, 1\}$, on mesure l'espérance de $Z$ sur un qubit :

$$
f(\mathbf{x}; \boldsymbol{\theta}) = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_0 | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, 1]
$$

La prédiction est :

$$
\hat{y} = \begin{cases} 1 & \text{si } f(\mathbf{x}; \boldsymbol{\theta}) > 0 \\ 0 & \text{sinon} \end{cases}
$$

### 1.3 Classification multi-classe

Pour $C$ classes, on utilise $C$ observables (ou $C$ mesures) :

$$
f_k(\mathbf{x}; \boldsymbol{\theta}) = \langle Z_k \rangle, \quad \hat{y} = \arg\max_k f_k(\mathbf{x}; \boldsymbol{\theta})
$$

```python
@qml.qnode(dev)
def vqc_multi(x, theta):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.BasicEntanglerLayers(theta, wires=range(n_qubits))
    # Mesure sur tous les qubits pour multi-classe
    return [qml.expval(qml.Z(i)) for i in range(n_classes)]
```

---

## 2. Fonction de coût

### 2.1 Cross-entropy binaire

Pour la classification binaire avec sortie sigmoid :

$$
\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(f_i)) + (1 - y_i) \log(1 - \sigma(f_i)) \right]
$$

où $\sigma(z) = 1/(1 + e^{-z})$ et $f_i = f(\mathbf{x}_i; \boldsymbol{\theta})$.

### 2.2 Cross-entropy multi-classe (softmax)

$$
\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C y_{i,k} \log\left( \frac{e^{f_{i,k}}}{\sum_j e^{f_{i,j}}} \right)
$$

### 2.3 Mean Squared Error (MSE)

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \left( y_i - f(\mathbf{x}_i; \boldsymbol{\theta}) \right)^2
$$

```python
def cost_function(theta, X, y, circuit):
    """Cross-entropy loss pour VQC binaire."""
    predictions = np.array([circuit(x, theta) for x in X])
    preds_sigmoid = 1 / (1 + np.exp(-predictions))
    loss = -np.mean(y * np.log(preds_sigmoid + 1e-10) +
                    (1 - y) * np.log(1 - preds_sigmoid + 1e-10))
    return loss
```

---

## 3. Optimisation hybride classique-quantique

### 3.1 Boucle d'entraînement

L'optimisation suit le schéma hybride : le circuit quantique calcule le gradient (via parameter-shift), l'optimiseur classique met à jour les paramètres.

```python
def train_vqc(X_train, y_train, X_test, y_test, n_qubits=4, n_layers=3, epochs=100):
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(x, theta):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.BasicEntanglerLayers(theta, wires=range(n_qubits))
        return qml.expval(qml.Z(0))
    
    # Initialisation aléatoire des paramètres
    theta = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits))
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    for epoch in range(epochs):
        theta, loss = opt.step_and_cost(
            lambda t: cost_function(t, X_train, y_train, circuit),
            theta
        )
        
        if epoch % 10 == 0:
            acc = accuracy(X_test, y_test, circuit, theta)
            print(f"Epoch {epoch}: loss = {loss:.4f}, acc = {acc:.4f}")
    
    return theta, circuit
```

### 3.2 Optimiseurs supportés

| Optimiseur | Classe PennyLane | Particularité |
|------------|-----------------|---------------|
| SGD | `qml.GradientDescentOptimizer` | Taux d'apprentissage fixe |
| Adam | `qml.AdamOptimizer` | Momentum adaptatif |
| Adagrad | `qml.AdagradOptimizer` | Pas par paramètre |
| SPSA | `qml.SPSAOptimizer` | Gradient approximé (2 évaluations) |
| COBYLA | `qml.COBYLAOptimizer` | Sans gradient |

---

## 4. Implémentation PennyLane

### 4.1 BasicEntanglerLayers

PennyLane fournit des couches variationnelles prêtes à l'emploi :

```python
import pennylane as qml
from pennylane import numpy as np

# BasicEntanglerLayers : RX + CNOT en cascade
def circuit(x, theta):
    qml.AngleEmbedding(x, wires=range(4))
    qml.BasicEntanglerLayers(theta, wires=range(4))
    return qml.expval(qml.Z(0))
```

### 4.2 qml.qnn.TorchLayer — Intégration PyTorch

`qml.qnn.TorchLayer` permet d'utiliser un circuit comme une couche differentiable d'un réseau PyTorch :

```python
import torch
import torch.nn as nn

n_qubits = 4
n_layers = 3

# Définition du circuit
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.Z(0))

# Conversion en couche PyTorch
qlayer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits)})

# Réseau hybride
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits)})
        self.fc = nn.Linear(1, 2)  # Sortie binaire → 2 classes
    
    def forward(self, x):
        x = self.qlayer(x)
        x = torch.sigmoid(x)  # ou softmax pour multi-classe
        return x

model = HybridModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
```

---

## 5. Implémentation Qiskit Machine Learning

### 5.1 EstimatorQNN

`EstimatorQNN` est une QNN qui retourne des valeurs d'espérance (idéal pour régression ou classification binaire) :

```python
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA

n_qubits = 4

# Feature map
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)

# Ansatz
ansatz = RealAmplitudes(num_qubits=n_qubits, reps=3)

# QNN
estimator_qnn = EstimatorQNN(
    circuit=ansatz.compose(feature_map, front=True),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# Classifieur
classifier = NeuralNetworkClassifier(
    estimator_qnn,
    optimizer=COBYLA(maxiter=100),
    loss="cross_entropy"
)

classifier.fit(X_train, y_train)
```

### 5.2 SamplerQNN

`SamplerQNN` retourne des probabilités d'échantillonnage (idéal pour classification) :

```python
from qiskit_machine_learning.neural_networks import SamplerQNN

sampler_qnn = SamplerQNN(
    circuit=ansatz.compose(feature_map, front=True),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=lambda x: x,  # interprétation directe des bits
    output_shape=2  # 2 classes
)

classifier_sampler = NeuralNetworkClassifier(
    sampler_qnn,
    optimizer=COBYLA(maxiter=100),
    loss="cross_entropy"
)
```

### 5.3 Différence EstimatorQNN vs SamplerQNN

| Aspect | EstimatorQNN | SamplerQNN |
|--------|-------------|------------|
| Sortie | Espérance d'observable | Probabilités des états |
| Type | Continue (régression, binaire) | Discrète (classification) |
| Observable | À spécifier | Naturelle (base Z) |
| Bruit | Plus sensible | Plus robuste |

---

## 6. Exemple complet : classification Iris avec VQC PennyLane

```python
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Préparation des données
iris = load_iris()
X = iris.data[:, :2]  # 2 features pour simplification
y = iris.target
y = np.where(y == 0, 0, 1)  # binaire : setosa vs non-setosa

# Normalisation
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Configuration du circuit
n_qubits = 2
n_layers = 4

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(x, theta):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.BasicEntanglerLayers(theta, wires=range(n_qubits))
    return qml.expval(qml.Z(0))

# 3. Fonction de coût
def cross_entropy(theta, X, y):
    preds = [circuit(x, theta) for x in X]
    preds = np.clip(preds, -0.999, 0.999)  # stabilité numérique
    preds = (preds + 1) / 2  # scaling [-1,1] → [0,1]
    return -np.mean(y * np.log(preds + 1e-10) + (1-y) * np.log(1-preds + 1e-10))

# 4. Entraînement
theta = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits))
opt = qml.AdamOptimizer(stepsize=0.1)

for epoch in range(100):
    theta, loss = opt.step_and_cost(
        lambda t: cross_entropy(t, X_train, y_train), theta
    )
    
    if epoch % 20 == 0:
        preds = [circuit(x, theta) for x in X_test]
        preds = ((np.array(preds) + 1) / 2 > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.4f}")

# 5. Évaluation finale
preds = [circuit(x, theta) for x in X_test]
preds = ((np.array(preds) + 1) / 2 > 0.5).astype(int)
final_acc = accuracy_score(y_test, preds)
print(f"\nAccuracy finale: {final_acc:.4f}")
```

**Résultat attendu** : Avec 2 features (longueur/largeur des sépales) et 2 qubits, le VQC atteint typiquement 95–100% sur la classification binaire setosa vs. non-setosa.

---

## 7. Exercices

1. **Iris à 3 classes** : Adaptez l'exemple complet pour les 3 classes d'Iris. Utilisez 3 qubits de sortie (softmax).
2. **Comparaison Qiskit vs PennyLane** : Implémentez le même classifieur avec Qiskit ML et comparez performance et temps d'entraînement.
3. **Architecture** : Comparez `BasicEntanglerLayers` avec `StronglyEntanglingLayers` sur Iris. Lequel converge plus vite ?

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. — §5.2–5.3.
- [QML25] Qiskit ML Contributors. « Qiskit Machine Learning: an open-source library for QML tasks at scale. » *arXiv:2505.17756*, 2025.
- [VQC26] « VQC-MLPNet: Unconventional Hybrid Quantum-Classical Architecture. » *arXiv:2506.10275*, 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [Pen26] Xanadu. « Why quantum computers could be great for machine learning after all. » *PennyLane Blog*, Mars 2026.
