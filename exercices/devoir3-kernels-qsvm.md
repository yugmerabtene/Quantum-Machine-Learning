# Devoir 3 — Kernels quantiques et QSVM

**Cours** : Quantum Machine Learning (Master/PhD)
**Semaine** : 8 | **À rendre** : Semaine 10
**Poids** : 7,5 % de la note finale

---

## 1. Problèmes théoriques

### Problème 1 — Fidelity Quantum Kernel

#### Question 1.1
Soit $$x, x' \in \mathbb{R}^d$$ deux points de données encodés par une feature map quantique $$|\psi(x)\rangle = U(x)|0\rangle$$. Le **fidelity quantum kernel** est défini par :

$$K_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$$

Montrer que $$K_Q$$ est un noyau valide, c'est-à-dire qu'il satisfait les conditions de Mercer (semi-défini positif pour tout ensemble de points).

**Indice** : écrivez $$K_Q(x, x') = \text{Tr}[\rho(x) \rho(x')]$$ où $$\rho(x) = |\psi(x)\rangle\langle \psi(x)|$$. Montrez que la matrice de Gram $$K_{ij} = K_Q(x_i, x_j)$$ peut s'écrire comme $$\text{Tr}[A^\dagger A]$$ pour une matrice $$A$$ bien choisie, donc semi-définie positive.

#### Question 1.2
Quelle est l'interprétation géométrique de $$K_Q$$ ? En quoi diffère-t-elle d'un noyau classique comme le RBF ?

---

### Problème 2 — Concentration exponentielle des kernels

#### Question 2.1
Expliquer le problème de **concentration exponentielle** des kernels quantiques. Pourquoi a-t-on :

$$K_Q(x, x') \xrightarrow{n \to \infty} \delta_{x, x'}$$

c'est-à-dire que le kernel devient une fonction delta (1 si $$x = x'$$, 0 sinon) quand le nombre de qubits $$n$$ devient grand ?

**Indice** : pour un circuit 2-design, la distribution des états $$|\psi(x)\rangle$$ est uniforme sur la sphère de Bloch généralisée. La probabilité que deux états aléatoires aient un recouvrement non nul décroît exponentiellement avec $$n$$.

#### Question 2.2
Quel est l'impact de cette concentration sur l'utilité pratique des kernels quantiques pour la classification ?
- La matrice de Gram devient proche de l'identité
- Plus de séparation possible entre classes
- Le modèle ne peut plus généraliser

#### Question 2.3
Qu'est-ce qu'un **kernel covariant** et comment la covariance par rapport à un groupe de symétrie peut-elle atténuer la concentration ?

**Indice** : voir Agliardi et al. (2026), *Mitigating exponential concentration in covariant quantum kernels*, npj Quantum Information 12, 12. Les kernels covariants exploitent la structure de symétrie des données.

---

### Problème 3 — Quantum Kernel Alignment

#### Question 3.1
Soit $$U(x; \theta)$$ une feature map paramétrée par $$\theta$$, définissant un kernel $$K_\theta(x, x') = |\langle 0| U^\dagger(x; \theta) U(x'; \theta) |0 \rangle|^2$$.

Le **quantum kernel alignment** (QKA) consiste à optimiser $$\theta$$ pour maximiser l'alignement avec un kernel cible $$K_{\text{target}}$$ correspondant aux labels :

$$A(\theta) = \frac{\langle K_\theta, K_{\text{target}} \rangle_F}{\sqrt{\langle K_\theta, K_\theta \rangle_F \langle K_{\text{target}}, K_{\text{target}} \rangle_F}}$$

où $$\langle \cdot, \cdot \rangle_F$$ est le produit de Frobenius.

Formuler la fonction de coût à minimiser pour le QKA.

#### Question 3.2
Quel est le kernel cible idéal pour un problème de classification binaire ?

**Indice** : $$K_{\text{target}}(x_i, x_j) = y_i y_j$$ pour des labels $$y_i \in \{-1, +1\}$$.

#### Question 3.3
Le QKA peut-il lui-même souffrir de concentration ? Discuter.

---

### Problème 4 — Bit-Flip Tolerance (BFT)

#### Question 4.1
Décrire la méthode **Bit-Flip Tolerance** (BFT) proposée par Agliardi et al. (2026). Quel est le principe général ?

**Indice** : la BFT introduit un seuil $$\tau$$ sur le poids de Hamming. Les paires de points dont la distance de Hamming dans l'espace binarisé dépasse $$\tau$$ sont traitées différemment.

#### Question 4.2
Comment le seuil $$\tau$$ doit-il être calibré ? Quelle est la relation entre $$\tau$$, le nombre de qubits $$n$$, et l'accuracy ?

**Indice** : la calibration suit une loi linéaire $$\tau(n) = \alpha n + \beta$$. Comment déterminer $$\alpha$$ et $$\beta$$ empiriquement ?

---

## 2. Problèmes d'implémentation

### Problème 5 — Implémentation du Fidelity Kernel (PennyLane)

#### Question 5.1
Charger le dataset Wine de `scikit-learn` et normaliser les features :

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_wine()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
# Réduire à 2 features pour visualisation (optionnel)
X = X[:, :2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

Le dataset Wine contient 3 classes (cultivars italiens). Normaliser entre $$[0, \pi]$$ pour l'Angle Encoding.

#### Question 5.2
Implémenter un **fidelity kernel** avec PennyLane :

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def fidelity_circuit(x1, x2):
    """Calcule |<psi(x1)|psi(x2)>|^2."""
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding(x2, wires=range(n_qubits)))
    return qml.probs(wires=range(n_qubits))

def fidelity_kernel(x1, x2):
    """Retourne la fidélité entre deux états encodés."""
    probs = fidelity_circuit(x1, x2)
    return probs[0]  # P(|0...0>) = |<psi(x1)|psi(x2)>|^2
```

Calculer la matrice de kernel $$K \in \mathbb{R}^{n \times n}$$ pour l'ensemble d'entraînement. Visualiser la matrice avec `plt.imshow()`.

#### Question 5.3
Analyser la matrice de kernel pour les 3 classes du dataset Wine :
1. Les blocs diagonaux (même classe) sont-ils plus proches de 1 ?
2. Les blocs hors-diagonale (classes différentes) sont-ils proches de 0 ?
3. Y a-t-il des signes de concentration ?

---

### Problème 6 — QSVM vs. SVM classique

#### Question 6.1
Implémenter un **QSVM** en utilisant le kernel quantique ci-dessus avec `scikit-learn` :

```python
from sklearn.svm import SVC

# QSVM avec kernel personnalisé
qsvm = SVC(kernel=fidelity_kernel)
qsvm.fit(X_train, y_train)
y_pred = qsvm.predict(X_test)
```

#### Question 6.2
Implémenter trois SVM classiques pour comparaison :

```python
# SVM RBF
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# SVM polynomial
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train, y_train)

# SVM linéaire
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
```

#### Question 6.3
Comparer les modèles sur les métriques suivantes :

| Modèle | Accuracy | F1-score (macro) | Temps d'entraînement | Temps d'inférence |
|--------|----------|-------------------|---------------------|-------------------|
| QSVM | | | | |
| SVM RBF | | | | |
| SVM Poly | | | | |
| SVM Lin. | | | | |

```python
from sklearn.metrics import accuracy_score, f1_score
import time

# Mesure du temps
start = time.time()
qsvm.fit(X_train, y_train)
fit_time = time.time() - start

start = time.time()
y_pred = qsvm.predict(X_test)
pred_time = time.time() - start

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
```

#### Question 6.4
Discussion :
1. Le QSVM est-il compétitif avec le SVM RBF sur ce dataset ?
2. Le temps d'exécution (surtout pour le kernel) est-il acceptable ?
3. Que se passe-t-il si on augmente le nombre de features ?

---

### Problème 7 — Quantum Kernel Alignment (optionnel)

#### Question 7.1
Implémenter le kernel alignment pour optimiser les paramètres du `ZZFeatureMap` :

```python
import pennylane as qml
from pennylane import numpy as np
from qiskit.circuit.library import ZZFeatureMap

# Feature map paramétrée (Qiskit pour le ZZFeatureMap)
def param_feature_map(x, n_qubits):
    """Crée un circuit ZZFeatureMap paramétré avec Qiskit."""
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    fm_circuit = fm.assign_parameters(x[:n_qubits])
    return fm_circuit
```

Définir la fonction de coût d'alignement :

```python
def kernel_alignment_cost(params, X, y):
    """Calcule -A(theta) où A est l'alignement de kernel."""
    # 1. Calculer la matrice de kernel K_theta
    K_theta = compute_kernel_matrix(X, params)
    # 2. Kernel cible
    K_target = np.outer(y, y)
    # 3. Alignement de Frobenius
    A = np.sum(K_theta * K_target) / (
        np.sqrt(np.sum(K_theta**2) * np.sum(K_target**2))
    )
    return -A  # On minimise -A pour maximiser A
```

Optimiser avec `scipy.optimize.minimize` ou `Adam` de PennyLane.

#### Question 7.2
Après optimisation, comparer le kernel aligné avec le kernel non-aligné (paramètres aléatoires). L'accuracy du QSVM s'améliore-t-elle ?

---

### Problème 8 — Bonus : Bruit dépolarisant (optionnel)

#### Question 8.1
Simuler un bruit dépolarisant sur le kernel quantique :

```python
import pennylane as qml

# Device avec bruit dépolarisant (utilise default.mixed pour simuler le bruit)
dev_noisy = qml.device("default.mixed", wires=n_qubits)

@qml.qnode(dev_noisy)
def noisy_kernel(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    # Ajout de bruit dépolarisant après chaque couche
    for i in range(n_qubits):
        qml.DepolarizingChannel(0.01, wires=i)
    return qml.probs(wires=range(n_qubits))[0]
```

#### Question 8.2
Mesurer l'accuracy du QSVM pour différents niveaux de bruit :
$$p \in \{0.0, 0.001, 0.005, 0.01, 0.05, 0.1\}$$

Produire un graphique : accuracy vs. taux de bruit. À partir de quel seuil le QSVM devient-il moins bon que le SVM RBF ?

---

## 3. Pistes et ressources

- **HavlÍček et al. (2019)** : *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209–212
- **Agliardi et al. (2026)** : *Mitigating exponential concentration in covariant quantum kernels*, npj Quantum Information 12, 12
- **PennyLane kernels tutorial** : https://pennylane.ai/qml/demos/tutorial_kernels_module
- **Qiskit ML — FidelityQuantumKernel** : https://qiskit-community.github.io/qiskit-machine-learning/
- **Schuld & Petruccione (2021)** : Chapitre 6 — Quantum kernel methods
