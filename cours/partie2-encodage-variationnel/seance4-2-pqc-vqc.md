# Séance 4.2 — Circuits quantiques paramétrés (PQC/VQC)

## 1. Définition et structure générale

Un **Parametrized Quantum Circuit** (PQC) est un circuit quantique dépendant de paramètres classiques $\boldsymbol{\theta} \in \mathbb{R}^p$. On le note :

$$
U(\boldsymbol{\theta}) = U_L(\theta_L) \cdots U_2(\theta_2) U_1(\theta_1)
$$

où chaque $U_\ell(\theta_\ell)$ est une porte ou un bloc de portes paramétrées. Appliqué à l'état initial $|0\rangle^{\otimes n}$, il produit :

$$
|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |0\rangle^{\otimes n}
$$

Un **Variational Quantum Circuit** (VQC) est un PQC utilisé comme modèle d'apprentissage : on mesure une observable $O$ pour obtenir une prédiction :

$$
f(\mathbf{x}; \boldsymbol{\theta}) = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | O | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle
$$

ou plus généralement, un vecteur de mesures pour la classification multi-classe.

### Structure canonique d'un VQC

```
|0⟩    ┌──────┐    ┌──────────┐    ┌───────┐    ┌───┐
|0⟩    │Encodage│    │Variationnel│    │ Mesure │    │   │
|0⟩    │ ϕ(x)  │    │  U(θ)    │    │⟨O⟩    │    │...│
|0⟩    └──────┘    └──────────┘    └───────┘    └───┘
        Layer         Layer           Layer
       d'encodage   variationnelle   de mesure
```

Les trois composantes essentielles sont :
1. **Layer d'encodage** : $S(\mathbf{x})$ transforme $|0\rangle$ en $|\phi(\mathbf{x})\rangle$
2. **Couches variationnelles** : $V(\boldsymbol{\theta})$ sont entraînables
3. **Mesure** : $\langle O \rangle$ produit la sortie

---

## 2. Composants d'un PQC

### 2.1 Layer d'encodage

L'encodage transforme les données d'entrée $\mathbf{x}$ en un état quantique (voir séance 4.1). Formellement :

$$
S(\mathbf{x}) = \bigotimes_i R_i(x_i) \quad \text{ou} \quad S(\mathbf{x}) = e^{-i \sum_j x_j H_j}
$$

### 2.2 Couches variationnelles

Une couche variationnelle consiste en une alternance de **portes à un qubit** (rotations paramétrées) et de **portes à deux qubits** (intrication) :

$$
V_\ell(\boldsymbol{\theta}_\ell) = \left( \bigotimes_{i=1}^n R_i(\theta_{\ell,i}) \right) U_{\text{ent}}
$$

où $U_{\text{ent}}$ est une porte d'intrication fixe (ex : CNOT entre paires de qubits).

### 2.3 Mesure

On mesure l'espérance d'une observable $O$ :

$$
f(\mathbf{x}; \boldsymbol{\theta}) = \langle 0 | S(\mathbf{x})^\dagger V(\boldsymbol{\theta})^\dagger O V(\boldsymbol{\theta}) S(\mathbf{x}) | 0 \rangle
$$

En classification binaire, on utilise souvent $O = Z_0$ (observable Pauli $Z$ sur le premier qubit) :

```python
@qml.qnode(dev)
def vqc_circuit(x, theta):
    # Layer d'encodage
    qml.AngleEmbedding(x, wires=range(n_qubits))
    # Couches variationnelles
    qml.BasicEntanglerLayers(theta, wires=range(n_qubits))
    # Mesure
    return qml.expval(qml.Z(0))
```

---

## 3. Ansätze

Un **ansatz** est une structure de circuit choisie *a priori*. Le choix de l'ansatz détermine l'expressivité, la trainabilité, et la profondeur du circuit.

### 3.1 Circuit-centric ansatz

Proposé par Schuld & Petruccione [SP21], c'est l'analogue quantique d'un réseau de neurones :

- Alternance systématique : rotations individuelles (non-linéarité) + portes d'intrication (connexions)
- Chaque couche a $n$ paramètres de rotation + $n-1$ CNOT

```python
def circuit_centric_layer(theta, wires):
    """Une couche circuit-centric."""
    for i in wires:
        qml.RX(theta[i], wires=i)
        qml.RZ(theta[i + len(wires)], wires=i)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
```

**Avantage** : expressivité maximale pour une profondeur donnée.
**Inconvénient** : sensible aux barren plateaux (voir séance 6.1).

### 3.2 Hardware-efficient ansatz

Adapté à la topologie de connexion du matériel quantique [VQA26] :

- Portes d'intrication limitées aux connexions physiques
- Minimise la transpilation (réduction du bruit)
- Souvent : couches de $R_Y R_Z$ alternées avec des CNOT sur un graphe en ligne ou en anneau

```python
def hardware_efficient_layer(theta, wires, topology="line"):
    """Couche hardware-efficient."""
    for i in wires:
        qml.RY(theta[i], wires=i)
        qml.RZ(theta[i + len(wires)], wires=i)
    if topology == "line":
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
    elif topology == "ring":
        for i in range(len(wires)):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
```

### 3.3 Tensor-network ansatz

Inspiré des réseaux de tenseurs (MPS, PEPS), cet ansatz limite l'intrication :

- Structure en chaîne : l'intrication est locale
- Nombre de paramètres linéaire en $n$
- Idéal pour les données 1D (séries temporelles)

```python
def mps_layer(theta, wires):
    """Matrix Product State ansatz (1 couche)."""
    for i in range(len(wires) - 1):
        qml.RY(theta[i], wires=wires[i])
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.RY(theta[-1], wires=wires[-1])
```

### 3.4 Tree Tensor Network (TTN)

Topologie hiérarchique en arbre binaire :

```
    ●
   / \
  ●   ●
 / \ / \
●   ●   ●
```

Avantage : chaque qubit interagit avec un nombre logarithmique de partenaires, limitant la propagation des barren plateaux.

```python
def ttn_block(theta, wires, block_size=2):
    """Bloc TTN de taille block_size."""
    if len(wires) == 1:
        qml.RY(theta[0], wires=wires[0])
        return
    mid = len(wires) // 2
    left, right = wires[:mid], wires[mid:]
    ttn_block(theta[:mid], left)
    ttn_block(theta[mid:], right)
    # Fusion
    for i in range(mid):
        qml.CNOT(wires=[left[i], right[i]])
        qml.RY(theta[i + mid], wires=right[i])
```

---

## 4. Expressivité des PQC

### 4.1 Mesure de l'expressivité

L'expressivité d'un PQC mesure sa capacité à générer des états différents dans l'espace de Hilbert. Elle est quantifiée par le **volume de l'espace des états accessibles** [SP21] :

$$
\mathcal{V}(\mathcal{U}) = \int_{\boldsymbol{\theta}} \mu(d\boldsymbol{\theta}) \, \delta(|\psi(\boldsymbol{\theta})\rangle)
$$

où $\mu$ est la distribution des paramètres et $\delta$ la mesure de Haar sur $\mathcal{H}$.

Un PQC est **universel** si, pour tout $|\psi\rangle \in \mathcal{H}$, il existe $\boldsymbol{\theta}$ tel que $|\psi(\boldsymbol{\theta})\rangle \approx |\psi\rangle$.

### 4.2 Distribution des états

Pour un PQC suffisamment profond (nombre de couches $L \to \infty$), la distribution des états $|\psi(\boldsymbol{\theta})\rangle$ tend vers la **mesure de Haar** :

$$
\mathbb{P}(|\psi(\boldsymbol{\theta})\rangle \in A) \xrightarrow{L \to \infty} \int_A d\psi_{\text{Haar}}
$$

Cette propriété est à double tranchant :
- **Positive** : le circuit peut explorer tout l'espace de Hilbert
- **Négative** : les gradients s'annulent (barren plateaux, séance 6.1)

### 4.3 Mesure de l'expressivité par le fidelity gap

On peut mesurer l'expressivité via la **fidélité moyenne** entre deux états générés aléatoirement :

$$
\overline{F} = \mathbb{E}_{\boldsymbol{\theta}, \boldsymbol{\theta}'} \left[ |\langle \psi(\boldsymbol{\theta}) | \psi(\boldsymbol{\theta}') \rangle|^2 \right]
$$

Pour la distribution de Haar, $\overline{F}_{\text{Haar}} = 2/(2^n + 1)$. Un PQC dont $\overline{F}$ est proche de cette valeur est dit **2-design approximatif**.

```python
def fidelity_gap(circuit, n_samples=100):
    """Calcule la fidélité moyenne d'un PQC."""
    states = []
    for _ in range(n_samples):
        theta = np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits))
        states.append(circuit(theta))
    
    fidelities = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            f = np.abs(np.dot(states[i].conj(), states[j]))**2
            fidelities.append(f)
    return np.mean(fidelities)
```

---

## 5. Universalité des PQC

### 5.1 Théorème d'approximation

Les PQC possèdent une propriété d'universalité analogue aux réseaux de neurones [SP21] :

> **Théorème (approximation par PQC)** : Pour toute fonction $f : \mathcal{X} \to \mathbb{R}$ continue et pour tout $\varepsilon > 0$, il existe un PQC $U(\boldsymbol{\theta})$ avec un nombre fini de qubits et de paramètres tel que :
>
> $$
> \sup_{\mathbf{x} \in \mathcal{X}} \left| f(\mathbf{x}) - \langle 0 | S(\mathbf{x})^\dagger U(\boldsymbol{\theta})^\dagger O U(\boldsymbol{\theta}) S(\mathbf{x}) | 0 \rangle \right| < \varepsilon
> $$

### 5.2 Preuve par construction

La démonstration repose sur l'encodage des données dans des amplitudes et l'utilisation de portes paramétrées pour approximer des séries de Fourier :

$$
f(\mathbf{x}; \boldsymbol{\theta}) = \sum_{\boldsymbol{\omega} \in \Omega} c_{\boldsymbol{\omega}}(\boldsymbol{\theta}) e^{i \boldsymbol{\omega} \cdot \mathbf{x}}
$$

Pour l'angle encoding, le spectre $\Omega$ est limité aux fréquences $\{-L, \dots, L\}^d$ où $L$ est le nombre de répétitions des rotations.

### 5.3 Limitations pratiques

Malgré l'universalité théorique :
- Le nombre de paramètres requis peut être exponentiel
- La convergence de l'optimisation est entravée par les barren plateaux
- Le bruit NISQ dégrade l'approximation

---

## 6. Exercices

1. **Expressivité** : Implémentez les 4 ansätze de cette séance pour $n=4$ qubits et $L=3$ couches. Comparez le nombre de paramètres et la profondeur de chaque circuit.
2. **Fidelity gap** : Calculez $\overline{F}$ pour chaque ansatz avec $n=4$ qubits. Lequel est le plus proche d'un 2-design ?
3. **Universalité** : Montrez qu'un circuit avec angle encoding et une couche variationnelle peut exprimer toute fonction $f(x) = a \cos(x) + b \sin(x)$. Que se passe-t-il avec deux features ?

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. — §3.6, §5.1.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [NOM26] Li, K. et al. « Learning parameterized quantum circuits with quantum gradient. » *npj Quantum Information* 12, 59 (2026).
- [Hav19] Havlíček, V. et al. « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019).
- [NC00] Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.* Cambridge University Press, 2000.
