# Séance 2.2 — Calcul quantique : rappel intensif

## Postulats de la mécanique quantique

Le calcul quantique repose sur quatre postulats fondamentaux :

**Postulat 1 (État)** : L'état d'un système quantique isolé est représenté par un vecteur unitaire $|\psi\rangle$ dans un espace de Hilbert $\mathcal{H}$.

**Postulat 2 (Évolution)** : L'évolution d'un système quantique fermé est décrite par une transformation unitaire $U$ telle que $|\psi'\rangle = U|\psi\rangle$, où $U^\dagger U = I$.

**Postulat 3 (Mesure)** : Une mesure projective est décrite par un ensemble d'opérateurs $\{P_m\}$ tels que $P_m^\dagger = P_m$ et $P_m P_{m'} = \delta_{mm'} P_m$. La probabilité d'obtenir le résultat $m$ est $p(m) = \langle \psi | P_m | \psi \rangle$.

**Postulat 4 (Composition)** : L'espace d'états d'un système composite est le produit tensoriel des espaces de ses composants : $\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B$.

## Le qubit

Un qubit est un système quantique à deux niveaux. Son état général s'écrit :

$$
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle, \quad \alpha, \beta \in \mathbb{C}, \quad |\alpha|^2 + |\beta|^2 = 1
$$

où $\{|0\rangle, |1\rangle\}$ est la base de calcul. Un qubit peut exister dans une **superposition** linéaire des deux états de base.

```python
import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def superposition():
    qml.Hadamard(wires=0)
    return qml.state()

print(f"État après Hadamard : {superposition()}")
# Retourne [1/√2, 1/√2]
```

## Portes quantiques

Les portes quantiques sont des opérateurs unitaires agissant sur un ou plusieurs qubits.

### Portes de Pauli

Les matrices de Pauli forment une base pour les opérateurs 1-qubit :

$$
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

La porte $X$ agit comme un NOT quantique : $X|0\rangle = |1\rangle$, $X|1\rangle = |0\rangle$.

### Porte de Hadamard

$$
H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$

Elle crée la superposition : $H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$, $H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$.

### Portes de phase et T

$$
S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \quad
T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}
$$

### Porte CNOT

La porte CNOT (Controlled-NOT) est une porte à 2 qubits essentielle pour l'intrication :

$$
\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}
$$

Elle agit comme : $|00\rangle \mapsto |00\rangle$, $|01\rangle \mapsto |01\rangle$, $|10\rangle \mapsto |11\rangle$, $|11\rangle \mapsto |10\rangle$.

```python
@qml.qnode(dev)
def bell_state():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

dev2 = qml.device("default.qubit", wires=2)
qml.QNode(bell_state, dev2)
print(f"État de Bell : {bell_state()}")
```

## Sphère de Bloch

Tout état pur 1-qubit peut être représenté sur la **sphère de Bloch** :

$$
|\psi\rangle = \cos\frac{\theta}{2} |0\rangle + e^{i\phi} \sin\frac{\theta}{2} |1\rangle
$$

avec $\theta \in [0, \pi]$ et $\phi \in [0, 2\pi)$. Les portes unitaires 1-qubit correspondent à des rotations sur cette sphère :

$$
R_x(\theta) = e^{-i\theta X/2}, \quad
R_y(\theta) = e^{-i\theta Y/2}, \quad
R_z(\theta) = e^{-i\theta Z/2}
$$

```python
@qml.qnode(dev)
def rotation(theta, phi):
    qml.RX(theta, wires=0)
    qml.RY(phi, wires=0)
    return qml.state()

theta, phi = np.pi/3, np.pi/4
state = rotation(theta, phi)
print(f"État sur la sphère de Bloch : θ={theta:.2f}, φ={phi:.2f}")
print(f"État : {state}")
```

## Mesure projective

La mesure dans la base $Z$ projette l'état sur $|0\rangle$ ou $|1\rangle$ avec probabilités $|\alpha|^2$ et $|\beta|^2$. Après mesure, l'état s'effondre sur le résultat observé.

Pour une observable $O$ (opérateur hermitien), l'espérance de mesure est :

$$
\langle O \rangle = \langle \psi | O | \psi \rangle
$$

```python
@qml.qnode(dev)
def measure_expectation():
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

print(f"⟨Z⟩ après Hadamard : {measure_expectation():.4f}")
```

## Intrication et états de Bell

L'intrication est une corrélation quantique sans analogue classique. Les **états de Bell** sont les états maximally intriqués pour 2 qubits :

$$
|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, \quad
|\Phi^-\rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}
$$
$$
|\Psi^+\rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}, \quad
|\Psi^-\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}
$$

### Inégalités de CHSH

Les inégalités de CHSH (Clauser-Horne-Shimony-Holt) fournissent un test expérimental de l'intrication. On définit :

$$
S = \langle A_0 B_0 \rangle + \langle A_0 B_1 \rangle + \langle A_1 B_0 \rangle - \langle A_1 B_1 \rangle
$$

où $A_i, B_j$ sont des observables avec valeurs $\pm 1$. Pour une théorie réaliste locale, $|S| \leq 2$ (borne de Bell). La mécanique quantique prédit $|S| = 2\sqrt{2}$, ce qui a été vérifié expérimentalement.

```python
import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def chsh_circuit(theta_a, theta_b):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta_a, wires=0)
    qml.RY(theta_b, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Angles pour CHSH
angles = [(0, np.pi/4), (0, 3*np.pi/4), (np.pi/2, np.pi/4), (np.pi/2, 3*np.pi/4)]
S = sum(chsh_circuit(a, b) for a, b in angles[:3]) - chsh_circuit(*angles[3])
print(f"Valeur S de CHSH : {S:.4f} (quantique max : {2*np.sqrt(2):.4f})")
```

## Circuit quantique formel

Un circuit quantique se représente comme une séquence de portes agissant sur $n$ qubits. Formellement, c'est la composition d'opérateurs unitaires :

$$
U = U_L U_{L-1} \cdots U_1
$$

L'état de sortie est $|\psi_{\text{out}}\rangle = U |0\rangle^{\otimes n}$.

**Universalité** : tout opérateur unitaire peut être approximé avec une précision arbitraire par un circuit utilisant seulement les portes $H$, $T$, et CNOT (ensemble universel). En pratique, on utilise des jeux de portes plus riches ($R_x, R_y, R_z, CX$) pour faciliter l'implémentation.

```python
@qml.qnode(dev)
def quantum_circuit(params):
    # Encodage
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    # Intrication
    qml.CNOT(wires=[0, 1])
    # Couche variationnelle
    qml.RZ(params[2], wires=0)
    qml.RX(params[3], wires=1)
    # Mesure
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

params = np.random.randn(4)
print(f"Mesures : {quantum_circuit(params)}")
```

## Liens avec le QML

Ces fondamentaux sont la base de tous les modèles QML :
- L'encodage des données utilise $R_x, R_y, R_z$ ou $H$ (séance 4.1)
- Les circuits variationnels sont des séquences paramétrées de portes (séance 4.2)
- La mesure donne les valeurs d'espérance servant de prédictions (séance 5.1)
- L'intrication est nécessaire pour la puissance expressive des feature maps (séance 7.1)

## Références

- [NC00] Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information.* Cambridge University Press, 2000. §1–2.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. §3.1–3.3.
- [Pre98] Preskill, J. *Lecture Notes for Physics 219: Quantum Computation.* Caltech, 1998–2023.
- [Aar13] Aaronson, S. *Quantum Computing Since Democritus.* Cambridge University Press, 2013.
