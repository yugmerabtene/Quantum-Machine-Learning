# Séance 5.2 — Différentiation automatique des circuits quantiques

## 1. Pourquoi différentier un circuit quantique ?

L'entraînement d'un VQC par descente de gradient nécessite le calcul de :

$$
\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \theta_k} = \frac{\partial}{\partial \theta_k} \left[ \langle \psi(\boldsymbol{\theta}) | O | \psi(\boldsymbol{\theta}) \rangle \right]
$$

Contrairement aux réseaux de neurones classiques, on ne peut pas utiliser la rétropropagation standard car le circuit quantique est une boîte noire unitaire — on ne peut pas "stocker" les activations intermédiaires. Il faut des méthodes spécifiques.

---

## 2. Parameter-shift rule

### 2.1 Principe fondamental

La **parameter-shift rule** permet de calculer le gradient exact d'une espérance quantique en utilisant uniquement des évaluations du circuit à des points décalés [SP21].

Soit un circuit $U(\theta) = e^{-i \theta P}$ où $P$ est un opérateur de Pauli ($P^2 = I$), et soit $f(\theta) = \langle 0 | U(\theta)^\dagger O U(\theta) | 0 \rangle$. Alors :

$$
\boxed{\frac{\partial f(\theta)}{\partial \theta} = \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}}
$$

### 2.2 Démonstration

Toute porte paramétrée $e^{-i \theta P/2}$ avec $P^2 = I$ génère une rotation $R_P(\theta) = \cos(\theta/2) I - i \sin(\theta/2) P$. L'espérance prend la forme :

$$
f(\theta) = a \cos(\theta) + b \sin(\theta) + c
$$

En dérivant :

$$
f'(\theta) = -a \sin(\theta) + b \cos(\theta)
$$

En évaluant aux points décalés :

$$
f(\theta + \pi/2) = a \cos(\theta + \pi/2) + b \sin(\theta + \pi/2) + c = -a \sin(\theta) + b \cos(\theta) + c
$$

$$
f(\theta - \pi/2) = a \cos(\theta - \pi/2) + b \sin(\theta - \pi/2) + c = a \sin(\theta) - b \cos(\theta) + c
$$

D'où :

$$
\frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2} = -a \sin(\theta) + b \cos(\theta) = f'(\theta)
$$

### 2.3 Généralisation multi-portes

Pour un paramètre $\theta_k$ qui apparaît dans plusieurs portes $U_k(\theta_k) = e^{-i \theta_k P_k/2}$ (où $P_k^2 = I$), le gradient est la somme des parameter-shifts de chaque occurrence :

```python
def parameter_shift(circuit, theta, k, args=()):
    """
    Calcule ∂f/∂θ_k par parameter-shift rule.
    
    Args:
        circuit: fonction f(theta, *args) → espérance
        theta: vecteur de paramètres
        k: index du paramètre à différentier
    """
    shift = np.pi / 2
    
    # θ_k + π/2
    theta_plus = theta.copy()
    theta_plus[k] += shift
    f_plus = circuit(theta_plus, *args)
    
    # θ_k - π/2
    theta_minus = theta.copy()
    theta_minus[k] -= shift
    f_minus = circuit(theta_minus, *args)
    
    return (f_plus - f_minus) / 2
```

### 2.4 Implémentation PennyLane

PennyLane utilise la parameter-shift rule automatiquement :

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(x, theta):
    qml.RX(x[0], wires=0)
    qml.RY(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(theta[1], wires=1)
    return qml.expval(qml.Z(0) @ qml.Z(1))

# PennyLane calcule le gradient automatiquement
x = np.array([0.5])
theta = np.array([0.3, 1.2])

grad = qml.grad(circuit, argnum=1)(x, theta)
print(f"Gradient dθ₀: {grad[0]:.4f}")
print(f"Gradient dθ₁: {grad[1]:.4f}")
```

---

## 3. Stochastic parameter-shift

Pour des portes qui ne sont pas de simples générateurs de Pauli ($G^2 \neq I$), on utilise la **stochastic parameter-shift rule** :

$$
\frac{\partial f(\theta)}{\partial \theta} = \sum_{k} c_k \, f(\theta + s_k)
$$

où les $c_k$ et $s_k$ dépendent des valeurs propres de $G$. Pour $G = \sum_j \lambda_j |\lambda_j\rangle\langle\lambda_j|$ :

$$
\frac{\partial f(\theta)}{\partial \theta} = \sum_j \frac{\lambda_j}{2} \, f\left(\theta + \frac{\pi}{2\lambda_j}\right)
$$

---

## 4. Adjoint differentiation

### 4.1 Principe

La **adjoint differentiation** (ou *adjoint backpropagation*) calcule le gradient en "propagation arrière" à travers le circuit, similaire à la rétropropagation classique [VQA26].

Pour un circuit $U = U_L \cdots U_1$, on calcule le gradient de $\langle O \rangle$ en :

1. **Forward pass** : appliquer $U$ à $|0\rangle$ pour obtenir $|\psi\rangle$
2. **Backward pass** : pour chaque $U_\ell$, calculer $\partial \langle O \rangle / \partial \theta_\ell$ en utilisant l'état $|\psi\rangle$ et un état "adjoint" $|\omega\rangle = O |\psi\rangle$

### 4.2 Complexité

| Méthode | Évaluations du circuit | Mémoire |
|---------|----------------------|---------|
| Parameter-shift | $2 \times p$ | $O(1)$ |
| Adjoint | $1$ | $O(p \times n)$ |
| Finite differences | $p + 1$ | $O(1)$ |

L'adjoint differentiation nécessite un simulateur (pas de matériel réel) car il faut stocker les états intermédiaires.

### 4.3 Utilisation PennyLane

Par défaut, PennyLane utilise parameter-shift sur le matériel et adjoint differentiation sur simulateur :

```python
@qml.qnode(dev, diff_method="adjoint")
def circuit_adjoint(x, theta):
    qml.AngleEmbedding(x, wires=range(2))
    qml.BasicEntanglerLayers(theta, wires=range(2))
    return qml.expval(qml.Z(0))
```

---

## 5. Finite differences (approximation)

La méthode des **différences finies** approxime le gradient sans connaissance de la structure du circuit :

$$
\frac{\partial f(\theta)}{\partial \theta} \approx \frac{f(\theta + \varepsilon) - f(\theta - \varepsilon)}{2\varepsilon} + O(\varepsilon^2)
$$

### 5.1 Implémentation

```python
def finite_diff_gradient(circuit, theta, eps=1e-4):
    """Gradient par différences finies centrées."""
    grad = np.zeros_like(theta)
    f_center = circuit(theta)
    
    for k in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[k] += eps
        theta_minus = theta.copy()
        theta_minus[k] -= eps
        
        grad[k] = (circuit(theta_plus) - circuit(theta_minus)) / (2 * eps)
    
    return grad
```

### 5.2 Inconvénients

- **Approximation** : erreur de troncature $O(\varepsilon^2)$ + erreur d'arrondi $O(1/\varepsilon)$
- **Bruit** : sur matériel NISQ, le bruit de mesure rend l'approximation imprécise
- **Nombre d'évaluations** : $2p$ évaluations (similaire à parameter-shift mais moins précis)

---

## 6. Intégration PyTorch / TensorFlow / JAX

### 6.1 Interface PyTorch

```python
import torch
import pennylane as qml

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def circuit_torch(x, theta):
    qml.RX(x[0], wires=0)
    qml.RY(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta[1], wires=1)
    return qml.expval(qml.Z(0))

# Utilisation avec PyTorch
x = torch.tensor([0.5], requires_grad=False)
theta = torch.tensor([0.3, 1.2], requires_grad=True)

output = circuit_torch(x, theta)
output.backward()

print(f"Gradient θ₀: {theta.grad[0]:.4f}")
print(f"Gradient θ₁: {theta.grad[1]:.4f}")
```

### 6.2 Interface TensorFlow

```python
import tensorflow as tf
import pennylane as qml

@qml.qnode(dev, interface="tf", diff_method="parameter-shift")
def circuit_tf(x, theta):
    qml.RX(x[0], wires=0)
    qml.RY(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta[1], wires=1)
    return qml.expval(qml.Z(0))

x = tf.Variable([0.5])
theta = tf.Variable([0.3, 1.2])

with tf.GradientTape() as tape:
    output = circuit_tf(x, theta)

grad = tape.gradient(output, theta)
```

### 6.3 Interface JAX

```python
import jax
import jax.numpy as jnp
import pennylane as qml

@qml.qnode(dev, interface="jax", diff_method="parameter-shift")
def circuit_jax(x, theta):
    qml.RX(x[0], wires=0)
    qml.RY(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta[1], wires=1)
    return qml.expval(qml.Z(0))

x = jnp.array([0.5])
theta = jnp.array([0.3, 1.2])

grad = jax.grad(circuit_jax, argnums=1)(x, theta)
```

---

## 7. Comparaison des méthodes

### 7.1 Tableau comparatif

| Méthode | Exact | Évaluations/circuit | Bruit | Implémentation |
|---------|-------|---------------------|-------|----------------|
| **Parameter-shift** | Oui | $2p$ | Robuste | PennyLane, Qiskit |
| **Adjoint** | Oui | $1$ | N/A (simulateur) | PennyLane (diff_method="adjoint") |
| **Stochastic PS** | Oui | $2p$ à $4p$ | Robuste | PennyLane (avancé) |
| **Finite differences** | Non | $p+1$ | Sensible | Manuel |
| **SPSA** | Non (approx.) | $2$ par étape | Robuste | `qml.SPSAOptimizer` |

### 7.2 Nombre d'évaluations

Pour un circuit à $p$ paramètres :

$$
\text{Parameter-shift} : 2p \text{ évaluations}
$$
$$
\text{Adjoint} : 1 \text{ forward} + 1 \text{ backward} = O(p) \text{ opérations}
$$
$$
\text{SPSA} : 2 \text{ évaluations} \times \text{iterations}
$$

### 7.3 Précision sur matériel NISQ

Sur un processeur NISQ avec bruit de mesure :

$$\text{Var}[\text{parameter-shift}] = \frac{\text{Var}[f(\theta + \pi/2)] + \text{Var}[f(\theta - \pi/2)]}{4} = \frac{\sigma_{\text{shot}}^2}{2}$$

où $\sigma_{\text{shot}}^2$ est la variance due au nombre fini de tirs de mesure. La parameter-shift rule est donc **2 fois plus coûteuse en tirs** que l'évaluation directe pour la même précision.

---

## 8. Exercices

1. **Dérivée manuelle** : Calculez $\partial \langle Z \rangle / \partial \theta$ pour $U(\theta) = R_X(\theta) R_Y(\pi/2)$ et $|0\rangle$. Vérifiez avec parameter-shift.
2. **Comparaison** : Implémentez les 4 méthodes (parameter-shift, adjoint, finite diff, SPSA) pour un même circuit $n=4$, $L=3$. Comparez le temps d'exécution et la précision.
3. **Gradient d'une porte contrôlée** : Montrez que la parameter-shift rule s'applique aussi à $CR_X(\theta)$. Quel est le générateur ?

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. — §5.4.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [Pen26] Xanadu. « Why quantum computers could be great for machine learning after all. » *PennyLane Blog*, Mars 2026.
- [NOM26] Li, K. et al. « Learning parameterized quantum circuits with quantum gradient. » *npj Quantum Information* 12, 59 (2026).
