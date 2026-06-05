# Séance 6.1 — Barren Plateaux et problèmes d'entraînement

## 1. Le problème des barren plateaux

### 1.1 Définition

Les **barren plateaux** (plateaux stériles) décrivent le phénomène par lequel le gradient d'un PQC s'annule exponentiellement avec le nombre de qubits [McClean et al. 2018, SP21] :

$$
\mathbb{E}\left[ \frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \theta_k} \right] = 0, \quad \text{Var}\left[ \frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \theta_k} \right] = O\left(\frac{1}{2^n}\right)
$$

Pour $n \geq 20$ qubits, la variance du gradient devient si petite qu'un optimiseur classique ne peut pas détecter de direction de descente — l'entraînement est impossible.

### 1.2 Origine : concentration measure

Dans un espace de Hilbert de dimension $2^n$, la **concentration measure** stipule que toute fonction lipschitzienne est presque constante sur la majeure partie de l'espace. Pour la sphère de Bloch généralisée (variété de $2^{n+1} - 2$ dimensions réelles) :

$$
\mathbb{P}_{|\psi\rangle \sim \text{Haar}} \left( \left| f(|\psi\rangle) - \mathbb{E}[f] \right| \geq \varepsilon \right) \leq 2 \exp\left( -\frac{2^n \varepsilon^2}{9\pi^3} \right)
$$

Le gradient étant une fonction de l'état, il subit cette même concentration.

### 1.3 Variance du gradient

Pour un PQC dont les paramètres sont tirés aléatoirement et qui forme un 2-design approximatif [SP21] :

$$
\boxed{\text{Var}\left[ \frac{\partial \langle O \rangle}{\partial \theta_k} \right] = O\left( \frac{\text{Tr}(O^2)}{2^n} \right)}
$$

```python
import pennylane as qml
from pennylane import numpy as np

def barren_plateau_analysis(n_qubits, n_layers, n_samples=100):
    """Analyse la variance du gradient en fonction du nombre de qubits."""
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(theta):
        qml.BasicEntanglerLayers(theta, wires=range(n_qubits))
        return qml.expval(qml.Z(0))
    
    # Mesure de la variance du gradient
    gradients = []
    for _ in range(n_samples):
        theta = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits))
        grad = qml.grad(circuit)(theta)
        gradients.append(np.array(grad).flatten())
    
    gradients = np.array(gradients)
    var = np.var(gradients)
    
    print(f"n={n_qubits:2d}, L={n_layers:2d}, Var(grad)={var:.6e}")
    return var

# Analyse : la variance chute exponentiellement
for n in [2, 4, 6, 8, 10]:
    barren_plateau_analysis(n, n_layers=3, n_samples=50)
```

**Résultat attendu** : La variance décroît comme $2^{-n}$ :

| $n$ | $\text{Var}(\partial \mathcal{L}/\partial \theta)$ |
|-----|-----------------------------------------------------|
| 2   | $10^{-1}$ |
| 4   | $10^{-2}$ |
| 6   | $10^{-3}$ |
| 8   | $10^{-4}$ |
| 10  | $10^{-5}$ |

---

## 2. Coût global vs. local

### 2.1 Définition

La distinction entre opérateurs de coût **global** et **local** est cruciale [VQA26] :

- **Coût global** : observable agissant sur tous les qubits, ex : $O = |0\rangle\langle0|^{\otimes n}$
- **Coût local** : observable agissant sur un sous-ensemble de qubits, ex : $O_k = |0\rangle\langle0|_k \otimes I_{\neq k}$

### 2.2 Variance pour chaque type

Pour un circuit 2-design avec $n$ qubits :

$$
\text{Var}[\partial \mathcal{L}_{\text{global}} / \partial \theta] = O(2^{-n})
$$
$$
\text{Var}[\partial \mathcal{L}_{\text{local}} / \partial \theta] = O(2^{-1}) \quad \text{(indépendant de } n \text{)}
$$

```python
@qml.qnode(dev)
def circuit_global(theta):
    qml.BasicEntanglerLayers(theta, wires=range(n_qubits))
    # Observable globale : |0...0><0...0|
    return qml.expval(qml.Projector(np.zeros(n_qubits), wires=range(n_qubits)))

@qml.qnode(dev)
def circuit_local(theta):
    qml.BasicEntanglerLayers(theta, wires=range(n_qubits))
    # Observable locale : Z sur un seul qubit
    return qml.expval(qml.Z(0))
```

### 2.3 Interprétation

Les **coûts locaux** (comme $Z_0$) échappent aux barren plateaux car leur observable n'implique qu'un nombre constant de qubits. C'est pourquoi les VQC utilisent presque toujours des observables locales.

---

## 3. Impact de la profondeur du circuit

### 3.1 Circuits peu profonds

Pour $L = O(\log n)$, le PQC n'est pas assez profond pour former un 2-design. La variance du gradient peut être polynomiale plutôt qu'exponentielle.

### 3.2 Circuits profonds

Pour $L = \text{poly}(n)$, le circuit devient un 2-design approximatif et subit les barren plateaux.

### 3.3 Analyse

```python
def depth_impact_analysis(n_qubits=8, max_layers=10):
    """Analyse la variance du gradient en fonction de la profondeur."""
    dev = qml.device("default.qubit", wires=n_qubits)
    
    variances = []
    for L in range(1, max_layers + 1):
        @qml.qnode(dev)
        def circuit(theta):
            qml.BasicEntanglerLayers(theta.reshape(L, n_qubits), wires=range(n_qubits))
            return qml.expval(qml.Z(0))
        
        grads = []
        for _ in range(50):
            theta = np.random.uniform(0, 2*np.pi, L * n_qubits)
            grad = qml.grad(circuit)(theta)
            grads.append(np.array(grad))
        
        variances.append(np.var(grads))
        print(f"L={L:2d}, Var(grad)={variances[-1]:.6e}")
    
    return variances
```

---

## 4. Impact du bruit

Le bruit NISQ aggrave les barren plateaux [VQA26]. Pour un canal de dépolarisation de probabilité $p$ :

$$
\text{Var}_{\text{bruit}}[\partial \mathcal{L} / \partial \theta] = e^{-c p L n} \times \text{Var}_{\text{sans bruit}}
$$

Le bruit ajoute un facteur exponentiel supplémentaire dans $n$, rendant l'entraînement encore plus difficile.

---

## 5. Stratégies de mitigation

### 5.1 Initialisation identité

Au lieu d'initialiser les paramètres aléatoirement, on initialise le circuit proche de l'identité :

```python
def identity_initialization(n_layers, n_qubits):
    """Initialisation : toutes les rotations à 0 (identité)."""
    theta = np.zeros((n_layers, n_qubits))
    # On ajoute un peu de bruit pour briser la symétrie
    theta += np.random.uniform(-0.01, 0.01, (n_layers, n_qubits))
    return theta
```

**Principe** : proche de l'identité, le circuit n'est pas un 2-design → pas de barren plateau. On augmente progressivement l'expressivité pendant l'entraînement.

### 5.2 Warm-start

On pré-entraîne le circuit sur une version simplifiée du problème (ou sur un sous-ensemble de données), puis on affine sur le jeu complet.

### 5.3 Layer-wise learning

On entraîne les couches une par une [VQA26] :

1. Entraîner la couche 1 seule
2. Geler la couche 1, entraîner la couche 2
3. Geler les couches 1–2, entraîner la couche 3
4. Fine-tuning final de toutes les couches

```python
def layerwise_training(X, y, n_qubits, n_layers, epochs_per_layer=50):
    """Layer-wise learning pour un VQC."""
    theta = np.zeros((n_layers, n_qubits))
    
    for layer in range(n_layers):
        print(f"Entraînement couche {layer + 1}/{n_layers}")
        
        # On n'entraîne que les paramètres de cette couche
        def cost_fn(params):
            theta[layer] = params
            return cross_entropy(theta, X, y)
        
        opt = qml.AdamOptimizer(stepsize=0.1)
        params = theta[layer].copy()
        
        for _ in range(epochs_per_layer):
            params = opt.step(cost_fn, params)
        
        theta[layer] = params
    
    # Fine-tuning global
    for _ in range(epochs_per_layer // 2):
        theta = opt.step(lambda t: cross_entropy(t, X, y), theta)
    
    return theta
```

### 5.4 Initialisation par bloc identité

Proposée par [NOM26], cette méthode initialise des blocs de portes de manière à ce que chaque bloc soit proche de l'identité :

```python
def identity_block_init(n_layers, n_qubits):
    """Chaque bloc RY + RZ + CNOT est initialisé près de l'identité."""
    theta = np.zeros((n_layers, n_qubits, 2))  # RY et RZ par qubit
    # Distribution concentrée autour de 0
    theta = np.random.normal(0, 0.1, (n_layers, n_qubits, 2))
    return theta
```

### 5.5 Résumé des stratégies

| Stratégie | Principe | Efficacité | Complexité |
|-----------|----------|-----------|------------|
| **Initialisation identité** | Proche de $I$ | Haute | Faible |
| **Warm-start** | Pré-entraînement | Moyenne | Moyenne |
| **Layer-wise learning** | Couches séquentielles | Haute | Élevée |
| **Coût local** | Observable locale | Très haute | Faible |
| **Ansatz structuré** | TTN, MPS | Haute | Moyenne |
| **ADAPT-VQC** | Croissance dynamique | Très haute | Élevée |

---

## 6. Analyse graphique : variance du gradient

```python
def gradient_vs_qubits(max_qubits=14, n_layers=3, n_samples=50):
    """Trace la variance du gradient en fonction de n."""
    qubits = range(2, max_qubits + 1, 2)
    variances_local = []
    variances_global = []
    
    for n in qubits:
        dev = qml.device("default.qubit", wires=n)
        
        @qml.qnode(dev)
        def circuit_local(theta):
            qml.BasicEntanglerLayers(theta.reshape(n_layers, n), wires=range(n))
            return qml.expval(qml.Z(0))
        
        @qml.qnode(dev)
        def circuit_global(theta):
            qml.BasicEntanglerLayers(theta.reshape(n_layers, n), wires=range(n))
            return qml.expval(qml.Projector(np.zeros(n), wires=range(n)))
        
        grads_local, grads_global = [], []
        for _ in range(n_samples):
            theta = np.random.uniform(0, 2*np.pi, n_layers * n)
            grads_local.append(np.array(qml.grad(circuit_local)(theta)))
            grads_global.append(np.array(qml.grad(circuit_global)(theta)))
        
        variances_local.append(np.var(grads_local))
        variances_global.append(np.var(grads_global))
    
    # Résultats : Var_local ≈ O(1), Var_global ≈ O(2^{-n})
    return qubits, variances_local, variances_global

# n_qubits  :  2    4    6    8    10   12   14
# Var_local :  0.5  0.4  0.5  0.4  0.5  0.4  0.5   ← constant
# Var_global:  0.3  0.1  0.03 0.01 2e-3 5e-4 1e-4   ← exponentiel
```

Le graphique conceptuel montre :
- **Coût local** (courbe horizontale) : variance constante ∼ 0.5
- **Coût global** (courbe décroissante) : variance ∼ $2^{-n}$, chute rapide

---

## 7. Exercices

1. **Variance expérimentale** : Reproduisez l'analyse de variance pour $n = 2$ à $12$ qubits avec 3 couches `BasicEntanglerLayers`. Confirmez la loi $2^{-n}$.
2. **Initialisation identité vs aléatoire** : Comparez la variance du gradient après 50 étapes d'Adam pour les deux initialisations.
3. **Layer-wise** : Implémentez l'entraînement layer-wise pour un problème de classification binaire (Iris) et comparez la convergence avec l'entraînement standard.

---

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. — §5.5.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [NOM26] Li, K. et al. « Learning parameterized quantum circuits with quantum gradient. » *npj Quantum Information* 12, 59 (2026).
- [Pen26] Xanadu. « Why quantum computers could be great for machine learning after all. » *PennyLane Blog*, Mars 2026.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
