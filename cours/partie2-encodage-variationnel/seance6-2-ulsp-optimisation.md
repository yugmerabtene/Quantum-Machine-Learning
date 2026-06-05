# Séance 6.2 — ULSP et optimisation avancée

## 1. Unfavorable Local Stationary Points (ULSP)

### 1.1 Définition

Les **ULSP** (Unfavorable Local Stationary Points) sont des points stationnaires $\boldsymbol{\theta}^*$ du paysage de coût $\mathcal{L}(\boldsymbol{\theta})$ qui ne correspondent pas à des minima globaux, et dont le **gradient quantique** permet de s'échapper [NOM26].

Formellement, $\boldsymbol{\theta}^*$ est un ULSP si :

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^*) = 0 \quad \text{mais} \quad \mathcal{L}(\boldsymbol{\theta}^*) \gg \mathcal{L}(\boldsymbol{\theta}^{\text{opt}})
$$

### 1.2 Différence entre plateaux et minima locaux

| Propriété | Barren Plateau | Minimum Local | ULSP |
|-----------|---------------|---------------|------|
| **Gradient** | Nul en tout point (exponentiellement petit) | Nul au point | Nul au point |
| **Voisinage** | Gradient nul partout | Gradient non nul ailleurs | Non-nul ailleurs |
| **Détection** | Var$(\nabla \mathcal{L}) = O(2^{-n})$ | Courbure $> 0$ | Gradient quantique $\neq 0$ |
| **Origine** | Concentration measure | Structure du problème | Ansatz sous-optimal |

Les ULSP sont spécifiques aux VQC : ils apparaissent parce que l'ansatz choisi ne peut pas exprimer la solution optimale dans son sous-espace variationnel.

### 1.3 Exemple simple

Considérons un VQC à 1 qubit avec ansatz $U(\theta) = R_Y(\theta)$. On veut atteindre l'état $|1\rangle$.

$$
\mathcal{L}(\theta) = 1 - |\langle 1 | R_Y(\theta) | 0 \rangle|^2 = 1 - \sin^2(\theta/2)
$$

Le gradient $\partial \mathcal{L}/\partial \theta = -\frac{1}{2} \sin(\theta)$ s'annule en $\theta = 0, \pi, 2\pi$ :

- $\theta = 0$ : ULSP ($\mathcal{L} = 1$, état $|0\rangle$)
- $\theta = \pi$ : optimum global ($\mathcal{L} = 0$, état $|1\rangle$)

Si on ajoute une porte $R_X$, on peut échapper à l'ULSP :

```python
def ulsp_example():
    """Illustration d'un ULSP simple."""
    dev = qml.device("default.qubit", wires=1)
    
    @qml.qnode(dev)
    def circuit_simple(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.Z(0))
    
    @qml.qnode(dev)
    def circuit_extended(theta):
        qml.RY(theta[0], wires=0)
        qml.RX(theta[1], wires=0)
        return qml.expval(qml.Z(0))
    
    # θ = 0 est un ULSP pour circuit_simple
    theta_ulsp = np.array([0.0])
    print(f"Gradient en ULSP: {qml.grad(circuit_simple)(theta_ulsp)}")  # ≈ 0
    
    # Avec l'ansatz étendu, on peut s'en échapper
    theta_ext = np.array([0.0, 0.1])
    print(f"Gradient RX en ULSP: {qml.grad(circuit_extended)(theta_ext)[1]:.4f}")  # ≠ 0
```

---

## 2. Nested Optimization Model (NOM)

### 2.1 Principe

Le **Nested Optimization Model** (NOM) [NOM26] utilise un **gradient quantique** (calculé dans l'espace de Hilbert plutôt que dans l'espace des paramètres) pour détecter et échapper aux ULSP.

L'idée centrale : le gradient dans l'espace des paramètres peut être nul même si le gradient dans l'espace de Hilbert ne l'est pas. En calculant :

$$
\mathbf{g}_{\text{Hilbert}} = \nabla_{|\psi\rangle} \mathcal{L}(|\psi\rangle)
$$

on peut déterminer s'il existe une direction d'amélioration que l'ansatz actuel ne capture pas.

### 2.2 Algorithme NOM

```
1. Initialiser θ aléatoirement
2. Tant que pas convergence :
   a. Calculer ∇_θ L(θ) (parameter-shift)
   b. Si ||∇_θ L|| < ε :  # point stationnaire détecté
        - Calculer ∇_ψ L(ψ(θ))  # gradient dans l'espace de Hilbert
        - Si ||∇_ψ L|| > δ :   # ULSP détecté
             * Ajouter une porte/paramètre à l'ansatz
             * Réinitialiser l'optimisation
        - Sinon : optimum global (convergence)
   c. Mettre à jour θ (Adam, SGD, etc.)
```

### 2.3 Implémentation du Hilbert-space gradient

Le gradient dans l'espace de Hilbert mesure la sensibilité de $\mathcal{L}$ à une modification de l'état quantique $|\psi\rangle$ dans une direction orthogonale au sous-espace variationnel :

```python
def hilbert_space_gradient(circuit, theta, dev):
    """
    Calcule le gradient dans l'espace de Hilbert.
    
    Retourne la norme du gradient hors du sous-espace variationnel.
    """
    n_params = len(theta)
    n_qubits = dev.num_wires
    
    # Jacobienne : ∂|ψ(θ)⟩/∂θ_k
    jacobian = []
    for k in range(n_params):
        # Parameter-shift sur l'état (pas sur l'espérance)
        shift = np.pi / 2
        theta_plus = theta.copy()
        theta_plus[k] += shift
        theta_minus = theta.copy()
        theta_minus[k] -= shift
        
        # On récupère l'état (statevector)
        @qml.qnode(dev)
        def state_circuit(params):
            circuit(params)
            return qml.state()
        
        psi_plus = state_circuit(theta_plus)
        psi_minus = state_circuit(theta_minus)
        dpsi_dtheta = (psi_plus - psi_minus) / 2
        
        jacobian.append(dpsi_dtheta)
    
    jacobian = np.array(jacobian)
    
    # Projection sur le sous-espace variationnel
    # Le gradient Hilbertien est la composante orthogonale
    # g_Hilbert = ∇_ψ L - P_V(∇_ψ L)
    # où P_V est la projection sur le sous-espace engendré par ∂|ψ⟩/∂θ_k
    
    # Norme du gradient hors de l'espace variationnel
    # (indicateur ULSP)
    return np.linalg.norm(jacobian @ jacobian.T)
```

### 2.4 Détection adaptative des ULSP

Un ULSP est caractérisé par deux conditions simultanées :

1. **Petit gradient paramétrique** : $\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\| < \varepsilon_{\theta}$
2. **Grand gradient Hilbertien** : $\|\nabla_{|\psi\rangle} \mathcal{L}\| > \varepsilon_{\psi}$

```python
def detect_ulsp(theta, circuit, dev, eps_theta=1e-3, eps_psi=1e-2):
    """
    Détecte si θ est un ULSP.
    
    Retourne :
        True si ULSP détecté, False sinon
    """
    # Gradient paramétrique
    grad_theta = qml.grad(circuit)(theta)
    norm_grad_theta = np.linalg.norm(grad_theta)
    
    if norm_grad_theta > eps_theta:
        return False  # Pas un point stationnaire
    
    # Gradient Hilbertien
    norm_grad_hilbert = hilbert_space_gradient(circuit, theta, dev)
    
    return norm_grad_hilbert > eps_psi
```

---

## 3. ADAPT-VQE et croissance dynamique d'ansatz

### 3.1 Principe de l'ADAPT

L'approche **ADAPT** (Adaptive Derivative-Assembled Pseudo-Trotter) [NOM26] part d'un ansatz vide et ajoute des opérateurs un par un, sélectionnés par leur gradient :

- Partir de $|\psi_0\rangle = |0\rangle^{\otimes n}$
- À chaque étape, choisir l'opérateur $P_k$ (parmi un pool) dont le gradient est maximal
- Ajouter $e^{-i \theta P_k}$ au circuit
- Optimiser tous les paramètres

```python
def adapt_vqe(pool_operators, n_qubits, max_layers=20, tol=1e-4):
    """
    ADAPT-VQE simplifié.
    
    Args:
        pool_operators : liste d'opérateurs (ex: [Z0, Z1, X0Y1, ...])
        n_qubits : nombre de qubits
        max_layers : nombre maximal d'opérateurs
        tol : tolérance de convergence
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    operators = []  # opérateurs sélectionnés
    params = []
    
    for layer in range(max_layers):
        # Circuit actuel
        @qml.qnode(dev)
        def circuit(theta):
            for i, op in enumerate(operators):
                qml.exp(op, theta[i])
            return qml.expval(qml.Hamiltonian(...))  # Hamiltonien du problème
        
        if len(params) == 0:
            # Première étape : gradient des opérateurs du pool
            grad_norms = []
            for op in pool_operators:
                @qml.qnode(dev)
                def pool_circuit(theta):
                    qml.exp(op, theta)
                    return qml.expval(qml.Hamiltonian(...))
                
                grad = qml.grad(pool_circuit)(0.0)
                grad_norms.append(abs(grad))
            
            # Ajouter l'opérateur avec le plus grand gradient
            best_idx = np.argmax(grad_norms)
            operators.append(pool_operators[best_idx])
            params.append(0.0)
        
        else:
            # Optimiser tous les paramètres
            opt = qml.AdamOptimizer()
            for _ in range(100):
                params = opt.step(lambda p: circuit(p), params)
            
            # Vérifier convergence
            grad_all = qml.grad(circuit)(params)
            if np.linalg.norm(grad_all) < tol:
                break
            
            # Sélectionner le prochain opérateur
            # (par gradient résiduel)
            ...
    
    return operators, params
```

### 3.2 ADAPT pour VQC (ADAPT-VQC)

L'ADAPT-VQC adapte la même idée à la classification : on ajoute des portes au circuit jusqu'à ce que la performance de validation cesse de s'améliorer.

```python
def adapt_vqc(X_train, y_train, X_val, y_val, n_qubits=4, pool_size=10):
    """
    ADAPT-VQC : croissance dynamique du circuit.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # Pool d'opérateurs : rotations sur différents qubits
    operator_pool = []
    for i in range(n_qubits):
        for gate in [qml.RX, qml.RY, qml.RZ]:
            operator_pool.append((gate, i))
    for i in range(n_qubits - 1):
        operator_pool.append(("CNOT", i, i + 1))
    
    current_ops = []  # séquence d'opérateurs
    current_params = []
    best_val_acc = 0.0
    
    for step in range(pool_size):
        # Essayer chaque opérateur du pool
        best_op = None
        best_grad = 0.0
        
        for op in operator_pool:
            circuit_ops = current_ops + [op]
            n_params = len(current_params) + 1
            
            @qml.qnode(dev)
            def test_circuit(params):
                for i, (gate, *wires) in enumerate(circuit_ops):
                    if gate == "CNOT":
                        qml.CNOT(wires=wires)
                    else:
                        gate(params[i], wires=wires[0])
                return qml.expval(qml.Z(0))
            
            # Gradient du nouveau paramètre à θ=0
            params_extended = np.append(current_params, 0.0)
            grad = abs(qml.grad(test_circuit)(params_extended)[-1])
            
            if grad > best_grad:
                best_grad = grad
                best_op = op
        
        # Ajouter le meilleur opérateur
        current_ops.append(best_op)
        current_params = np.append(current_params, 0.0)
        
        # Optimiser tous les paramètres
        @qml.qnode(dev)
        def full_circuit(params):
            for i, (gate, *wires) in enumerate(current_ops):
                if gate == "CNOT":
                    qml.CNOT(wires=wires)
                else:
                    gate(params[i], wires=wires[0])
            return qml.expval(qml.Z(0))
        
        opt = qml.AdamOptimizer(stepsize=0.1)
        for _ in range(50):
            current_params = opt.step(
                lambda p: cost_fn(p, X_train, y_train, full_circuit),
                current_params
            )
        
        # Évaluation
        val_acc = evaluate(X_val, y_val, full_circuit, current_params)
        print(f"Step {step+1}: {len(current_ops)} ops, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        else:
            # Arrêt précoce si la performance ne s'améliore pas
            break
    
    return current_ops, current_params
```

---

## 4. Comparaison : fixed-ansatz vs. ADAPT vs. NOM

### 4.1 Tableau comparatif

| Critère | Fixed-Ansatz | ADAPT-VQC | NOM |
|---------|-------------|-----------|-----|
| **Structure** | Circuit fixe | Croissance dynamique | Détection + adaptation |
| **Paramètres** | Prédéfinis | Ajustés automatiquement | Ajustés si ULSP |
| **Barren plateaux** | Très sensible | Moins sensible (démarre petit) | Résistant |
| **ULSP** | Bloqué | Peut échapper | Détecte et échappe |
| **Coût calcul** | Faible (fixe) | Élevé (recherche de pool) | Moyen |
| **Expressivité** | Limitée par l'ansatz | Maximale (pool large) | Adaptative |
| **Convergence** | Rapide si bon ansatz | Lente (pas à pas) | Rapide (ciblée) |

### 4.2 Courbes de convergence conceptuelles

```
Perte L(θ)
│
│  Fixed-Ansatz:          converge vite mais plateau élevé (ULSP)
│  ────┐
│      └───────__ plateau
│
│  ADAPT-VQC:            converge lentement mais descend bas
│  ──┐
│    └───┐
│        └───┐
│            └─── optimum
│
│  NOM:                  détecte ULSP et s'en échappe
│  ────┐    ────┐
│      └───────┘   └─────── optimum
│
└────────────────────────── Époques
```

### 4.3 Guide de choix

| Situation | Recommandation |
|-----------|---------------|
| Problème bien compris, ansatz connu | **Fixed-ansatz** (efficace) |
| Exploration d'un nouveau problème | **ADAPT-VQC** (robuste) |
| Entraînement bloqué à un plateau élevé | **NOM** (détection ULSP) |
| Nombre de qubits $n > 10$ | **NOM + coût local** |
| Matériel NISQ (profondeur limitée) | **Fixed-ansatz** (peu profond) |

---

## 5. Exercices

1. **ULSP sur Iris** : Entraînez un VQC à 1 qubit sur Iris (2 classes). Combien de ULSP observez-vous ? Comment y échapper ?
2. **ADAPT vs. Fixed** : Comparez ADAPT-VQC et fixed-ansatz sur un problème de classification synthétique avec $n=4$ qubits. Mesurez le nombre d'époques pour atteindre 90% d'accuracy.
3. **Hilbert gradient** : Implémentez la détection d'ULSP pour un circuit à 2 qubits avec 1 couche. Vérifiez que l'ajout d'une porte réduit le gradient Hilbertien.

---

## Références

- [NOM26] Li, K. et al. « Learning parameterized quantum circuits with quantum gradient. » *npj Quantum Information* 12, 59 (2026).
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
- [Pen26] Xanadu. « Why quantum computers could be great for machine learning after all. » *PennyLane Blog*, Mars 2026.
