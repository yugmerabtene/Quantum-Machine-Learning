# Devoir 2 — VQC et Barren Plateaux

**Cours** : Quantum Machine Learning (Master/PhD)
**Semaine** : 6 | **À rendre** : Semaine 8
**Poids** : 7,5 % de la note finale

---

## 1. Problèmes théoriques

### Problème 1 — Phénomène de Barren Plateau

#### Question 1.1
Expliquer le phénomène de **barren plateau** dans l'entraînement des circuits quantiques variationnels (VQC). En quoi diffère-t-il d'un simple problème de gradient qui s'annule ?

**Indice** : référez-vous à l'article fondateur de McClean et al. (2018), *Barren plateaus in quantum neural network training landscapes*.

#### Question 1.2
Soit un circuit quantique aléatoire à $$n$$ qubits et $$L$$ couches. La fonction de coût est définie comme :

$$L(\theta) = \langle 0| U^\dagger(\theta) O U(\theta) |0 \rangle$$

où $$O$$ est un observable quelconque. Démontrer que pour un circuit suffisamment profond ($$L$$ grand), l'entraînement est exponentiellement difficile :

$$\text{Var}\left(\frac{\partial L}{\partial \theta_k}\right) = \mathcal{O}(2^{-n})$$

**Indice** : utilisez la théorie des $$t$$-designs. Un circuit aléatoire profond forme un 2-design approximatif. Montrez que $$\mathbb{E}[\partial L/\partial \theta_k] = 0$$ et que $$\mathbb{E}[(\partial L/\partial \theta_k)^2]$$ décroît exponentiellement avec $$n$$.

#### Question 1.3
Quelles sont les causes physiques du barren plateau ? Discutez le rôle de :
- L'intrication
- L'expressivité du circuit
- La profondeur $$L$$
- Le nombre de qubits $$n$$

---

### Problème 2 — Coût global vs. local

#### Question 2.1
On considère deux fonctions de coût pour un VQC sur $$n$$ qubits :

**Coût global** : $$L_{\text{global}}(\theta) = \langle \psi(\theta) | O_{\text{global}} | \psi(\theta) \rangle$$ avec $$O_{\text{global}} = |0\rangle\langle 0|^{\otimes n}$$

**Coût local** : $$L_{\text{local}}(\theta) = \frac{1}{n} \sum_{i=1}^n \langle \psi(\theta) | O^{(i)}_{\text{local}} | \psi(\theta) \rangle$$ avec $$O^{(i)}_{\text{local}} = |0\rangle\langle 0|_i \otimes I_{\bar{i}}$$

Expliquer pourquoi :
1. Le coût global subit un barren plateau plus sévère que le coût local.
2. La variance du gradient pour le coût local décroît comme $$\mathcal{O}(2^{-n} + 1/n)$$ alors que le coût global décroît comme $$\mathcal{O}(2^{-n})$$.

**Indice** : analysez la concentration de la mesure. Pour le coût global, l'observable est globale et sa variance se concentre exponentiellement. Pour le coût local, chaque terme est une mesure partielle.

#### Question 2.2
Proposer une stratégie de mitigation basée sur ce constat. Quel type de fonction de coût recommandez-vous pour un VQC sur plus de 8 qubits ?

---

### Problème 3 — Parameter-Shift Rule

#### Question 3.1
Démontrer la **parameter-shift rule** pour une porte $$R_Y(\theta)$$ :

$$\frac{\partial}{\partial \theta} \langle \psi(\theta) | O | \psi(\theta) \rangle = \frac{1}{2} \left[ \langle \psi(\theta + \pi/2) | O | \psi(\theta + \pi/2) \rangle - \langle \psi(\theta - \pi/2) | O | \psi(\theta - \pi/2) \rangle \right]$$

**Indice** : commencez par écrire $$R_Y(\theta) = e^{-i \theta Y/2}$$. Utilisez la relation de commutation et le fait que $$Y^2 = I$$. Comparez cette méthode avec les différences finies.

#### Question 3.2
Quels sont les avantages de la parameter-shift rule par rapport à la différenciation par différences finies ?
- Précision
- Bruit
- Nombre d'évaluations de circuit
- Compatibilité avec le matériel quantique

---

### Problème 4 — ULSP (Unfavorable Local Stationary Points)

#### Question 4.1
Qu'est-ce qu'un ULSP (Unfavorable Local Stationary Point) ? En quoi diffère-t-il d'un barren plateau ?

**Indice** : contrairement au barren plateau où le gradient est nul en moyenne sur tout le paysage, un ULSP est un point stationnaire local qui n'est pas un minimum. Voir [NOM26] — Nested Optimization Model.

#### Question 4.2
Décrire la méthode **NOM (Nested Optimization Model)** proposée pour échapper aux ULSP. Quel est le rôle du quantum gradient dans cette méthode ?

---

## 2. Problèmes d'implémentation

### Problème 5 — Variance du gradient en fonction du nombre de qubits

#### Question 5.1
Implémenter un VQC générique avec PennyLane. Structure du circuit :

```python
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

def random_vqc(n_qubits, n_layers=3):
    """Génère un VQC avec des portes aléatoires."""
    def circuit(params):
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[l, i, 0], wires=i)
                qml.RZ(params[l, i, 1], wires=i)
            # Couche d'intrication
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))
    return circuit
```

Pour $$n_{\text{qubits}} \in \{2, 4, 6, 8, 10, 12\}$$ :
1. Initialiser les paramètres aléatoirement (distribution uniforme sur $$[0, 2\pi]$$).
2. Calculer le gradient $$\partial L / \partial \theta_k$$ pour chaque paramètre via la parameter-shift rule (ou `qml.grad`).
3. Calculer la variance du gradient sur 100 initialisations aléatoires.
4. Produire un graphique $$\log(\text{Var}(\nabla L))$$ vs. $$n_{\text{qubits}}$$.

**Résultat attendu** : la variance décroît exponentiellement avec $$n$$ (pente négative en échelle semi-log).

```python
# Structure attendue
n_qubits_list = [2, 4, 6, 8, 10, 12]
variances = []

for n in n_qubits_list:
    # Créer le device et le QNode
    dev = qml.device("default.qubit", wires=n)
    qnode = qml.QNode(random_vqc(n), dev)
    
    # Mesurer la variance du gradient sur N samples
    grads = []
    for _ in range(100):
        params = np.random.uniform(0, 2*np.pi, size=(3, n, 2))
        grad = qml.grad(qnode)(params)
        grads.append(...)
    
    variances.append(np.var(grads))

# Graphique
plt.semilogy(n_qubits_list, variances, 'o-')
plt.xlabel("Nombre de qubits")
plt.ylabel("Var(gradient)")
plt.title("Barren plateau : variance du gradient")
plt.show()
```

#### Question 5.2
Interpréter le graphique. La variance décroît-elle comme $$\mathcal{O}(2^{-n})$$ ? Vérifier en traçant la droite théorique $$y \propto 2^{-n}$$ en superposition.

---

### Problème 6 — Initialisation identity vs. aléatoire

#### Question 6.1
Au lieu d'initialiser les paramètres aléatoirement, utiliser une initialisation **identity** : tous les angles à 0 (ou proches de 0, e.g. $$\mathcal{N}(0, 0.01)$$). Refaire l'expérience du problème 5.

```python
def identity_init(n_qubits, n_layers=3):
    """Paramètres proches de zéro."""
    return np.random.normal(0, 0.01, size=(n_layers, n_qubits, 2))
```

#### Question 6.2
Comparer les deux courbes de variance (init aléatoire vs. identity). L'initialisation identity atténue-t-elle le barren plateau ? Pourquoi ?

**Indice** : l'initialisation identity maintient le circuit proche de l'identité, réduisant l'expressivité et donc la concentration. Voir Grant et al. (2019), *An initialization strategy for addressing barren plateaus in parametrized quantum circuits*.

---

### Problème 7 — Bonus : ADAPT-VQC simple (optionnel)

#### Question 7.1
Implémenter un ADAPT-VQC (Adaptive VQC) qui ajoute dynamiquement des couches :

```python
def adapt_vqc(X, y, max_layers=10, threshold=1e-3):
    """
    Ajoute une couche à la fois tant que le gradient moyen
    reste inférieur à 'threshold'.
    
    Retourne : modèle entraîné, historique de loss
    """
    n_qubits = X.shape[1]
    n_layers = 0
    history = []
    
    while n_layers < max_layers:
        n_layers += 1
        # Construire le circuit avec n_layers couches
        # Entraîner (quelques epochs)
        # Mesurer le gradient moyen
        # Si gradient > threshold : continuer
        # Sinon : arrêter
    
    return model, history
```

#### Question 7.2
Tester sur le dataset Iris (binaire : classe 0 vs. 1). Le nombre de couches optimal est-il corrélé à la complexité de la tâche ?

---

## 3. Pistes et ressources

- **McClean et al. (2018)** : *Barren plateaus in quantum neural network training landscapes*, Nature Communications 9, 4812
- **Grant et al. (2019)** : *An initialization strategy for addressing barren plateaus in parametrized quantum circuits*, Quantum 3, 214
- **NOM — Li et al. (2026)** : *Learning parameterized quantum circuits with quantum gradient*, npj Quantum Information 12, 59
- **PennyLane gradient documentation** : https://docs.pennylane.ai/en/stable/introduction/interfaces.html
- **Schuld & Petruccione (2021)** : Chapitre 5 — Variational quantum circuits
