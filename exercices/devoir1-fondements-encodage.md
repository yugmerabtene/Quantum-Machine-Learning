# Devoir 1 — Fondements et encodage

**Cours** : Quantum Machine Learning (Master/PhD)
**Semaine** : 3 | **À rendre** : Semaine 5
**Poids** : 7,5 % de la note finale

---

## 1. Problèmes théoriques

### Problème 1 — Comparaison des stratégies d'encodage

#### Question 1.1
Soit un vecteur de features $$x \in \mathbb{R}^n$$. Démontrer que l'**Angle Encoding** avec $$n$$ qubits peut encoder au maximum un espace de features de dimension $$n$$, tandis que l'**Amplitude Encoding** avec $$n$$ qubits peut encoder jusqu'à $$2^n$$ features.

**Indice** : comptez le nombre de degrés de liberté indépendants dans l'état à $$n$$ qubits. Rappel : un état pur à $$n$$ qubits est un vecteur dans $$\mathbb{C}^{2^n}$$ de norme 1, défini à une phase globale près.

#### Question 1.2
À partir de ce constat, discuter du compromis entre les deux méthodes en termes de :
- Ressources matérielles (nombre de qubits)
- Profondeur de circuit
- Difficulté d'encodage (préparation d'état)
- Robustesse au bruit

**Indice** : pourquoi l'Amplitude Encoding est-elle difficile à implémenter sur du matériel NISQ ?

---

### Problème 2 — Circuit à 2 qubits

#### Question 2.1
Soit le circuit suivant :

```
|0⟩ — H — RY(θ) — •
                   |
|0⟩ — — — — — — — ⊕
```

1. Écrire l'état $$|\psi(\theta)\rangle$$ résultant en notation de Dirac.
2. Calculer la probabilité $$P(|0\rangle)$$ de mesurer $$|0\rangle$$ sur le premier qubit.
3. Pour quelle valeur de $$\theta$$ l'état est-il maximalement intriqué ?

**Indice** : appliquez les opérateurs pas à pas. $$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$, $$R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$.

---

### Problème 3 — No-Free-Lunch en QML

#### Question 3.1
Le théorème de No-Free-Lunch (NFL) stipule qu'aucun algorithme d'apprentissage n'est universellement meilleur qu'un autre en moyenne sur toutes les distributions de données possibles.

Expliquer pourquoi le théorème NFL est particulièrement important en QML pour :
1. La comparaison équitable entre modèles classiques et quantiques
2. Le choix du type d'encodage et d'ansatz
3. L'évaluation sur des benchmarks standardisés

**Indice** : un modèle quantique peut être performant sur une tâche mais mauvais sur une autre. Comment éviter les conclusions hâtives sur un « avantage quantique » ?

---

### Problème 4 — Noyau RBF et conditions de Mercer

#### Question 4.1
Soit le noyau RBF (Radial Basis Function) défini par :

$$K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right), \quad x, x' \in \mathbb{R}^d$$

Démontrer que $$K$$ satisfait les conditions de Mercer, c'est-à-dire qu'il existe une application $$\phi : \mathbb{R}^d \to \mathcal{H}$$ vers un espace de Hilbert $$\mathcal{H}$$ tel que $$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$.

**Indice** : utilisez la transformée de Fourier. Le noyau RBF peut s'écrire comme une intégrale sur des fonctions de base radiales — cherchez la représentation de Bochner.

#### Question 4.2
Quel est le lien entre ce résultat et la « feature map » quantique utilisée dans les circuits paramétrés ?

---

## 2. Problèmes d'implémentation

### Problème 5 — Comparaison des 4 types d'encodage (PennyLane)

#### Question 5.1
Implémenter en PennyLane les 4 types d'encodage suivants sur un point $$x = [0.5, -0.3, 0.8]$$ :

| Encodage | Classe PennyLane | Qubits nécessaires |
|----------|-----------------|-------------------|
| Angle Encoding | `qml.AngleEmbedding` | 3 |
| Basis Encoding | encodage personnalisé | 3 (après discrétisation) |
| Amplitude Encoding | `qml.AmplitudeEmbedding` | 2 (car $$2^2 = 4 \geq 3$$) |
| IQP Encoding | `qml.IQPEmbedding` | 3 |

Pour chaque encodage :

```python
import pennylane as qml
import numpy as np

x = np.array([0.5, -0.3, 0.8])

def angle_encoding_circuit(x):
    # Implémenter
    pass

def basis_encoding_circuit(x):
    # Discrétiser x en bits d'abord
    pass

def amplitude_encoding_circuit(x):
    # Normaliser x et l'encoder en amplitude
    pass

def iqp_encoding_circuit(x):
    pass
```

1. Afficher le nombre de qubits utilisé par chaque méthode.
2. Dessiner le circuit avec `qml.draw()` ou `print(qml.draw(circuit)(x))`.
3. Compter la profondeur (nombre de couches de portes).
4. Compter le nombre total de portes.

#### Question 5.2
Remplir un tableau de comparaison :

| Critère | Angle | Basis | Amplitude | IQP |
|---------|-------|-------|-----------|-----|
| Nb qubits | | | | |
| Profondeur | | | | |
| Nb portes | | | | |
| Normalisation nécessaire ? | | | | |
| Perte d'information ? | | | | |

---

### Problème 6 — Encodage et séparabilité sur Iris

#### Question 6.1
Charger le dataset Iris depuis `scikit-learn` :

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

1. Normaliser les features entre $$[0, \pi]$$ (pour l'Angle Encoding).
2. Visualiser les 3 classes dans l'espace des features classique (2D, PCA).

#### Question 6.2
Implémenter un circuit PennyLane qui :
1. Prend 2 features d'Iris (petal length, petal width).
2. Applique un Angle Encoding sur 2 qubits.
3. Mesure l'état dans la base de calcul.

```python
dev = qml.device("default.qubit", shots=1000)

@qml.qnode(dev)
def encoding_circuit(x):
    qml.AngleEmbedding(x, wires=[0, 1])
    return qml.probs(wires=[0, 1])
```

Pour chaque point du dataset, collecter les probabilités $$P(|00\rangle), P(|01\rangle), P(|10\rangle), P(|11\rangle)$$.

#### Question 6.3
1. Projeter les données dans l'espace de Hilbert en prenant les probabilités comme nouvelles features.
2. Appliquer une PCA (classique) pour visualiser en 2D la séparabilité dans l'espace quantique.
3. Comparer visuellement la séparabilité des classes dans l'espace classique vs. l'espace de Hilbert quantique.
4. Discuter : l'Angle Encoding améliore-t-il la séparabilité linéaire par rapport aux features originales ?

---

## 3. Pistes et ressources

- **PennyLane documentations** : https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html
- **Schuld & Petruccione (2021)** : Chapitre 4 — Data encoding
- **HavlÍček et al. (2019)** : *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209–212
- **Rappel théorème de Mercer** : Mohri, Rostamizadeh & Talwalkar, *Foundations of Machine Learning*, Ch. 6
- **No-Free-Lunch** : https://en.wikipedia.org/wiki/No_free_lunch_theorem
