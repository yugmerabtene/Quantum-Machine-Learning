# Séance 1.1 — Introduction historique et panorama du Quantum Machine Learning

## De l'intelligence artificielle classique au calcul quantique

L'histoire de l'intelligence artificielle (IA) est ponctuée de cycles d'optimisme et de désillusion. Depuis les premiers perceptrons (Rosenblatt, 1958) jusqu'à la révolution de l'apprentissage profond (Krizhevsky et al., 2012), chaque avancée majeure a été portée par une augmentation de la puissance de calcul disponible. La loi de Moore, qui prédisait un doublement de la densité de transistors tous les deux ans, a soutenu cette progression pendant cinq décennies. Cependant, la fin de cette loi — annoncée par les limites physiques de la lithographie — impose une redirection fondamentale : l'informatique classique atteint un plafond énergétique et physique.

C'est dans ce contexte que le calcul quantique émerge comme une alternative paradigmatique. Plutôt que de miniaturiser davantage des bits classiques, l'ordinateur quantique exploite les principes de superposition et d'intrication pour représenter et manipuler l'information. Le *Quantum Machine Learning* (QML) se situe à l'intersection de ces deux révolutions : il propose d'utiliser les propriétés de la mécanique quantique pour améliorer — ou repenser — les algorithmes d'apprentissage automatique.

## Pourquoi le quantique pour le Machine Learning ?

L'hypothèse centrale du QML est que certains calculs fondamentaux pour l'apprentissage — produits scalaires en haute dimension, échantillonnage de distributions complexes, optimisation non convexe — peuvent bénéficier d'un avantage quantique. En particulier, les espaces de Hilbert quantiques, dont la dimension croît exponentiellement avec le nombre de qubits, offrent un espace de représentation d'une richesse inaccessible classiquement.

Un argument clé est celui de la *feature map quantique* : une donnée classique $x \in \mathbb{R}^d$ peut être encodée dans un état quantique $|\psi(x)\rangle$ dans un espace de Hilbert $\mathcal{H}$ de dimension $2^n$. Le produit scalaire quantique $\langle \psi(x) | \psi(x') \rangle$ — accessible par mesure — agit comme un noyau (kernel) potentiellement plus expressif que les noyaux classiques (RBF, polynomial). Formellement, on définit :

$$
k_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2
$$

Ce *fidelity quantum kernel* est au cœur des méthodes à noyau quantique [Hav19].

### La boucle hybride classique-quantique

Les dispositifs quantiques actuels sont dits NISQ (*Noisy Intermediate-Scale Quantum*). Ils comportent de 50 à 1000 qubits, mais sans correction d'erreur complète. Dans ce régime, l'architecture dominante est celle de la *boucle hybride* :

```
Données → [Classique] → paramètres → [Circuit quantique] → mesure → [Classique] → perte → gradient → ...
```

Le circuit quantique est utilisé comme un module différentiable au sein d'un pipeline classique. L'optimisation des paramètres du circuit se fait par descente de gradient classique, utilisant la *parameter-shift rule* pour estimer le gradient quantique. Cette architecture est implémentée dans PennyLane [Pen26] :

```python
import pennylane as qml
import torch

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(x, params):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.BasicEntanglerLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

x = torch.tensor([0.1, 0.2, 0.3, 0.4])
params = torch.randn((3, n_qubits), requires_grad=True)
output = circuit(x, params)
```

Cette boucle est au cœur de tous les modèles variationnels (VQC, VQE, QAOA) et constitue le paradigme dominant du QML en 2026.

### La parameter-shift rule en détail

Le calcul du gradient dans la boucle hybride repose sur la *parameter-shift rule*, résultat fondamental des circuits quantiques paramétés. Pour une observable $O$ et un paramètre $\theta$ apparaissant dans une porte $U(\theta) = e^{-i\theta H/2}$ avec $H$ un Pauli, le gradient exact est :

$$
\frac{\partial}{\partial \theta} \langle O \rangle_\theta = \frac{1}{2} \left[ \langle O \rangle_{\theta + \pi/2} - \langle O \rangle_{\theta - \pi/2} \right]
$$

Cette formule est exacte (pas d'approximation numérique) et requiert deux évaluations du circuit par paramètre. Pour un circuit à $p$ paramètres, une étape de gradient coûte donc $2p$ évaluations, ce qui constitue un surcoût significatif mais garantit des gradients sans biais.

## État des lieux 2026 : chiffres et avancées

L'année 2026 marque un tournant pour le QML, avec des avancées significatives tant sur le plan matériel qu'algorithmique.

### Marché et investissements

- Le marché global du calcul quantique est estimé à **~2,5 milliards USD** en 2026, dont environ **1,4 milliard USD** pour les applications QML specifically (McKinsey Quantum Technology Monitor, 2026). Le taux de croissance annuel composé (TCAC) est de **35 %** sur la période 2024–2030.
- Les investissements privés ont dépassé **5 milliards USD** cumulés depuis 2019, avec les secteurs **finance** (optimisation de portefeuille, pricing d'options), **pharma** (discovery de molécules, docking protéique) et **défense** (cryptographie, logistique) comme principaux moteurs.
- Les principaux acteurs QML sont **IBM** (Qiskit ML, 156+ qubits), **Google** (Cirq + Willow, 105 qubits), **Xanadu** (PennyLane, photonic QPU), **IonQ** (trapped ions, 36+ qubits algorithmiques) et **Rigetti** (Ankaa-3, 84 qubits).

### Matériel : qubits, fidélité et connectivité

| Plateforme | Qubes (2026) | Fidélité 2-qubits | Connectivité |Correction d'erreur |
|---|---|---|---|---|
| IBM Heron (ibm_marrakesh) | 156 | ~99,5 % | Heavy-hex | Code de surface (démo) |
| Google Willow | 105 | ~99,7 % | Grille 2D | Correction exponentielle |
| IonQ Forte Enterprise | 36 (algorithmique) | ~99,4 % | Tout-à-tout | En développement |
| Xanadu Borealis | 216 modes | ~99 % (photonique) | Programmable | N/A (mesure seule) |
| Quantinuum H2 | 56 | ~99,8 % | Tout-à-tout | Circuits logiques démo |

La **fidélité moyenne des portes 2-qubits** a progressé de ~95 % (2019) à ~99,7 % (2026) sur les meilleures plateformes. Le seuil de **99 %** est désormais considéré comme le minimum pour des circuits variationnels de profondeur modérée (>20 couches).

### Résultats algorithmiques majeurs

- **IBM / Agliardi et al. (2026)** : La plus grande expérience QML sur hardware réel, menée sur *ibm_marrakesh* (156 qubits). Un *covariant quantum kernel* avec *Bit-Flip Tolerance* (BFT) atteint **80 % d'accuracy** sur une tâche de classification réelle (dataset Higgs), atténuant le problème de concentration exponentielle [Agl26].
- **Xanadu** : Résultats sur les *méthodes spectrales quantiques*, montrant que la transformée de Fourier quantique peut être utilisée pour le ML avec un avantage prouvé dans certains régimes de données [Pen26].
- **Google Willow** : Le processeur de 105 qubits a démontré une réduction exponentielle du taux d'erreur lors de l'augmentation du code de correction d'erreur (seuil de seuil franchi), ouvrant la voie au régime *Early Fault-Tolerant* (EFT).
- **Quantum LEGO** : Le framework *Quantum LEGO Learning* [QuL26] propose une approche modulaire où des blocs classiques gelés (backbone pré-entraîné) sont combinés avec des VQC adaptatifs, simplifiant la conception d'architectures hybrides.
- **Born Machines et QGAN** : Les modèles génératifs quantiques montrent des avantages sur des distributions fortement multimodales, avec des expériences jusqu'à 50 qubits sur simulateurs bruités [SP21].

### Limites actuelles du NISQ

Malgré ces avancées, le régime NISQ impose des contraintes sévères :
- **Bruit** : Les circuits de profondeur > 50 couches sont encore peu fiables sans correction d'erreur complète.
- **Barren plateaus** : Pour $n > 20$ qubits, les gradients deviennent exponentiellement petits, rendant l'entraînement prohibitif [Mc18].
- **Concentration des kernels** : Le fidelity quantum kernel converge vers $\frac{1}{2^n}$ pour $n$ grand, rendant la matrice de Gram indistinguable d'une matrice constante [Agl26].
- **Coût d'évaluation** : Chaque entrée du kernel requiert $O(1/\epsilon^2)$ mesures pour une précision $\epsilon$, ce qui devient rapidement prohibitif.

### Tableau comparatif : algorithmes classiques vs quantiques

| Problème | Algorithme classique | Algorithme quantique | Speedup théorique | Speedup pratique (2026) |
|---|---|---|---|---|
| Recherche non structurée | $O(N)$ | Grover | $O(\sqrt{N})$ | Démontré jusqu'à ~3 qubits |
| Factorisation | Sub-exponentiel (NFS) | Shor | Exponentiel | 2048 bits hors de portée |
| Simulation quantique | $O(\exp(n))$ | Simulation variationnelle / Trotter | Exponentiel | Avantage pour chimie (VQE) |
| Problème de matrice (HHL) | $O(N)$ | HHL | $O(\log N)$ (conditionné) | Conditionné par conditionnement |
| SVM à noyau | $O(m^2 d)$ | QSVM (fidelity kernel) | $O(\log d)$ (feature map exponentielle) | Limité par concentration |
| Optimisation combinatoire | Heuristiques (SA, GA) | QAOA | Pas de preuve formelle | Pas d'avantage démontré |
| Réseau de neurones | $O(N_{\text{params}} \cdot m)$ | VQC + parameter-shift | Pas de preuve formelle | Compétitif sur petits datasets |
| Échantillonnage | MCMC | Born Machine / QAOA sampling | Avantage quantique potentiel | Avantage sur distributions multimodales |
| PCA | $O(d^2 m)$ | Quantum PCA | Exponentiel | Nécessite mémoire quantique |

**Observation clé** : Le speedup théorique est conditionné par des hypothèses fortes (préparation d'état efficace, accès quantique à la mémoire, faible conditionnement). En 2026, aucun avantage quantique inconditionnel et pratique n'a été démontré pour une tâche ML réelle.

### Hiérarchie des avantages quantiques en ML

On peut distinguer plusieurs niveaux d'avantage quantique, du plus faible au plus fort :

1. **Avantage pratique (heuristique)** : Un modèle quantique atteint une meilleure performance qu'un modèle classique comparable (en accuracy, temps d'entraînement, etc.) sur un dataset réel. C'est le niveau le plus modeste — et le plus contesté, car il est difficile de garantir un comparatif équitable.

2. **Avantage asymptotique** : Le modèle quantique a une complexité algorithmique meilleures ($O(\log n)$ vs $O(n)$, par exemple). C'est le cas théorique de HHL pour les systèmes linéaires, mais les conditions (conditionnement, sparsité, préparation d'état) restreignent la portée pratique.

3. **Avantage de complexité** : Il existe une preuve qu'aucun algorithme classique ne peut résoudre le problème en temps polynomial, tandis que l'algorithme quantique le peut. C'est le « Graal » du QML — non atteint en 2026 pour une tâche ML.

4. **Avantage de démonstration (benchmarking)** : Sur des benchmarks standardisés, un modèle quantique surpasse les baselines classiques de manière reproductible. Les résultats d'IBM (80 % accuracy avec covariant kernel) s'inscrivent dans cette catégorie.

## Exemple complet : boucle hybride avec entraînement

L'exemple suivant illustre une boucle hybride complète avec PennyLane : un circuit variationnel est entraîné pour classifier des données 2D, avec visualisation de la frontière de décision.

```python
import pennylane as qml
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. Préparation des données ---
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = 2 * y - 1  # Convertir en {-1, +1}
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- 2. Définition du circuit quantique ---
n_qubits = 2
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

def layer(params, wires):
    """Couche variationnelle : rotations + intrication."""
    for i in wires:
        qml.RY(params[i], wires=i)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[-1], wires[0]])

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(x, params):
    """Circuit variationnel complet : encodage + couches paramétrées."""
    qml.AngleEmbedding(x, wires=range(n_qubits))
    for layer_params in params:
        layer(layer_params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# --- 3. Fonction de perte et entraînement ---
def loss_fn(params, X, y):
    """Perte hinge régularisée."""
    predictions = np.array([circuit(x, params) for x in X])
    return np.mean(np.maximum(0, 1 - y * predictions)) + 0.01 * np.sum(params ** 2)

def accuracy(params, X, y):
    predictions = np.array([circuit(x, params) for x in X])
    return np.mean(np.sign(predictions) == y)

# Initialisation
params = np.random.randn(n_layers, n_qubits) * 0.1
lr = 0.1
epochs = 50

# Entraînement par descente de gradient (parameter-shift)
for epoch in range(epochs):
    grad = qml.grad(loss_fn)(params, X_train, y_train)
    params -= lr * grad
    if epoch % 10 == 0:
        train_loss = loss_fn(params, X_train, y_train)
        train_acc = accuracy(params, X_train, y_train)
        test_acc = accuracy(params, X_test, y_test)
        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")

# --- 4. Visualisation de la frontière de décision ---
xx, yy = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, np.pi, 50))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([np.sign(circuit(x, params)) for x in grid]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=['blue', 'red'])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k', s=30)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Frontière de décision du VQC (Make Moons)')
plt.tight_layout()
plt.savefig('vqc_decision_boundary.png', dpi=150)
plt.show()
```

Cet exemple démontre le pipeline complet QML : (1) préparation des données avec mise à l'échelle dans $[0, \pi]$, (2) encodage par `AngleEmbedding`, (3) couches variationnelles paramétrées, (4) mesure d'une observable, (5) optimisation par *parameter-shift rule*, et (6) visualisation. La loss hinge est naturelle pour un classifieur à marge, inspirée du SVM.

## Quand le QML est-il pertinent ? Limitations actuelles

### Critères de pertinence

Le QML n'est pas universellement supérieur au ML classique. Son intérêt potentiel se situe dans les régimes suivants :

1. **Données intrinsèquement quantiques** : Si les données proviennent d'un système quantique (états de matière, simulations moléculaires), le traitement quantique évite le *quantum data bottleneck* — la conversion classique-quantique coûteuse.

2. **Espaces de features exponentiels** : Les fidelity quantum kernels exploitent un espace de dimension $2^n$ avec un coût d'évaluation en $O(n)$ qubits. Classiquement, un noyau de même dimensionnalité nécessiterait $O(2^n)$ opérations.

3. **Données de haute dimension avec structure exploitable** : Si la structure des données est compatible avec la feature map quantique (symétries du circuit, propriétés de covariance), le quantum kernel peut concentrer l'information utile.

4. **Ressources limitées pour l'entraînement** : Certains résultats théoriques montrent que les circuits quantiques peu profonds peuvent apprendre avec moins d'échantillons que les réseaux classiques de même taille effective [SP21].

### Limitations majeures

- **Barren plateaux** : Pour $n \gtrsim 20$ qubits et des circuits profond ou fortement intriqués, la variance du gradient décroît exponentiellement : $\text{Var}[\partial_\theta \mathcal{L}] \in O(1/2^n)$ [Mc18]. Cela rend l'optimisation pratiquement impossible.

- **Concentration des kernels** : Le fidelity kernel $k_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$ tend vers $1/2^n$ pour les circuits expressifs, rendant la matrice de Gram indistinguable d'une matrice constante. Les covariant kernels et les stratégies de BFT sont des solutions partielles [Agl26].

- **Bruit matériel** : En régime NISQ, le bruit des portes et la décohérence limitent la profondeur des circuits exécutables. Le *error mitigation* reste coûteux en ressources.

- **Coût de l'évaluation** : Chaque entrée du kernel quantique nécessite $O(1/\epsilon^2)$ mesures pour une précision $\epsilon$. Pour une matrice $m \times m$, le coût total est $O(m^2/\epsilon^2)$ tirs, ce qui peut dépasser le coût classique.

- **Absence d'avantage quantique prouvé et inconditionnel** : En 2026, aucun théorème ne garantit un avantage quantique pour une tâche ML concrète et réaliste. Les speedups existent sous des hypothèses restrictives (accès quantique à la donnée, circuits spécifiques).

### La question centrale

Le QML en 2026 se situe dans une tension fondamentale entre **expressivité exponentielle** (l'espace de Hilbert offre $2^n$ dimensions) et **trainabilité** (les gradients et les kernels se concentrent). L'enjeu de la recherche actuelle est de concevoir des architectures qui exploiteraient l'expressivité quantique sans tomber dans ces pièges — c'est l'objet des séances 5-6 (barren plateaux) et 7-8 (kernels quantiques).

## Taxonomie du QML

On peut classer les approches QML en quatre grandes familles :

### 1. Quantum Kernel Methods

Les méthodes à noyau quantique remplacent le noyau classique d'un SVM par un noyau calculé sur un dispositif quantique. Le *fidelity kernel* $k_Q(x, x')$ est central. L'avantage espéré est un *feature space* plus riche. Inconvénient : le problème de concentration exponentielle pour $n$ grand [Agl26].

### 2. Variational Quantum Circuits (VQC)

Les circuits variationnels sont des modèles paramétrés entraînés par descente de gradient hybride. Ils sont utilisés pour la classification (VQC), l'optimisation (QAOA), et la chimie quantique (VQE). La *parameter-shift rule* permet le calcul exact du gradient :

$$
\frac{\partial f(\theta)}{\partial \theta} = \frac{1}{2} \left[ f\left(\theta + \frac{\pi}{2}\right) - f\left(\theta - \frac{\pi}{2}\right) \right]
$$

### 3. Architectures hybrides

Les architectures hybrides combinent des réseaux de neurones classiques (CNN, Transformers) avec des couches quantiques. Le transfer learning quantique, où un backbone CNN pré-entraîné est suivi d'une tête VQC, atteint 97,44 % sur Hymenoptera [Tra25]. Les Quantum Transformers intègrent des VQC dans l'attention multi-têtes [Tra26].

### 4. Modèles génératifs quantiques

Les *Born Machines*, *Quantum GAN* (qGAN) et *Quantum Boltzmann Machines* exploitent l'échantillonnage intrinsèque des états quantiques pour modéliser des distributions de probabilité. Les résultats montrent des avantages pour des distributions fortement multimodales [SP21].

## Conclusion

Le QML en 2026 est un domaine mature mais en pleine effervescence. Les fondations théoriques sont solides, les premières démonstrations à grande échelle existent (156 qubits, 80 % accuracy), mais des défis majeurs persistent : barren plateaux, concentration des kernels, bruit matériel, et absence d'avantage quantique prouvé et inconditionnel. L'objectif de ce cours est de fournir les outils mathématiques, algorithmiques et pratiques pour comprendre, implémenter et faire progresser ces méthodes.

Le chemin vers un avantage quantique pratique en ML passe probablement par trois voies complémentaires : (1) l'amélioration matérielle (fidélité, correction d'erreur), (2) la conception d'architectures prudentes (circuits peu profonds, kernels structurés, architectures hybrides), et (3) l'identification de classes de problèmes où la structure quantique apporte un avantage mesurable — un programme de recherche actif.

## Références

- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. Ch. 1.
- [Hav19] Havlíček, V. et al. « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019).
- [Agl26] Agliardi, G. et al. « Mitigating exponential concentration in covariant quantum kernels for subspace and real-world data. » *npj Quantum Information* 12, 12 (2026).
- [Pen26] Xanadu. « Why quantum computers could be great for machine learning after all. » *PennyLane Blog*, Mars 2026.
- [QuL26] « Quantum LEGO Learning: A Modular and Architecture-Agnostic Hybrid QML Framework. » *arXiv:2601.21780*, 2026.
- [Tra25] « Quantum transfer learning for image classification. » *arXiv:2603.16973*, 2025–2026.
- [Tra26] « Quantum Transformers for Image Classification. » *Quantum Machine Intelligence* 8, 43 (2026).
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [Fra26] « A Framework-Agnostic Quantum Neural Network Architecture. » *arXiv:2604.04414*, Avril 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [Mc18] McClean, J. R. et al. « Barren plateaus in quantum neural network training landscapes. » *Nature Communications* 9, 4812 (2018).
- [McKin26] McKinsey & Company. *Quantum Technology Monitor*, 2026.
