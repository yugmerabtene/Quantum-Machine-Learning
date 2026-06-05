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

## État des lieux 2026

L'année 2026 marque un tournant pour le QML :

- **Marché** : Le marché du QML est estimé à 1,4 milliards de dollars (McKinsey), avec une croissance annuelle de 35 %. Les secteurs finance, pharma et défense sont les principaux investisseurs.
- **IBM** : La plus grande expérience QML jamais réalisée a été menée sur *ibm_marrakesh* (156 qubits) par Agliardi et al. (2026). L'étude démontre qu'un *covariant quantum kernel* avec *Bit-Flip Tolerance* (BFT) atteint 80 % d'accuracy sur une tâche de classification réelle, atténuant le problème de concentration exponentielle [Agl26].
- **Xanadu** : Les chercheurs de Xanadu ont publié des résultats sur les *méthodes spectrales quantiques*, montrant que la transformée de Fourier quantique peut être utilisée pour le ML avec un avantage prouvé dans certains régimes de données [Pen26].
- **Google** : Willow, le processeur de 105 qubits de Google, a démontré une réduction exponentielle du taux d'erreur avec la correction d'erreur, ouvrant la voie au régime EFT (*Early Fault-Tolerant*).
- **Quantum LEGO** : Le framework *Quantum LEGO Learning* [QuL26] propose une approche modulaire où des blocs classiques gelés sont combinés avec des VQC adaptatifs, simplifiant la conception d'architectures hybrides.

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
