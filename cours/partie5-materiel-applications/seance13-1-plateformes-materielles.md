# Séance 13.1 — Plateformes matérielles pour le Quantum Machine Learning

## Panorama des technologies de qubits

Cinq grandes technologies de qubits sont en compétition pour le QML, chacune avec des compromis distincts en termes de fidélité, connectivité, temps de cohérence, et scalabilité.

## Qubits supraconducteurs : Transmons

### IBM Condor — 433 qubits

Le processeur Condor d'IBM utilise des qubits *transmon* (jonctions Josephson) avec un motif en hexagone pour optimiser la connectivité. Les caractéristiques clés :

- **Temps de cohérence** : $T_1 \approx 250\,\mu\text{s}$, $T_2 \approx 150\,\mu\text{s}$
- **Fidélité des portes** : 99.9 % (1-qubit), 99.5 % (2-qubits)
- **Connectivité** : chaque qubit est connecté à 3 voisins en moyenne
- **Jeu de portes** : $\{R_X, R_Z, \text{CNOT}, \text{ECR}\}$

Pour le QML, la connectivité limitée impose un transpilation importante. Un circuit VQC avec $n$ qubits et des portes à 2 qubits arbitraires subit un overhead de profondeur $O(\sqrt{n})$ sur une topologie en grille.

### Google Willow — 105 qubits

Le processeur Willow a démontré une réduction exponentielle du taux d'erreur sous correction, avec un *below threshold* confirmé expérimentalement. C'est la première plateforme à entrer dans le régime EFT :

- **Taux d'erreur logique** : $< 10^{-6}$ avec le code de surface $d=5$
- **Jeu de portes** : $\{X, \sqrt{X}, \text{SQRT\_ISWAP}, R_Z\}$
- **Temps de cycle** : 0.6–1.0 $\mu$s

```python
import cirq

def willow_vqc_ansatz(qubits, params):
    """Ansatz compatible avec le jeu de portes Willow."""
    operations = []
    for i, q in enumerate(qubits):
        # Native : sqrt(X) et RZ
        sqrt_x = cirq.X(q)**0.5
        rz = cirq.Rz(rads=params[i])(q)
        operations.append(sqrt_x)
        operations.append(rz)
    # Entanglement via SQRT_ISWAP (native Willow)
    for i in range(len(qubits) - 1):
        operations.append(cirq.ISWAP(qubits[i], qubits[i+1])**0.5)
    return operations
```

## Atomes neutres : reconfigurabilité dynamique

Les processeurs à atomes neutres (QuEra, Harvard) utilisent des pinces optiques (optical tweezers) pour piéger et déplacer des atomes individuels, permettant une reconfigurabilité dynamique — un atout majeur pour l'allocation de qubits dans les VQC.

| Caractéristique | QuEra (2026) | Harvard (2026) |
|----------------|--------------|-----------------|
| Qubits | 256 | 512 |
| Fidélité des portes | 99.5 % | 99.7 % |
| Connectivité | Reconfigurable | Reconfigurable |
| Portes natives | $CZ$, $R_Z$, global $R_X$ | $CZ$, $R_Z$, $R_X$ |

L'avantage clé pour le QML est la **connectivité all-to-all reconfigurable** : un circuit VQC qui nécessite des interactions entre des paires de qubits distantes peut être exécuté sans transpilation coûteuse, simplement en repositionnant les atomes entre les couches :

$$
\text{Overhead}_{\text{supra}}(n) \sim O(\sqrt{n}) \quad \text{vs.} \quad \text{Overhead}_{\text{atomes}}(n) \sim O(1)
$$

```python
# Pseudocode QuEra : reconfigurabilité entre couches
# Couche 1 : entanglement A-B, C-D
reconfigure({"A": (0,0), "B": (1,0), "C": (2,0), "D": (3,0)})
apply_cz(("A", "B"))
apply_cz(("C", "D"))
# Couche 2 : entanglement A-C, B-D (reconfiguration)
reconfigure({"A": (0,0), "B": (0,1), "C": (1,0), "D": (1,1)})
apply_cz(("A", "C"))
apply_cz(("B", "D"))
```

## Ions piégés : haute fidélité

Les processeurs à ions piégés (IonQ, Oxford Ionics, Quantinuum) offrent les fidélités les plus élevées parmi toutes les plateformes, grâce à un environnement électromagnétique extrêmement contrôlé.

### IonQ Forte-1

- **Fidélité porte 1-qubit** : 99.99 % (state-of-the-art)
- **Fidélité porte 2-qubits** : 99.9 %
- **Connectivité** : all-to-all via chaîne ionique
- **Temps de cohérence** : $T_2^* > 1\,\text{s}$
- **Jeu de portes** : $\{R_X, R_Y, R_Z, \text{MS}(\theta)\}$

La porte Mølmer–Sørenson (MS) génère un intriquement direct entre tous les ions, ce qui est particulièrement adapté aux ansätze QML nécessitant une forte connectivité :

$$
\text{MS}(\theta) = \exp\left(-i\frac{\theta}{4} \sum_{i<j} X_i X_j\right)
$$

```python
import pennylane as qml

dev_ionq = qml.device("ionq", wires=8, shots=1000)

@qml.qnode(dev_ionq)
def ionq_vqc(x, params):
    qml.AngleEmbedding(x, wires=range(4))
    for d in range(3):
        # Porte MS sur tous les qubits (all-to-all natif)
        qml.MølmerSørensen(params[d], wires=range(4))
    return qml.expval(qml.PauliZ(0))

print(ionq_vqc([0.1, 0.2, 0.3, 0.4], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
# Affiche la valeur d'espérance mesurée
```

### Oxford Ionics

Oxford Ionics utilise une approche unique : contrôle électronique sans laser, réduisant la sensibilité au bruit de phase. Leur fidélité 2-qubits de 99.97 % est la plus haute publiée pour un processeur à ions.

## Qubits topologiques : Majorana 1 (Microsoft)

Microsoft a annoncé le processeur Majorana 1 utilisant des **qubits topologiques** — des états de Majorana protégés topologiquement. L'avantage fondamental est une résistance intrinsèque au bruit local :

- **Protection topologique** : les erreurs locales (changement de $10^{-3}$ par porte) sont supprimées exponentiellement avec la distance topologique
- **Portes Clifford** : fidélité pratiquement illimitée
- **Portes non-Clifford** : nécessitent *magic state distillation* (bottleneck)

Pour le QML, les qubits topologiques offriraient des VQC avec des profondeurs de circuit 100× plus grandes avant saturation par le bruit, permettant des ansätze beaucoup plus expressifs.

## Photonique : QML distribué

Les plateformes photoniques (Xanadu Borealis, Photonic Inc.) utilisent des états de lumière comprimés pour l'informatique quantique. L'avantage pour le QML est la **distribuabilité** : les photons peuvent être transmis via fibre optique.

- **Connectivité réseau** : Télémécanique naturelle pour le QML distribué
- **Fréquence** : Opération à température ambiante
- **Défi** : Intrication probabiliste, pertes photoniques

Xanadu a démontré un *Gaussian Boson Sampling* avec 216 modes comprimés (2023) et développe des *continuous-variable quantum neural networks* (CV-QNN) où les données sont encodées dans des déplacements d'amplitude :

$$
|\psi_{\text{CV}}(x)\rangle = \bigotimes_{i=1}^n D(x_i) |0\rangle,\quad D(\alpha) = \exp(\alpha \hat{a}^\dagger - \alpha^* \hat{a})
$$

## Comparaison pour le QML

| Critère | Supraconducteur | Atomes neutres | Ions piégés | Topologique | Photonique |
|---------|----------------|----------------|-------------|-------------|------------|
| **Profondeur max** | 10–50 couches | 20–100 | 50–200 | 100+ | 10–30 |
| **Fidélité 2-qubits** | 99.5 % | 99.5 % | 99.9 % | ~99.9 % | 99.0 % |
| **Connectivité** | 3–4 voisins | Reconfigurable | All-to-all | Variable | All-to-all |
| **Scalabilité** | 1000+ | 500+ | 100+ | ~10 | 200+ |
| **Taux d'erreur** | $10^{-3}$ | $10^{-3}$ | $10^{-4}$ | $10^{-6}$* | $10^{-2}$ |
| **Idéal pour QML** | VQC, kernels | VQC profonds | Ansätze denses | Circuits longs | QML distribué |

*Protégé topologiquement ; non-Clifford nécessite distillation.

Le choix de la plateforme pour un problème QML donné dépend du *bottleneck* :
- **Profondeur limitée** → atomes neutres ou ions piégés
- **Fidélité critique** → ions piégés
- **Scale** → supraconducteurs
- **Distribution** → photonique
- **Tolérance aux fautes** → topologique (quand disponible)

## Références

- [Fra26] « A Framework-Agnostic Quantum Neural Network Architecture. » *arXiv:2604.04414*, Avril 2026.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
- [Tra25] « Quantum transfer learning for image classification. » *arXiv:2603.16973*, 2025–2026.
