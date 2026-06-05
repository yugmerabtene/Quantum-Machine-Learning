# Séance 14.2 — Défis ouverts et horizon 2026–2030

## Standardisation des benchmarks QML

Un des problèmes majeurs du QML en 2026 est l'absence de standardisation des benchmarks. Les résultats rapportés dans la littérature sont difficilement comparables car :

1. **Jeux de données** : utilisation de datasets ad-hoc, tailles réduites (100–1000 échantillons)
2. **Simulateurs vs. matériel** : performance sur simulateur parfait vs. matériel bruité
3. **Métriques** : accuracy, fidélité du circuit, approximation ratio, temps d'exécution — rarement tous rapportés
4. **Répétabilité** : instructions d'implémentation insuffisantes, code non publié

L'appel à contributions de [Rev26] propose un cadre unifié :

```
Benchmark QML standardisé :
├── Dataset
│   ├── Classique (MNIST, CIFAR-10, Fashion MNIST, PathMNIST)
│   └── Quantique (Hamiltoniens, états intriqués)
├── Implémentation
│   ├── Référence classique (SVM, CNN, ResNet18, ViT)
│   └── Quantique (VQC, QSVM, QCNN, Quantum Transformer)
├── Métriques
│   ├── Performance : accuracy, F1, approximation ratio
│   ├── Ressources : qubits, profondeur, nombre de portes
│   └── Quantiques : fidélité du circuit, variance du gradient
└── Reproductibilité
    ├── Code public (GitHub, Zenodo)
    ├── Seeds aléatoires fixes
    └── Backend spécifié (simulateur, matériel, version SDK)
```

## Passage à l'échelle : plus de qubits, plus de profondeur

Le passage à l'échelle en QML fait face à des obstacles fondamentaux :

### Problème de la profondeur de circuit

Les VQC actuels utilisent typiquement $L = 2$–$10$ couches variationnelles. La théorie montre que pour atteindre une expressivité suffisante, il faudrait $L = O(2^n/\text{poly}(n))$ — exponentiel. Sans correction d'erreur, le bruit limite la profondeur à $L < 1/p$ où $p$ est le taux d'erreur par porte.

### Concentration des kernels

Pour $n$ qubits, le fidelity kernel $k(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$ se concentre autour de $2^{-n}$ pour des données aléatoires, rendant le QSVM inutilisable pour $n > 20$ sans stratégie de mitigation (BFT, kernel alignment).

### Barren plateaux et échelle

La variance du gradient décroît comme $\text{Var}[\partial_\theta C] \sim e^{-cnL}$, ce qui rend l'entraînement impossible pour $nL$ grand. Les solutions partielles (initialisation adaptée, coût local) n'éliminent pas le problème fondamental.

## Avantage quantique pratique : feuille de route 2027–2029

L'avantage quantique pratique — un problème d'importance industrielle résolu significativement mieux par un ordinateur quantique que par tout classique — n'a pas encore été démontré en QML. La feuille de route consensuelle s'articule ainsi :

| Période | Régime | Capacité | QML atteignable |
|---------|--------|----------|-----------------|
| 2026–2027 | NISQ avancé | 1000+ qubits physiques, pQEC | VQC 10-20 qubits, kernels 40+ qubits (BFT) |
| 2027–2028 | EFT débutant | 1–5 qubits logiques | VQC logiques 4 qubits, fidélité ×100 |
| 2028–2029 | EFT intermédiaire | 10–50 qubits logiques | QCNN, Quantum Transformers (qubits logiques) |
| 2030+ | FT partiel | 100+ qubits logiques | VQC profonds, kernels 100+ qubits, avantage ? |

L'incertitude majeure est la date de disponibilité des processeurs EFT à l'échelle. Google Willow et IBM Condor/Heron sont les candidats les plus avancés.

## Formation : le goulot d'étranglement humain

Le QML souffre d'une pénurie critique de talents. En 2026, on estime à **600–700** le nombre de spécialistes mondiaux capables de concevoir et implémenter des algorithmes de correction d'erreur quantique. Les projections pour 2030 indiquent un besoin de **5 000 à 16 000** spécialistes [Rev26].

Les lacunes de compétences sont dans trois domaines :

1. **Théorie de l'information quantique** : codes stabilisateurs, seuils de bruit, fusion de codes
2. **Apprentissage automatique** : architectures profondes, kernels, optimisation non convexe, théorie de la généralisation
3. **Génie logiciel quantique** : PennyLane, Qiskit, Cirq, Stim, Braket, transpliation, déploiement cloud

Des initiatives de formation émergent : cours en ligne (IBM Quantum Learning, Xanadu QML Demos), masters spécialisés (NTNU, Oxford, MIT, INRIA), et programmes doctoraux dédiés.

## Normalisation NIST post-quantique

La normalisation des algorithmes cryptographiques post-quantiques (NIST, 2024–2026) a un impact indirect mais important sur le QML. Les standards retenus (CRYSTALS-Kyber, CRYSTALS-Dilithium, SPHINCS+) influencent :

- **Sécurité des données** : les données médicales et financières traitées par QML doivent être chiffrées avec des algorithmes résistants au quantique
- **QML sécurisé** : développement de protocoles de QML homomorphe et *privacy-preserving*

## NISQ → EFT → FT : implications pour le QML

Chaque régime impose des contraintes différentes sur les algorithmes QML :

### NISQ (2026)

- **Contrainte** : bruit $10^{-2}$–$10^{-3}$, profondeur limitée
- **Optimisation** : pQEC, mitigation (ZNE, PEC), ansatz peu profonds
- **Vainqueurs** : transfer learning, kernels (avec BFT), VQC peu profonds

### EFT (2027–2029)

- **Contrainte** : quelques qubits logiques, overhead de portes magiques
- **Optimisation** : portes Clifford abondantes, $T$ rares
- **Vainqueurs** : circuits avec portes $T$-efficaces, QCNN, Quantum LEGO

### FT (2030+)

- **Contrainte** : overhead mémoire important, latence
- **Optimisation** : profondeur de circuit arbitraire, pas de bruit
- **Vainqueurs** : VQC profonds, QSVM à grande échelle, simulation Hamiltonienne

```python
def regime_adaptatif(n_qubits, depth, p_noise, has_logical=False):
    """Suggère le meilleur régime pour un VQC donné."""
    if has_logical:
        return "EFT"
    if depth > 1 / p_noise:
        return "NISQ (profondeur non supportée)"
    if n_qubits > 20:
        return "NISQ (mitigation nécessaire)"
    return "NISQ"
```

## Méthodes spectrales : une nouvelle direction

[Mil27] Xanadu a introduit une direction radicalement nouvelle : utiliser la **transformée de Fourier quantique** (QFT) comme primitive de ML. L'idée est que la QFT calcule le spectre complet d'une fonction en $O(n^2)$ opérations, là où un classique nécessite $O(n 2^n)$.

Pour une fonction $f : \{0,1\}^n \to \mathbb{R}$ encodée dans un état quantique, la QFT donne accès à tous les coefficients de Fourier $\hat{f}(k)$ simultanément :

$$
|f\rangle = \sum_{x \in \{0,1\}^n} f(x) |x\rangle \quad \xrightarrow{\text{QFT}} \quad \sum_{k \in \{0,1\}^n} \hat{f}(k) |k\rangle
$$

Cette approche pourrait fournir un avantage quantique prouvé pour certaines classes de problèmes de ML où la structure fréquentielle est exploitée (séries temporelles, convolution, débruitage).

## Appel à contributions

Le QML a besoin de travaux fondamentaux pour progresser vers l'avantage pratique :

1. **Frameworks ouverts** : contributions à PennyLane, Qiskit ML, Cirq, Braket
2. **Benchmarks reproductibles** : publication systématique du code, des seeds, des configurations
3. **Théorie unifiée** : lien entre expressivité quantique, généralisation, et avantage computationnel
4. **Matériel ouvert** : collaboration avec les fournisseurs de matériel pour des expériences QML à l'échelle

Le futur du QML ne viendra pas d'une seule avancée, mais de l'accumulation de progrès en théorie des algorithmes, conception de matériel, mitigation d'erreur, et standardisation des benchmarks — un effort collectif qui façonnera l'informatique des prochaines décennies.

## Références

- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [Pen26] Xanadu. « Why quantum computers could be great for machine learning after all. » *PennyLane Blog*, Mars 2026.
- [Tra25] « Quantum transfer learning for image classification. » *arXiv:2603.16973*, 2025–2026.
- [Fra26] « A Framework-Agnostic Quantum Neural Network Architecture. » *arXiv:2604.04414*, Avril 2026.
- [NOI26] « NOISE-VQA: Convergence and Complexity Analysis of VQA with Finite-Shot and Biased Oracles. » *Journal of Computational and Applied Mathematics*, Juin 2026.
