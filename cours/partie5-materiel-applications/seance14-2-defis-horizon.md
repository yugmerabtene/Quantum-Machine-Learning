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

### Métriques : quelles mesures et pourquoi

Le choix des métriques est crucial pour évaluer correctement les approches QML. On distingue trois catégories :

**Métriques de performance prédictionnelle :**

- **Accuracy** : proportion de prédictions correctes. Standard mais insuffisant pour les classes déséquilibrées.
- **F1-score** (macro/micro) : moyenne harmonique de la précision et du rappel. Préférable pour les jeux de données déséquilibrés.
- **AUC-ROC** : aire sous la courbe ROC. Mesure la capacité de séparation indépendamment du seuil de décision.
- **Approximation ratio** $\alpha$ : ratio entre la valeur trouvée par l'algorithme et l'optimum théorique. Utilisé pour les problèmes combinatoires (MaxCut, QAOA).

**Métriques de ressources quantiques :**

- **Nombre de qubits** $n$ : première mesure de la scalabilité d'un algorithme.
- **Profondeur du circuit** $d$ : détermine la faisabilité sur matériel NISQ ($d < 1/p$ où $p$ est le taux d'erreur par porte).
- **Nombre de portes** total : corrélé avec le temps d'exécution et l'erreur cumulée.
- **Nombre de paramètres variationnels** $|\theta|$ : détermine la dimension de l'espace d'optimisation.
- **Shot budget** $N_{\text{shots}}$ : nombre de mesures nécessaires pour estimer chaque observable. Typiquement $N_{\text{shots}} \in [1024, 8192]$.

**Métriques d'entraînement et de convergence :**

- **Variance du gradient** $\text{Var}[\nabla_\theta C]$ : indicateur de barren plateau. $\text{Var} \to 0$ exponentiellement signale un probleme.
- **Nombre d'epochs** pour convergence : comparaison directe des vitesses d'apprentissage.
- **Temps wall-clock** total : inclut simulation classique + compilation quantique + exécution.

### Datasets de référence pour le QML

| Dataset | Taille | Dimension | Classes | Origine | Particularité |
|---------|--------|-----------|---------|---------|---------------|
| Iris | 150 | 4 | 3 | UCI | Standard pour QSVM |
| MNIST (binaire) | 10 000 | 784 | 2 | LeCun | Vision par VQC |
| Fashion MNIST | 70 000 | 784 | 10 | Zalando | Variante MNIST plus robuste |
| CIFAR-10 | 60 000 | 3072 | 10 | Krizhevsky | Transfer learning hybride |
| AdS/CFT | 500 | 8 | 2 | IBM | Classification d'états quantiques |
| TFBS | 5 000 | 4–16 | 2 | Biologie | Génomique, classique + Q |
| Hamiltoniens | 200 | variable | variable | Chimie | Chimie quantique |

Le choix du dataset influence directement la conclusion sur l'avantage quantique : un dataset trop simple (Iris) peut conduire à des conclusions optimistes, tandis qu'un dataset trop grand (CIFAR-10 complet) peut masquer l'avantage par la dominance du backbone classique.

### Code : Benchmark PennyLane vs sklearn sur Iris

```python
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

# ─── Chargement et préparation du dataset Iris ───
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binaire : classe 0 vs classe 1-2 pour comparaison plus claire
X_binary = X[y != 2]
y_binary = y[y != 2]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_binary)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# ─── QSVM (Kernel quantique) ───
n_qubits = 4
dev_kernel = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev_kernel)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

def quantum_kernel(X1, X2):
    return np.array([[kernel_circuit(x1, x2) for x2 in X2] for x1 in X1])

t0 = time.time()
K_train = quantum_kernel(X_train, X_train)
qsvm = SVC(kernel="precomputed")
qsvm.fit(K_train, y_train)
K_test = quantum_kernel(X_test, X_train)
y_pred_qsvm = qsvm.predict(K_test)
t_qsvm = time.time() - t0

# ─── VQC (Circuit variationnel) ───
n_layers = 2
dev_vqc = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev_vqc)
def vqc_circuit(x, weights):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def vqc_predict(X, weights):
    predictions = []
    for x in X:
        raw = vqc_circuit(x, weights)
        predictions.append(0 if raw > 0 else 1)
    return np.array(predictions)

def cost_fn(weights, X, y):
    preds = pnp.array([vqc_circuit(x, weights) for x in X])
    return pnp.mean((preds - (2 * y - 1)) ** 2)

weights_init = pnp.random.uniform(0, 2 * np.pi, (n_layers, n_qubits), requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

t0 = time.time()
weights = weights_init
for epoch in range(60):
    weights, cost = opt.step_and_cost(lambda w: cost_fn(w, X_train, y_train), weights)
    if epoch % 10 == 0:
        print(f"VQC Epoch {epoch:3d} — Cost: {cost:.4f}")
y_pred_vqc = vqc_predict(X_test, weights)
t_vqc = time.time() - t0

# ─── SVM classique (référence) ───
t0 = time.time()
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_rbf.fit(X_train, y_train)
y_pred_svm = svm_rbf.predict(X_test)
t_svm = time.time() - t0

# ─── Résultats comparatifs ───
print("=" * 60)
print(f"{'Modèle':<25} {'Accuracy':<12} {'F1':<12} {'Temps (s)':<10}")
print("=" * 60)
print(f"{'QSVM (kernel PennyLane)':<25} {accuracy_score(y_test, y_pred_qsvm):<12.4f} "
      f"{f1_score(y_test, y_pred_qsvm):<12.4f} {t_qsvm:<10.2f}")
print(f"{'VQC (PennyLane)':<25} {accuracy_score(y_test, y_pred_vqc):<12.4f} "
      f"{f1_score(y_test, y_pred_vqc):<12.4f} {t_vqc:<10.2f}")
print(f"{'SVM-RBF (sklearn)':<25} {accuracy_score(y_test, y_pred_svm):<12.4f} "
      f"{f1_score(y_test, y_pred_svm):<12.4f} {t_svm:<10.2f}")
print("=" * 60)
print(f"QSVM — qubits: {n_qubits}, depth: {qml.specs(kernel_circuit)(X_train[0], X_train[1])['resources'].depth}")
print(f"VQC  — params: {n_layers * n_qubits}")
```

Ce benchmark permet de comparer les approches quantiques (QSVM, VQC) avec une référence classique (SVM-RBF) sur un même dataset. Les métriques rapportées (accuracy, F1, temps) suivent le cadre standardisé proposé ci-dessus.

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

### Timeline détaillée : jalons concrets 2026–2030

**2026 — NISQ avancé et premiers benchmarks standardisés**

- Qubits physiques : 1\,000–1\,500 (IBM Heron r2, Quantinuum H3)
- Taux d'erreur par porte 2Q : $10^{-3}$–$10^{-2}$
- Jalons QML :
  - Standardisation des benchmarks QML (métriques, datasets, protocoles)
  - Transfer learning quantique sur 4–8 qubits démontré
  - QSVM avec kernel alignment sur 20–40 qubits (BFT)
  - Premières expériences de QML sur le cloud quantique à l'échelle du laboratoire
- Limite principale : profondeur de circuit limitée à $d \approx 50$–$100$ opérations

**2027 — Transition EFT débutant**

- Qubits logiques : 1–5 (codes de surface, ratio $\approx 1000$:1)
- Taux d'erreur logique : $10^{-5}$–$10^{-6}$
- Jalons QML :
  - Premiers VQC sur qubits logiques (fidélité ×100 vs. NISQ)
  - Correction d'erreur partielle pour les couches d'embedding
  - Benchmark QML hardware-aware (prise en compte du bruit dans l'architecture)
  - Démonstration de principe : QAOA sur 10–20 qubits logiques
- Limite principale : overhead en portes magiques ($T$-count) pour les opérations non-Clifford

**2028 — EFT intermédiaire**

- Qubits logiques : 10–50
- Profondeur utilisable : $10^3$–$10^4$ portes logiques
- Jalons QML :
  - QCNN profonds sur qubits logiques
  - Quantum Transformers de petite taille ($n = 8$–$16$ qubits)
  - Kernels quantiques à grande échelle ($n > 50$) avec correction d'erreur
  - Premières comparaisons QML classiquement difficiles à simuler
- Limite principale : latence du cycle de mesure-correction ($\sim 1\,\mu$s)

**2029 — EFT mature**

- Qubits logiques : 50–200
- Profondeur utilisable : $10^4$–$10^5$ portes logiques
- Jalons QML :
  - QML compétitif sur des problèmes industriels (optimisation combinatoire, chimie quantique)
  - Méthodes spectrales (QFT) pour le ML à l'échelle
  - Quantum transfer learning à 50+ qubits
- Limite principale : scalabilité du contrôle classique

**2030+ — FT partiel et avantage quantique ?**

- Qubits logiques : 200–1\,000+
- Portes logiques arbitrairement fiables
- Jalons QML :
  - Avantage quantique démontré pour des tâches de ML spécifiques
  - Simulation Hamiltonienne pour le drug discovery
  - QSVM à l'échelle industrielle ($n > 100$)
  - Intégration QML dans les pipelines ML industriels

## Formation : le goulot d'étranglement humain

Le QML souffre d'une pénurie critique de talents. En 2026, on estime à **600–700** le nombre de spécialistes mondiaux capables de concevoir et implémenter des algorithmes de correction d'erreur quantique. Les projections pour 2030 indiquent un besoin de **5 000 à 16 000** spécialistes [Rev26].

Les lacunes de compétences sont dans trois domaines :

1. **Théorie de l'information quantique** : codes stabilisateurs, seuils de bruit, fusion de codes
2. **Apprentissage automatique** : architectures profondes, kernels, optimisation non convexe, théorie de la généralisation
3. **Génie logiciel quantique** : PennyLane, Qiskit, Cirq, Stim, Braket, transpliation, déploiement cloud

Des initiatives de formation émergent : cours en ligne (IBM Quantum Learning, Xanadu QML Demos), masters spécialisés (NTNU, Oxford, MIT, INRIA), et programmes doctoraux dédiés.

### Compétences requises pour le QML : profil détaillé

Le profil idéal d'un chercheur ou ingénieur QML combine des compétences dans trois piliers, chacun avec des niveaux de maîtrise spécifiques :

**Pilier 1 — Physique et informatique quantique :**

- Algèbre linéaire dans les espaces de Hilbert : opérateurs, valeurs propres, décomposition spectrale
- Théorie de l'information quantique : entropie de von Neumann, inégalités de Holevo, bornes de communication
- Calcul quantique : portes universelles, circuits, téléportation, codes de correction d'erreur
- Simulation quantique : méthodes de Trotter-Suzuki, VQE, algorithmes variationnels

**Pilier 2 — Apprentissage automatique classique :**

- Théorie de la généralisation : complexité de Rademacher, PAC-learning, bornes de généralisation
- Architectures de réseaux de neurones : CNN, Transformers, architectures résiduelles
- Optimisation : descente de gradient, méthodes adaptatives (Adam, LAMB), paysages de perte
- Kernels et méthodes à noyau : kernel trick, approximation de Nyström, kernel alignment

**Pilier 3 — Ingénierie logicielle quantique :**

- Frameworks : PennyLane (différentiable), Qiskit (IBM), Cirq (Google), Braket (AWS)
- Compilation et transpilation : mapping de qubits, routage, optimisation de portes
- Mitigation d'erreur : ZNE, PEC, lecture mitigée, virtual Z
- Déploiement cloud : IBM Quantum, Amazon Braket, Azure Quantum, Google Quantum AI

### Parcours de formation et carrières QML

**Niveaux de compétence et parcours types :**

| Niveau | Durée estimée | Compétences | Rôles visés |
|--------|---------------|-------------|-------------|
| Débutant | 3–6 mois | Linéaire, portes quantiques, VQE, Python | Stagiaire QML |
| Intermédiaire | 6–12 mois | VQC, kernels, PennyLane, Qiskit | Ingénieur QML junior |
| Avancé | 1–2 ans | Théorie de la généralisation, correction d'erreur, hardware-aware | Chercheur QML |
| Expert | 2–4 ans | Architecture hybride, compilation, benchmarking | Lead QML, poste académique |

**Principaux employeurs et secteurs :**

- **Big Tech** : IBM (Qiskit, hardware), Google (Cirq, Willow), Amazon (Braket), Microsoft (Azure Quantum)
- **Startups quantiques** : Xanadu (PennyLane, hardware photonique), IonQ (trapped ions), Quantinuum (H series), Pasqal (atomes neutres)
- **Finance** : JP Morgan, Goldman Sachs, Barclays — optimisation de portefeuille, tarification d'options
- **Pharmacie** : Roche, Merck, Recursion — chimie quantique pour le drug discovery
- **Académie** : INRIA, MIT, Oxford, Max Planck, chaires de professeur en QML

**Programmes de formation recommandés :**

- **Cours en ligne** : IBM Quantum Learning (Qiskit textbook), Xanadu Quantum Code School, Coursera (Quantum Machine Learning — IBM), edX (Quantum Computing — MIT)
- **Mastères spécialisés** : NTNU (MSc Quantum Computing), Oxford (MSc Quantum Technology), ETH Zürich (MSc Quantum Engineering), University of Waterloo (MSc Quantum Information)
- **Doctorats** : programmes doctoraux en informatique quantique à l'INRIA, Max Planck Institute, MIT, dans le cadre de consortiums comme le Quantum Flagship européen

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

[Pen26] Xanadu a introduit une direction radicalement nouvelle : utiliser la **transformée de Fourier quantique** (QFT) comme primitive de ML. L'idée est que la QFT calcule le spectre complet d'une fonction en $O(n^2)$ opérations, là où un classique nécessite $O(n 2^n)$.

Pour une fonction $f : \{0,1\}^n \to \mathbb{R}$ encodée dans un état quantique, la QFT donne accès à tous les coefficients de Fourier $\hat{f}(k)$ simultanément :

$$
|f\rangle = \sum_{x \in \{0,1\}^n} f(x) |x\rangle \quad \xrightarrow{\text{QFT}} \quad \sum_{k \in \{0,1\}^n} \hat{f}(k) |k\rangle
$$

Cette approche pourrait fournir un avantage quantique prouvé pour certaines classes de problèmes de ML où la structure fréquentielle est exploitée (séries temporelles, convolution, débruitage).

### Fondements théoriques des méthodes spectrales quantiques

La connexion entre les circuits variationnels et la décomposition en série de Fourier a été établie par Schuld et al. [SP19]. Tout circuit variationnel de la forme $U(x) = \prod_j e^{-i x H_j / 2} W_j$ réalise une fonction de Fourier :

$$
f_\theta(x) = \sum_{\omega \in \Omega} c_\omega(\theta) \, e^{i \omega x}
$$

où $\Omega \subseteq \mathbb{R}$ est l'ensemble des fréquences accessibles (le **spectre** du circuit), déterminé par les valeurs propres des matrices de Pauli dans l'ansatz. Les fréquences maximales sont bornées par :

$$
|\omega| \leq \sum_{j} |\lambda_j| \leq L \cdot n
$$

où $L$ est le nombre de couches et $n$ le nombre de qubits. Ce résultat montre que le VQC est un **approximateur de Fourier universel** : en augmentant $L$, on peut approcher toute fonction périodique avec une précision arbitraire.

L'avancée de Xanadu [Pen26] exploite cette connexion de manière plus directe : plutôt que d'utiliser un VQC comme approximateur de Fourier « implicite », on utilise la QFT comme primitive « explicite » pour extraire le spectre complet en une seule opération quantique.

### Applications potentielles des méthodes spectrales

**Séries temporelles quantiques :** Pour une série temporelle $y_t$ de longueur $T = 2^n$, l'encodage amplitudeal $\sum_t y_t |t\rangle$ suivi d'une QFT donne la transformée de Fourier discrète en $O(n^2)$ portes, contre $O(T \log T) = O(n 2^n)$ classiquement. L'avantage est exponentiel en $n$ pour l'obtention du spectre complet.

**Convolution quantique :** La convolution circulaire de deux signaux $f$ et $g$ se calcule via la QFT en $O(n^2)$ :

$$
f * g = \text{QFT}^{-1}\!\left[\text{QFT}(f) \odot \text{QFT}(g)\right]
$$

Cette primitive est fondamentale pour les architectures de Quantum CNN et Quantum Vision Transformers.

**Débruiteur spectral quantique :** En appliquant un seuillage spectral (spectral filtering) dans le domaine de Fourier quantique, on peut débruiter un signal en supprimant les composantes haute fréquence associées au bruit, sans passer par l'espace classique.

### Limites actuelles des méthodes spectrales

Malgré leur potentiel, les méthodes spectrales quantiques font face à plusieurs défis :

1. **Encodage amplitudeal** : la préparation de l'état $|f\rangle = \sum_x f(x) |x\rangle$ coûte en général $O(2^n)$ opérations, annihilant potentiellement l'avantage de la QFT. Des encodages spéciaux (parallèles, basés sur des oracles) sont nécessaires.

2. **Lecture des résultats** : la mesure de l'état de Fourier $\sum_k \hat{f}(k) |k\rangle$ donne accès à un seul coefficient $\hat{f}(k)$ par mesure. Pour obtenir le spectre complet, il faut $O(2^n)$ mesures — ce qui réduit à néant l'avantage computationnel. Des techniques comme l'estimation d'amplitude quantique ou la tomographie compressée peuvent partiellement contourner cette limite.

3. **Bruit matériel** : la QFT est sensible aux erreurs de phase, qui se propagent dans tout le spectre. La correction d'erreur est essentielle pour les applications spectrales pratiques.

4. **Conditions de ressource** : l'avantage n'existe que si les données sont déjà disponibles sous forme quantique (par ex. sortie d'un capteur quantique) ou si l'encodage peut être effectué en $O(\text{poly}(n))$ opérations.

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
- [SP19] Schuld, M. et al. « Quantum machine learning models are kernel methods. » *arXiv:2101.11020*, 2019–2021.
- [Mc18] McClean, J. R. et al. « Barren plateaus in quantum neural network training landscapes. » *Nature Communications*, 2018.
- [BFT22] Huang, H.-Y. et al. « Quantum advantage in learning from experiments. » *Science*, 2022.
- [Hua21] Huang, H.-Y. et al. « Power of data in quantum machine learning. » *Nature Communications*, 2021.
