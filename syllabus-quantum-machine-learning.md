# Syllabus — Apprentissage Automatique Quantique : Théorie, Algorithmes et Applications

**Niveau :** Master / Doctorat (M2 / PhD)
**Durée :** 28 séances (2 × 2h par semaine, 14 semaines)
**Langue :** Français

---

## 1. Description du cours

Ce cours propose une couverture complète de l'apprentissage automatique quantique (Quantum Machine Learning, QML), depuis les fondements mathématiques de la théorie de l'apprentissage et du calcul quantique jusqu'aux architectures hybrides les plus avancées, aux kernels quantiques, aux réseaux de neurones variationnels, et aux applications industrielles. Il s'appuie sur les développements les plus récents (2024–2026) — notamment les avancées en kernels quantiques covarants (IBM 156 qubits, *npj Quantum Information* 2026), les réseaux de neurones non-unitaires (LCU, *arXiv* 2026), le Quantum LEGO Learning, les Quantum Transformers, les méthodes spectrales (Xanadu/PennyLane 2026), et les benchmarks systématiques QML vs. classique (*Discover Computing* 2026).

Une importance particulière est accordée à l'articulation entre théorie et pratique : chaque module théorique est associé à une implémentation en Python utilisant **PennyLane**, **Qiskit Machine Learning**, **PyTorch**/**TensorFlow**, **Cirq**, et **Amazon Braket**. La progression pédagogique va des **fondements classiques du ML** (théorie de l'apprentissage, réseaux de neurones, kernels) à **l'encodage de données quantique** (angle, amplitude, IQP), puis aux **modèles variationnels** (VQC, VQE, QAOA), aux **kernels quantiques** (fidelity kernels, QSVM), aux **architectures hybrides avancées** (QCNN, Quantum Transformers, QRL), et enfin à la **correction d'erreur pour QML** et aux **applications industrielles**.

---

## 2. Prérequis

- Algèbre linéaire (espaces vectoriels complexes, produits tensoriels, valeurs propres)
- Probabilités et statistiques de base
- Notions de calcul quantique (qubits, portes, circuits) recommandées — un rappel est prévu
- Programmation en Python (NumPy, PyTorch ou TensorFlow de base)
- Notions de base en apprentissage automatique (régression, classification, SGD) — un rappel intensif est prévu en début de cours

---

## 3. Objectifs d'apprentissage

À l'issue du cours, l'étudiant sera capable de :

1. Maîtriser le formalisme mathématique de l'apprentissage automatique classique et quantique (théorie de l'apprentissage, espaces de Hilbert à noyau reproduisant RKHS, complexité de Rademacher)
2. Concevoir et analyser des circuits quantiques variationnels (VQC) comme modèles d'apprentissage
3. Implémenter et analyser les différentes stratégies d'encodage de données (angle, amplitude, IQP, Hamiltonian)
4. Comprendre et implémenter les kernels quantiques (fidelity kernels, quantum kernel alignment, covariant kernels)
5. Maîtriser les architectures hybrides classique-quantique (transfer learning, QCNN, Quantum Transformers)
6. Analyser les problèmes d'entraînement des VQC (barren plateaux, ULSP, concentration exponentielle)
7. Expliquer et mettre en œuvre les techniques de mitigation d'erreur pour QML sur matériel NISQ
8. Appliquer le QML à des domaines concrets : chimie quantique (VQE), optimisation (QAOA), santé, finance
9. Appréhender les avancées récentes (2024–2026) : non-unitaire QML, Quantum LEGO, VQC-MLPNet, méthodes spectrales
10. Déployer des modèles QML sur simulateurs et matériels quantiques réels via le cloud (IBM, Amazon Braket, IonQ)

---

## 4. Manuels de référence

### Ouvrages fondamentaux

| Réf. | Ouvrage |
|------|---------|
| [SP21] | **Schuld, M. & Petruccione, F.** *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. ISBN 978-3-030-83098-4. — *La référence absolue en QML, couvre théorie et implémentation.* |
| [DWQ25] | **Du, Y. et al.** *A Gentle Introduction to Quantum Machine Learning.* Springer, 2025. ISBN 978-981-95-1284-3. |
| [NC00] | **Nielsen, M. A. & Chuang, I. L.** *Quantum Computation and Quantum Information.* Cambridge University Press, 2000 (10th Anniversary Ed. 2010). |
| [Bis06] | **Bishop, C. M.** *Pattern Recognition and Machine Learning.* Springer, 2006. ISBN 978-0-387-31073-2. |
| [MRT18] | **Mohri, M., Rostamizadeh, A. & Talwalkar, A.** *Foundations of Machine Learning.* 2nd ed., MIT Press, 2018. ISBN 978-0-262-03940-6. |
| [GBC16] | **Goodfellow, I., Bengio, Y. & Courville, A.** *Deep Learning.* MIT Press, 2016. ISBN 978-0-262-03561-3. |

### Ouvrages spécialisés

| Réf. | Ouvrage |
|------|---------|
| [SC24] | **Schuld, M.** *Supervised Learning with Quantum Computers.* Springer, 2024. ISBN 978-3-319-96423-2. |
| [Con22] | **Cong, I.** « Quantum Machine Learning, Error Correction, and Topological Phases of Matter. » *Thèse de doctorat, Harvard University*, 2022. |
| [Pre98] | **Preskill, J.** *Lecture Notes for Physics 219: Quantum Computation.* Caltech, 1998–2023. |
| [Aar13] | **Aaronson, S.** *Quantum Computing Since Democritus.* Cambridge University Press, 2013. |

---

## 5. Articles de recherche récents (2024–2026)

### Kernels quantiques et méthodes à noyau

| Réf. | Article |
|------|---------|
| [Agl26] | **Agliardi, G. et al.** « Mitigating exponential concentration in covariant quantum kernels for subspace and real-world data. » *npj Quantum Information* 12, 12 (2026). — *Plus grande expérience QML sur IBM : 156 qubits, accuracy 80% avec BFT.* |
| [Var26] | « A Versatile Variational Quantum Kernel Framework for Non-Trivial Classification. » *arXiv:2511.10831*, Février 2026. — *Framework de kernels variationnels sur 8 jeux de données réels.* |
| [Hav19] | **HavlÍček, V. et al.** « Supervised learning with quantum-enhanced feature spaces. » *Nature* 567, 209–212 (2019). — *Article fondateur des feature maps quantiques.* |

### Architectures hybrides et réseaux de neurones quantiques

| Réf. | Article |
|------|---------|
| [QuL26] | « Quantum LEGO Learning: A Modular and Architecture-Agnostic Hybrid QML Framework. » *arXiv:2601.21780*, 2026. — *Framework modulaire bloc gelé + VQC adaptatif.* |
| [VQC26] | « VQC-MLPNet: Unconventional Hybrid Quantum-Classical Architecture. » *arXiv:2506.10275*, 2026. — *VQC générant les poids d'un MLP classique.* |
| [Tra26] | « Quantum Transformers for Image Classification: Integrating VQC and Quantum Wavelet KAN. » *Quantum Machine Intelligence* 8, 43 (2026). — *Accuracy 94.42% sur Fashion MNIST, 90.57% sur CIFAR-10.* |
| [Pen26] | **Xanadu.** « Why quantum computers could be great for machine learning after all. » *PennyLane Blog*, Mars 2026. — *Méthodes spectrales et transformée de Fourier quantique pour le ML.* |
| [NOM26] | **Li, K. et al.** « Learning parameterized quantum circuits with quantum gradient. » *npj Quantum Information* 12, 59 (2026). — *Nested Optimization Model (NOM) pour échapper aux barren plateaux.* |

### Non-unitaire QML et expressivité

| Réf. | Article |
|------|---------|
| [LCU26] | « Non-Unitary Quantum Machine Learning: Fisher Efficiency Transitions from Distributed Quantum Expressivity. » *arXiv:2603.27377*, Mars 2026. — *570+ expériences, LCU, gain +0.2% à +5.8%, transition Fisher à 10-12 qubits.* |
| [Fra26] | « A Framework-Agnostic Quantum Neural Network Architecture. » *arXiv:2604.04414*, Avril 2026. — *QNN multi-framework (Qiskit, Cirq, PennyLane, Braket) avec export ONNX.* |

### Benchmarking et revues

| Réf. | Article |
|------|---------|
| [Rev26] | « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026. — *Revue comparative systématique QML vs. classique.* |
| [VQA26] | « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026. — *VQA de NISQ à FT, barren plateaux, mitigation.* |
| [NOI26] | « NOISE-VQA: Convergence and Complexity Analysis of VQA with Finite-Shot and Biased Oracles. » *Journal of Computational and Applied Mathematics*, Juin 2026. |
| [Ben26] | « Benchmarking variational quantum algorithms for combinatorial optimization in practice. » *Quantum Machine Intelligence* 8, 26 (2026). |
| [Ader26] | **Adermann, L. et al.** « Lightweight Error Detection for Variational Quantum Classifiers. » *arXiv*, 2026. — *Code [[4,2,2]] pour VQC, amélioration accuracy sous bruit.* |

### Applications

| Réf. | Article |
|------|---------|
| [Tra25] | « Quantum transfer learning for image classification. » *arXiv:2603.16973*, 2025–2026. — *Comparaison PennyLane vs. Qiskit, accuracy 97.44% sur Hymenoptera.* |
| [QML25] | **Qiskit ML Contributors.** « Qiskit Machine Learning: an open-source library for QML tasks at scale. » *arXiv:2505.17756*, 2025. |

---

## 6. Plan détaillé — Calendrier des séances

### PARTIE I : FONDEMENTS — APPRENTISSAGE CLASSIQUE ET QUANTIQUE

| Séance | Contenu | Références |
|--------|---------|------------|
| **1.1** | **Introduction historique et panorama.** De l'IA classique au QML. Pourquoi le quantique pour le ML ? La boucle hybride classique-quantique. Aperçu des architectures NISQ, EFT, FT. État des lieux 2026 : marché 1,4 G$ (McKinsey), 156 qubits IBM, avancées Xanadu/Google. | [SP21] Ch. 1 ; [Rev26] ; [Fra26] |
| **1.2** | **Théorie de l'apprentissage automatique (rappel).** Apprentissage supervisé, non-supervisé, par renforcement. Fonction de perte, risque empirique vs. structurel. Généralisation, sur-apprentissage. Théorème de No-Free-Lunch. | [Bis06] Ch. 1 ; [MRT18] Ch. 2 |
| **2.1** | **Algèbre linéaire et espaces de Hilbert.** Rappel : espaces vectoriels complexes, notation de Dirac, produit tensoriel, opérateurs. RKHS (Reproducing Kernel Hilbert Space) : théorème de Mercer, noyau, feature map implicite. | [SP21] §2.1–2.3 ; [MRT18] Ch. 6 |
| **2.2** | **Calcul quantique (rappel intensif).** Postulats de la MQ, qubit, portes (Pauli, Hadamard, CNOT), mesure, circuit quantique. Sphère de Bloch. Intrication et Bell. | [NC00] §1–2 ; [SP21] §3.1–3.3 |
| **3.1** | **Réseaux de neurones classiques (rappel).** Perceptron, MLP, rétropropagation, SGD, Adam. Réseaux convolutifs (CNN). Transformers : attention, self-attention, mécanisme de QKV. | [GBC16] Ch. 6, 9 ; [Bis06] Ch. 5 |
| **3.2** | **Méthodes à noyau classiques (rappel).** SVM, kernel trick, noyaux RBF, polynomial, laplacien. Alignement de noyaux (kernel alignment). Théorie de la régularisation par noyau. | [MRT18] Ch. 6–7 ; [Bis06] Ch. 6 |

### PARTIE II : ENCODAGE ET CIRCUITS VARIATIONNELS

| Séance | Contenu | Références |
|--------|---------|------------|
| **4.1** | **Encodage de données classiques en états quantiques.** Angle encoding, amplitude encoding, basis encoding, IQP encoding, Hamiltonian encoding. Comparaison : ressources, expressivité, complexité. Feature maps : ZZFeatureMap, PauliFeatureMap. | [SP21] Ch. 4 ; [Hav19] |
| **4.2** | **Circuits quantiques paramétrés (PQC/VQC).** Définition, structure (embedding + variational layers + mesure). Ansätze : circuit-centric, hardware-efficient, tensor-network. Théorème d'universalité. | [SP21] §3.6, §5.1 ; [VQA26] |
| **5.1** | **Variational Quantum Classifier (VQC).** Architecture complète : encodage → couches variationnelles → mesure → interprétation. Fonction de coût, optimisation. Implémentation PennyLane et Qiskit ML. | [SP21] §5.2–5.3 ; [QML25] |
| **5.2** | **Différentiation automatique des circuits quantiques.** Parameter-shift rule, méthode des différences finies, adjoint differentiation. Comparaison des méthodes. Intégration avec PyTorch/TensorFlow/JAX. | [SP21] §5.4 ; [Pen26] |
| **6.1** | **Barren Plateaux et problèmes d'entraînement.** Définition, origine (McClean et al. 2018). Coût global vs. local. Impact de la profondeur, du bruit, de l'intrication. Stratégies de mitigation : initialization, warm-start, layer-wise learning. | [SP21] §5.5 ; [VQA26] ; [NOM26] |
| **6.2** | **ULSP et optimisation avancée.** Unfavorable Local Stationary Points. Nested Optimization Model (NOM). Quantum gradient pour échapper aux minima. ADAPT-VQC et croissance dynamique d'ansatz. | [NOM26] |

### PARTIE III : KERNELS QUANTIQUES

| Séance | Contenu | Références |
|--------|---------|------------|
| **7.1** | **Quantum Kernel Methods (QKM).** Définition : kernel quantique = fidélité entre états encodés. Fidelity quantum kernel. Quantum kernel estimator. Quantum-enhanced feature spaces. | [SP21] Ch. 6 ; [Hav19] |
| **7.2** | **Quantum Support Vector Machine (QSVM).** Implémentation du QSVM. Least-squares QSVM. Comparaison classique vs. quantique. Kernel alignment : optimisation des paramètres du feature map. | [SP21] §6.3–6.4 ; [Agl26] |
| **8.1** | **Concentration exponentielle des kernels.** Problème fondamental : la concentration rend les kernels inutilisables à grande échelle. Analyse théorique. Groupe de symétrie et structure covariante. | [Agl26] |
| **8.2** | **Mitigation : Bit-Flip Tolerance et Quantum Kernel Alignment.** BFT : seuil de poids de Hamming, calibration linéaire. Résultats IBM : 156 qubits, 80% accuracy. Variational Quantum Kernel Framework : kernels sur 8 jeux de données réels. | [Agl26] ; [Var26] |

### PARTIE IV : ARCHITECTURES HYBRIDES AVANCÉES

| Séance | Contenu | Références |
|--------|---------|------------|
| **9.1** | **Hybrid classical-quantum architectures.** Transfer learning quantique : backbone CNN gelé + tête quantique. Quantum LEGO Learning : modularité bloc gelé + VQC adaptatif. Analyse théorique de la généralisation. | [Tra25] ; [QuL26] |
| **9.2** | **Quantum Convolutional Neural Networks (QCNN).** Architecture : convolution quantique → pooling → mesure. Avantages : réduction de paramètres, stabilité. Applications : classification d'états quantiques, reconnaissance d'images. | [SP21] §7.3 |
| **10.1** | **Quantum Transformers.** Intégration de VQC dans l'attention multi-têtes. Quantum Wavelet KAN. Résultats : Fashion MNIST 94.42%, CIFAR-10 90.57%. Comparaison avec classique. | [Tra26] |
| **10.2** | **VQC-MLPNet et architectures non-unitaires.** VQC générant les poids d'un MLP classique (amplitude encoding). Analyse NTK. Non-unitaire QML via LCU : Fisher efficiency transition à 10-12 qubits. Gains +0.2% à +5.8% sur 4 domaines. | [VQC26] ; [LCU26] |
| **11.1** | **Quantum Generative Models.** Quantum GAN (qGAN). Quantum Boltzmann Machines. Born Machines. Circuit Born Machine pour la modélisation de distributions. | [SP21] §7.5 ; [QML25] |
| **11.2** | **Quantum Reinforcement Learning (QRL) et Federated Learning (QFL).** Policy gradient quantique. Quantum federated learning. Applications : robotique, optimisation de portefeuille. | [Rev26] |

### PARTIE V : CORRECTION D'ERREUR, MITIGATION ET MATÉRIEL

| Séance | Contenu | Références |
|--------|---------|------------|
| **12.1** | **Bruit et mitigation pour QML.** Modèles de bruit (dépolarisant, amplitude/phase damping). Error mitigation : zero-noise extrapolation, probabilistic error cancellation, CVaR. Impact sur l'entraînement des VQC. | [VQA26] ; [NOI26] |
| **12.2** | **Partial Quantum Error Correction (pQEC) pour QML.** Correction partielle pour VQC : code [[4,2,2]]. Fidélité améliorée ×9.27. Compromis ressources-performance. Passage de NISQ à EFT. | [VQA26] ; [Ader26] |
| **13.1** | **Plateformes matérielles pour QML.** Supraconducteurs (IBM Condor, Google Willow) : avantages, limites pour QML. Atomes neutres (Harvard/QuEra) : reconfigurabilité. Ions piégés (IonQ, Oxford Ionics) : haute fidélité 99.99%. Photonique (Photonic Inc., Xanadu). | [Fra26] ; [Rev26] |
| **13.2** | **Déploiement cloud et multi-plateforme.** IBM Quantum, Amazon Braket, Azure Quantum. Framework-agnostic QNN : ONNX export, transpilation automatique. Benchmark multi-backend : gradient MAE ≤ 0.006 sur 4 fournisseurs. | [Fra26] ; [QML25] |
| **14.1** | **Applications industrielles du QML.** Chimie quantique et drug discovery (VQE). Finance : optimisation de portefeuille (QAOA), détection de fraudes (QSVM). Santé : PathMNIST, classification histopathologique. Matériaux. Agriculture : PlantVillage. | [Rev26] ; [Tra25] ; [LCU26] |
| **14.2** | **Défis ouverts et horizon 2026–2030.** Standardisation des benchmarks. Passage à l'échelle. Avantage quantique pratique. Feuille de route NISQ → EFT → FT. Normalisation NIST post-quantique. Formation : besoin de 5 000–16 000 spécialistes d'ici 2030. | [Rev26] ; [VQA26] ; [Pen26] |

---

## 7. Travaux pratiques et projets — Laboratoires Python

### 7.1 Écosystème Python pour le QML

Huit bibliothèques Python sont utilisées dans ce cours, choisies pour leur complémentarité, leur adoption académique et industrielle, et leur pertinence pédagogique :

| Bibliothèque | Version | Rôle dans le cours | Domaine | Install |
|---|---|---|---|---|
| **PennyLane** (Xanadu) | 0.45+ | Cœur du cours : différentiation automatique des circuits, VQC, kernels, intégration PyTorch/TF/JAX | QML général, VQE, QAOA | `pip install pennylane` |
| **Qiskit ML** (IBM) | 1.x | Kernels quantiques, QSVM, qGAN, SamplerQNN, EstimatorQNN, accès IBM Quantum | Kernels, classification, génératif | `pip install qiskit-machine-learning` |
| **PyTorch / TensorFlow** | 2.x | Backend classique pour les architectures hybrides, extraction de features, CNNs | Deep learning classique et hybride | `pip install torch torchvision` |
| **Cirq** (Google) | 1.x | Circuits NISQ, optimisation de portes, intégration Stim | Circuits, optimisation | `pip install cirq` |
| **Qiskit** (IBM) | 2.x | Construction de circuits, transpilation, accès au matériel IBM | Bases des circuits | `pip install qiskit qiskit-aer` |
| **Stim** (Google) | 1.x | Simulation de circuits stabilisateurs pour la correction d'erreur | QEC pour QML | `pip install stim` |
| **Amazon Braket** | 1.x | Accès unifié multi-matériel (IonQ, Rigetti, QuEra) | Cloud, multi-plateforme | `pip install amazon-braket-sdk` |
| **scikit-learn** | 1.x | SVM classique, métriques, jeux de données, pipelines | ML classique, benchmark | `pip install scikit-learn` |

**Justification du choix :**

- **PennyLane** est la librairie de référence en apprentissage automatique quantique, avec différentiation automatique native des circuits et intégration profonde avec PyTorch, TensorFlow, et JAX. Elle supporte de multiples backends (simulateurs, IBM, Braket, IonQ, Rigetti) via son architecture de plugins, et est utilisée dans la majorité des publications QML récentes (Xanadu, 2026). Elle offre les outils les plus avancés pour la recherche en QML : parameter-shift, adjoint differentiation, noise models, et optimiseurs classiques.
- **Qiskit ML** est le cadre officiel IBM pour le QML, avec des implémentations optimisées pour les kernels quantiques (FidelityQuantumKernel, ComputeUncompute), les QNN (SamplerQNN, EstimatorQNN), et l'accès natif aux processeurs IBM Quantum via Qiskit Runtime. Il a servi pour la plus grande expérience QML jamais réalisée : 156 qubits sur *ibm_marrakesh* [Agl26].
- **PyTorch/TensorFlow** sont indispensables pour les architectures hybrides : extraction de features par CNN pré-entraînés, optimisation classique, et intégration des circuits quantiques comme couches différentiables.
- **Cirq** et **Stim** sont utilisés pour les aspects correction d'erreur et optimisation de circuits NISQ, en complément de PennyLane et Qiskit.
- **Amazon Braket** permet l'exécution multi-fournisseur (IonQ, Rigetti, QuEra) sans changement d'API.
- **scikit-learn** fournit les classifieurs classiques (SVM, Random Forest, MLP) pour les benchmarks indispensables à toute étude QML rigoureuse.

### 7.2 Structure des laboratoires

Chaque laboratoire suit un canevas identique : **fondement théorique** → **implémentation classique** (baseline) → **implémentation quantique** (PennyLane/Qiskit) → **benchmark et analyse**. Cette progression permet à l'étudiant de comprendre à la fois les bases classiques et l'apport quantique.

#### Laboratoire 1 — Sphère de Bloch et premiers circuits quantiques (Semaine 2)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Maîtriser la représentation des qubits, les portes fondamentales, et la mesure |
| **PennyLane** | `qml.BlochSphere()`, `qml.RX`, `qml.RY`, `qml.RZ`, `qml.CNOT`, `qml.Hadamard`. Création d'états de Bell. Mesure : `qml.sample()`, `qml.expval()` |
| **Qiskit** | `Statevector`, `plot_bloch_sphere()`, portes X, Y, Z, H, S, T, CNOT |
| **Notions** | Sphère de Bloch, état pur, superposition, intrication, mesure projective, probabilités |

#### Laboratoire 2 — Classifieur classique (baseline) sur Iris / Wine (Semaine 1)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Implémenter un pipeline ML classique complet, établir la baseline |
| **scikit-learn** | SVM (RBF, polynomial), Random Forest, MLP. Train/test split, cross-validation, métriques (accuracy, F1, ROC-AUC). Feature scaling, PCA |
| **PyTorch** | MLP simple à 2 couches, SGD/Adam, fonction de perte cross-entropy |
| **Notions** | Pipeline ML, validation, hyperparamètres, overfitting, baseline |

#### Laboratoire 3 — Encodage de données et feature maps (Semaine 4)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Implémenter et comparer les stratégies d'encodage quantique |
| **PennyLane** | `qml.AngleEmbedding`, `qml.AmplitudeEmbedding`, `qml.IQPEmbedding`. Comparaison de l'espace des features. Visualisation de l'espace de Hilbert |
| **Qiskit ML** | `ZZFeatureMap`, `PauliFeatureMap`. `ZFeatureMap`. Visualisation via `plot_bloch_multivector` |
| **Notions** | Encoding, feature map, espace de Hilbert, richesse d'expressivité, coût en qubits |

#### Laboratoire 4 — Variational Quantum Classifier (VQC) sur Iris (Semaine 5)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Implémenter un classifieur quantique variationnel complet |
| **PennyLane** | Circuit : `AngleEmbedding` → `BasicEntanglerLayers` → mesure. Fonction de coût : cross-entropy. Optimisation : Adam, SPSA. Comparaison avec classique |
| **Qiskit ML** | `EstimatorQNN` ou `SamplerQNN`. `NeuralNetworkClassifier`. Pipeline d'entraînement |
| **Notions** | VQC, ansatz, fonction de coût, optimisation hybride, parameter-shift |

#### Laboratoire 5 — Barren Plateaux et analyse de trainabilité (Semaine 6)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Observer et caractériser le phénomène de barren plateau |
| **PennyLane** | Variance du gradient en fonction du nombre de qubits et de couches. Ansatz aléatoire vs. structuré. Comparaison coût global vs. local. Initialisation : uniforme, normal, identity |
| **Notions** | Barren plateau, gradient variance, expressivité, coût local vs. global, initialization |

#### Laboratoire 6 — Quantum Support Vector Machine (QSVM) (Semaine 7)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Implémenter un kernel quantique et l'utiliser dans un SVM |
| **PennyLane** | `qml.kernels.embedding_kernel_matrix`. Fidelity kernel. Quantum kernel alignment : optimisation des paramètres de l'embedding |
| **Qiskit ML** | `FidelityQuantumKernel`. `ComputeUncompute`. `QSVC`. Comparaison avec SVM classique (RBF, polynomial) |
| **scikit-learn** | `SVC` avec kernel RBF, polynomial. Kernel alignment classique |
| **Notions** | Fidelity kernel, quantum kernel alignment, concentration, QSVM, feature space |

#### Laboratoire 7 — Transfer Learning quantique (Semaine 9)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Hybridation CNN classique + tête quantique pour la classification d'images |
| **PyTorch + PennyLane** | ResNet18/ EfficientNet gelé + VQC 4 qubits (AngleEmbedding + BasicEntanglerLayers). Comparaison avec tête classique fully-connected. Métriques : accuracy, temps d'entraînement |
| **Qiskit ML** | `SamplerQNN` comme tête de classification. Comparaison PennyLane vs. Qiskit |
| **Notions** | Transfer learning, hybridation, feature extractor, fine-tuning vs. frozen |

#### Laboratoire 8 — Quantum Convolutional Neural Network (QCNN) (Semaine 9)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Implémenter un réseau de neurones convolutionnel quantique |
| **PennyLane** | Couches de convolution quantique (circuit 2×2), pooling quantique. Classification sur MNIST réduit. Comparaison avec CNN classique |
| **Notions** | QCNN, convolution quantique, pooling, réduction de paramètres, hiérarchie de features |

#### Laboratoire 9 — VQE et chimie quantique (Semaine 11)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Calculer l'état fondamental de H₂ par VQE |
| **PennyLane** | Construction de l'Hamiltonien moléculaire. Ansatz : `qml.UCCSD`, `qml.HartreeFock`. Optimisation : Adam, SPSA. Comparaison avec diagonalisation exacte |
| **Qiskit Nature** | `PySCFDriver` pour l'Hamiltonien. `VQE` avec `Estimator`. Comparaison des résultats |
| **Notions** | VQE, Hamiltonien moléculaire, ansatz UCCSD, état fondamental, chimie quantique |

#### Laboratoire 10 — QAOA pour MaxCut (Semaine 11)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Résoudre un problème d'optimisation combinatoire par QAOA |
| **PennyLane** | Circuit QAOA : couches de phase separator + mixer. Optimisation des angles (β, γ). Approximation ratio. Graphes 3-réguliers |
| **Qiskit** | `QAOA` de `qiskit-optimization`. Comparaison avec solveur classique (brute force, greedy) |
| **Notions** | QAOA, MaxCut, approximation ratio, couches, optimisation combinatoire |

#### Laboratoire 11 — Non-unitaire QML avec LCU (Semaine 10)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Implémenter une couche non-unitaire via Linear Combination of Unitaries |
| **PennyLane** | Circuit LCU : ancilla + post-sélection. Comparaison unitaire vs. non-unitaire. Analyse de la Fisher efficiency |
| **Notions** | LCU, non-unitaire, post-sélection, ancilla, Fisher information, expressivité |

#### Laboratoire 12 — Benchmark multi-plateforme (Semaine 13)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Déployer un même circuit QML sur plusieurs plateformes et comparer |
| **Amazon Braket** | Soumission sur IonQ (ions piégés) et Rigetti (supraconducteur). Comparaison des fidélités |
| **IBM Quantum** | Soumission sur `ibm_brisbane` ou `ibm_torino`. Analyse du bruit, readout error, gate error |
| **PennyLane + plugins** | Même circuit sur backends simulateur, IBM, Braket. Analyse comparative |
| **Notions** | NISQ, calibration, bruit matériel, multi-plateforme, reproductibilité |

#### Laboratoire 13 — Projet intégrateur (Semaine 14)

| Aspect | Détail |
|--------|--------|
| **Objectifs** | Réaliser un projet complet de bout en bout |
| **Choix** | 1) Kernel quantique à 40+ qubits avec BFT, 2) Transfer learning quantique sur dataset médical, 3) QAOA pour optimisation financière, 4) Benchmark systématique QML vs. classique sur 5 datasets |

### 7.3 Projets

#### Projet de mi-parcours (Semaine 7 — 20 % de la note)

Implémentation complète d'un pipeline QML avec analyse comparative :
- **Choix 1 — QSVM vs. SVM** : Kernel quantique avec alignment, benchmark sur 3 datasets, analyse de la concentration
- **Choix 2 — VQC avec analyse de bruit** : Classifieur variationnel sous modèles de bruit réalistes (QuTiP/Qiskit), comparaison des stratégies de mitigation

#### Projet final (Semaine 14 — 30 % de la note)

Au choix parmi :

1. **Kernel quantique à l'échelle** : Implémentation d'un fidelity kernel avec BFT sur 40+ qubits, expérience sur IBM Quantum, analyse de la concentration exponentielle
2. **Architecture hybride complète** : Transfer learning (CNN + VQC) sur un dataset réel (Hymenoptera, PlantVillage, PathMNIST), benchmark classique vs. quantique
3. **QAOA application industrielle** : Optimisation combinatoire pour la finance (portefeuille) ou la logistique, comparaison avec recuit simulé et branch-and-bound
4. **Benchmark multi-framework multi-plateforme** : Même modèle QML implémenté en PennyLane, Qiskit ML, et Cirq, exécuté sur IBM, IonQ, et Rigetti. Analyse des coûts, fidélité, et performance
5. **Non-unitaire QML** : Implémentation LCU pour une tâche de classification, analyse de la Fisher efficiency transition (article de référence [LCU26])

---

## 8. Évaluation

| Élément | Poids | Description |
|---------|-------|-------------|
| **Devoirs** (×4) | 30 % | Problèmes théoriques + implémentations QML |
| **Quiz** (×2) | 10 % | Validation des connaissances : théorie de l'apprentissage, encodage, barren plateaux |
| **Projet de mi-parcours** | 20 % | Pipeline QML complet : encodage → VQC/QSVM → benchmark → analyse |
| **Projet final** | 30 % | Projet intégrateur : architecture hybride, kernel à l'échelle, ou application industrielle |
| **Présentation orale** | 10 % | Analyse critique d'un article de recherche récent (2025–2026) parmi une liste fournie |

---

## 9. Ressources complémentaires

### Cours en ligne et vidéos
- **Qiskit Machine Learning** — qiskit.org/learn — Tutoriels officiels IBM
- **PennyLane QML Demos** — pennylane.ai/qml — Démonstrations interactives (Xanadu)
- **IBM Quantum Learning** — learning.quantum.ibm.com
- **MIT 18.435J** — Quantum Computation (Peter Shor) — MIT OCW
- **Caltech Ph229** — Quantum Computation (John Preskill) — theory.caltech.edu/~preskill/ph229
- **NTNU MA3001** — Quantum Machine Learning (2026) — wiki.math.ntnu.no

### Logiciels et SDK
- **PennyLane** (Xanadu) — github.com/PennyLaneAI/pennylane — QML, différentiation automatique, VQE, QAOA
- **Qiskit Machine Learning** (IBM) — github.com/Qiskit/qiskit-machine-learning — QSVM, QNN, qGAN
- **Qiskit** (IBM) — github.com/Qiskit/qiskit — SDK circuits, Aer, accès IBM
- **Cirq** (Google) — github.com/quantumlib/Cirq — Circuits NISQ, optimisation
- **Stim** (Google) — github.com/quantumlib/Stim — Simulation stabilisateur
- **Qualtran** (Google) — github.com/quantumlib/Qualtran — Algorithmes tolérants aux fautes
- **Amazon Braket** — github.com/amazon-braket/amazon-braket-sdk-python — Accès multi-plateforme
- **Cuda-Q** (NVIDIA) — github.com/NVIDIA/cuda-quantum — Simulation GPU-accélérée

### Conférences et séminaires
- **QIP** (Quantum Information Processing) — conférence annuelle
- **QML** (Quantum Machine Learning) — workshops spécialisés (NeurIPS, ICML, AAAI)
- **IEEE QCE** (IEEE International Conference on Quantum Computing and Engineering)
- **APS March Meeting** — sessions QML
- **Xanadu QML Seminar Series** — séminaires en ligne mensuels

---

## 10. Structure du dépôt

```
cours/                              # Contenu théorique (Markdown + LaTeX)
├── partie1-fondements/            # Séances 1-3 (ML classique + rappels quantiques)
├── partie2-encodage-variationnel/ # Séances 4-6 (encodage, VQC, barren plateaux)
├── partie3-kernels/               # Séances 7-8 (kernels quantiques, concentration, BFT)
├── partie4-hybrides/              # Séances 9-11 (transfer learning, QCNN, Transformers, non-unitaire)
└── partie5-materiel-applications/ # Séances 12-14 (bruit, plateformes, applications)

labs/                               # Travaux pratiques (Jupyter notebooks)
├── lab1-bloch-circuit/
├── lab2-classical-baseline/
├── lab3-encoding-feature-maps/
├── lab4-vqc-classifier/
├── lab5-barren-plateaux/
├── lab6-qsvm/
├── lab7-transfer-learning/
├── lab8-qcnn/
├── lab9-vqe-chemistry/
├── lab10-qaoa-maxcut/
├── lab11-non-unitary-lcu/
├── lab12-multi-platform/
└── lab13-integration-project/

code/                               # Scripts Python et utilitaires
exercices/                          # Devoirs
references/                         # Ressources complémentaires
syllabus-quantum-machine-learning.md # Syllabus détaillé original
```

---

*Dernière mise à jour : Juin 2026*

*Ce syllabus intègre les résultats de recherche les plus récents issus de Nature, npj Quantum Information, Quantum Machine Intelligence, Discover Computing, Physical Review Letters, arXiv, et des conférences QIP, NeurIPS, ICML et IEEE QCE. Les références aux publications de Xanadu (PennyLane), IBM Quantum, Google Quantum AI, Amazon Braket, et Microsoft Quantum sont citées tout au long du cours.*
