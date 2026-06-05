# Apprentissage Automatique Quantique : Théorie, Algorithmes et Applications

**Niveau :** Master / Doctorat (M2 / PhD)
**Durée :** 28 séances (2 × 2h par semaine, 14 semaines)
**Langue :** Français

## Description

Ce cours propose une couverture complète de l'apprentissage automatique quantique (Quantum Machine Learning, QML), depuis les fondements mathématiques de la théorie de l'apprentissage et du calcul quantique jusqu'aux architectures hybrides les plus avancées, aux kernels quantiques, aux réseaux de neurones variationnels, et aux applications industrielles. Il s'appuie sur les développements les plus récents (2024–2026) — notamment les avancées en kernels quantiques covarants (IBM 156 qubits, *npj Quantum Information* 2026), les réseaux de neurones non-unitaires (LCU, 2026), le Quantum LEGO Learning, les Quantum Transformers, les méthodes spectrales (Xanadu/PennyLane 2026), et les benchmarks systématiques QML vs. classique (*Discover Computing* 2026).

## Plan du cours

### Partie I : Fondements — Apprentissage classique et quantique (Séances 1–3)

| Séance | Sujet |
|--------|-------|
| 1.1 | Introduction historique et panorama |
| 1.2 | Théorie de l'apprentissage automatique (rappel) |
| 2.1 | Algèbre linéaire et RKHS |
| 2.2 | Calcul quantique (rappel intensif) |
| 3.1 | Réseaux de neurones classiques (rappel) |
| 3.2 | Méthodes à noyau classiques (rappel) |

### Partie II : Encodage et circuits variationnels (Séances 4–6)

| Séance | Sujet |
|--------|-------|
| 4.1 | Encodage de données classiques en états quantiques |
| 4.2 | Circuits quantiques paramétrés (PQC/VQC) |
| 5.1 | Variational Quantum Classifier (VQC) |
| 5.2 | Différentiation automatique des circuits quantiques |
| 6.1 | Barren Plateaux et problèmes d'entraînement |
| 6.2 | ULSP et optimisation avancée (NOM) |

### Partie III : Kernels quantiques (Séances 7–8)

| Séance | Sujet |
|--------|-------|
| 7.1 | Quantum Kernel Methods (QKM) |
| 7.2 | Quantum Support Vector Machine (QSVM) |
| 8.1 | Concentration exponentielle des kernels |
| 8.2 | Mitigation : Bit-Flip Tolerance et Quantum Kernel Alignment |

### Partie IV : Architectures hybrides avancées (Séances 9–11)

| Séance | Sujet |
|--------|-------|
| 9.1 | Hybrid classical-quantum architectures (Transfer Learning, LEGO) |
| 9.2 | Quantum Convolutional Neural Networks (QCNN) |
| 10.1 | Quantum Transformers |
| 10.2 | VQC-MLPNet et architectures non-unitaires (LCU) |
| 11.1 | Quantum Generative Models (qGAN, Born Machines) |
| 11.2 | Quantum Reinforcement Learning (QRL) et Federated Learning (QFL) |

### Partie V : Correction d'erreur, mitigation et matériel (Séances 12–14)

| Séance | Sujet |
|--------|-------|
| 12.1 | Bruit et mitigation pour QML |
| 12.2 | Partial Quantum Error Correction (pQEC) pour QML |
| 13.1 | Plateformes matérielles pour QML |
| 13.2 | Déploiement cloud et multi-plateforme |
| 14.1 | Applications industrielles du QML |
| 14.2 | Défis ouverts et horizon 2026–2030 |

### Laboratoires

| Lab | Sujet | Bibliothèque |
|-----|-------|-------------|
| 1 | Sphère de Bloch et premiers circuits | PennyLane, Qiskit |
| 2 | Classifieur classique (baseline) | scikit-learn, PyTorch |
| 3 | Encodage de données et feature maps | PennyLane, Qiskit ML |
| 4 | Variational Quantum Classifier (VQC) | PennyLane, Qiskit ML |
| 5 | Barren Plateaux et analyse de trainabilité | PennyLane |
| 6 | Quantum Support Vector Machine (QSVM) | PennyLane, Qiskit ML |
| 7 | Transfer Learning quantique | PyTorch + PennyLane/Qiskit |
| 8 | Quantum Convolutional Neural Network (QCNN) | PennyLane |
| 9 | VQE et chimie quantique | PennyLane, Qiskit Nature |
| 10 | QAOA pour MaxCut | PennyLane, Qiskit |
| 11 | Non-unitaire QML avec LCU | PennyLane |
| 12 | Benchmark multi-plateforme | Braket, IBM, PennyLane |
| 13 | Projet intégrateur | Libre |

## Prérequis

- Algèbre linéaire (espaces vectoriels complexes, produits tensoriels, valeurs propres)
- Probabilités et statistiques de base
- Notions de calcul quantique recommandées mais non obligatoires
- Programmation en Python (NumPy, PyTorch)
- Notions de base en apprentissage automatique

## Bibliothèques Python utilisées

| Bibliothèque | Rôle |
|---|---|
| **PennyLane** | Cœur du cours : différentiation automatique, VQC, kernels |
| **Qiskit ML** | Kernels quantiques, QSVM, SamplerQNN, accès IBM |
| **PyTorch / TensorFlow** | Backend classique pour architectures hybrides |
| **Cirq** | Circuits NISQ, optimisation |
| **Stim** | Simulation stabilisateur pour correction d'erreur |
| **Amazon Braket** | Accès multi-plateforme (IonQ, Rigetti, QuEra) |
| **scikit-learn** | SVM classique, métriques, pipelines |

## Installation

```bash
pip install pennylane qiskit-machine-learning qiskit qiskit-aer cirq stim amazon-braket-sdk scikit-learn torch torchvision
```

## Structure du dépôt

```
cours/                              # Contenu théorique (Markdown + LaTeX)
├── partie1-fondements/
├── partie2-encodage-variationnel/
├── partie3-kernels/
├── partie4-hybrides/
└── partie5-materiel-applications/

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

## Évaluation

| Élément | Poids |
|---------|-------|
| Devoirs (×4) | 30 % |
| Quiz (×2) | 10 % |
| Projet de mi-parcours | 20 % |
| Projet final | 30 % |
| Présentation orale | 10 % |

## Références principales

### Ouvrages fondamentaux
- [SP21] Schuld & Petruccione, *Machine Learning with Quantum Computers.* Springer, 2021.
- [DWQ25] Du et al., *A Gentle Introduction to Quantum Machine Learning.* Springer, 2025.
- [NC00] Nielsen & Chuang, *Quantum Computation and Quantum Information.* Cambridge, 2000.
- [Bis06] Bishop, *Pattern Recognition and Machine Learning.* Springer, 2006.

### Articles récents (2024–2026)
- [Agl26] Agliardi et al., « Mitigating exponential concentration in covariant quantum kernels. » *npj Quantum Information* 12, 12 (2026).
- [LCU26] « Non-Unitary QML: Fisher Efficiency Transitions. » *arXiv:2603.27377* (2026).
- [Tra26] « Quantum Transformers for Image Classification. » *Quantum Machine Intelligence* 8, 43 (2026).
- [Pen26] Xanadu, « Why quantum computers could be great for ML after all. » *PennyLane Blog* (2026).
- [Rev26] « A review of QML algorithms, applications, and emerging advantages. » *Discover Computing* (2026).
- [VQA26] « A Review of VQAs: Insights into Fault-Tolerant QC. » *arXiv:2604.07909* (2026).

## Licence

Ce cours est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.
