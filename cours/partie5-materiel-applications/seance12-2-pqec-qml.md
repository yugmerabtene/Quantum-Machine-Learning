# Séance 12.2 — Partial Quantum Error Correction (pQEC) pour QML

## De NISQ à Early Fault-Tolerant : une transition progressive

La correction d'erreur quantique complète (FT) nécessite des codes avec des overheads matériels considérables : le code de surface nécessite ~$10^3$–$10^4$ qubits physiques par qubit logique. Sur les processeurs actuels (433–1056 qubits), ce rapport est inaccessible. Le régime **Early Fault-Tolerant (EFT)** — où l'on dispose de quelques qubits logiques imparfaits mais avec des taux d'erreur réduits — émerge comme une étape intermédiaire cruciale.

$$
\begin{array}{c}
\text{NISQ} \\
\begin{array}{c}
\text{50--1000 qubits physiques} \\
\text{pas de correction} \\
\text{bruit $\sim 10^{-2}$--$10^{-3}$}
\end{array}
\end{array}
\longrightarrow
\begin{array}{c}
\text{EFT} \\
\begin{array}{c}
\text{1--10 qubits logiques} \\
\text{correction partielle} \\
\text{bruit $\sim 10^{-4}$--$10^{-6}$}
\end{array}
\end{array}
\longrightarrow
\begin{array}{c}
\text{FT} \\
\begin{array}{c}
\text{100+ qubits logiques} \\
\text{correction complète} \\
\text{bruit $\sim 10^{-12}$}
\end{array}
\end{array}
$$

La **Partial Quantum Error Correction (pQEC)** est la stratégie qui consiste à n'appliquer la correction d'erreur que sur les parties les plus sensibles du circuit QML, acceptant un bruit résiduel sur le reste.

## Le code [[4,2,2]] : détection d'erreur pour VQC

Le code [[4,2,2]] est un code de détection d'erreur stabilisateur qui encode $k=2$ qubits logiques dans $n=4$ qubits physiques, avec distance $d=2$ (détecte 1 erreur, n'en corrige pas).

### Définition stabilisatrice

Les générateurs du stabilisateur sont :

$$
S_1 = X \otimes X \otimes X \otimes X,\quad S_2 = Z \otimes Z \otimes Z \otimes Z
$$

L'espace de code est l'espace $+1$ commun aux deux stabilisateurs. Les états logiques sont :

$$
|\bar{0}\rangle = \frac{1}{\sqrt{2}} (|0000\rangle + |1111\rangle),\quad
|\bar{1}\rangle = \frac{1}{\sqrt{2}} (|1100\rangle + |0011\rangle)
$$

Pour un VQC, l'encodage se fait au début du circuit, et le syndrome (les valeurs de $S_1$ et $S_2$) est mesuré à la fin :

```python
import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=6)  # 4 données + 2 ancilla

def encode_logical():
    """Encodage [[4,2,2]]."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[0, 3])

def measure_syndrome():
    """Mesure des stabilisateurs via ancilla."""
    # Syndrome X (S_1)
    qml.Hadamard(wires=4)
    qml.CNOT(wires=[0, 4])
    qml.CNOT(wires=[1, 4])
    qml.CNOT(wires=[2, 4])
    qml.CNOT(wires=[3, 4])
    qml.Hadamard(wires=4)
    # Syndrome Z (S_2)
    qml.CNOT(wires=[0, 5])
    qml.CNOT(wires=[1, 5])
    qml.CNOT(wires=[2, 5])
    qml.CNOT(wires=[3, 5])

@qml.qnode(dev)
def vqc_pqec(x, params):
    encode_logical()
    # Opérations logiques du VQC sur les qubits 0-1 (logiques)
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    # Syndrome
    measure_syndrome()
    return qml.probs(wires=[0, 1, 4, 5])

# Post-sélection : on ne garde que les runs avec syndrome |00>
# (pas de détection d'erreur)
x_test = np.array([0.5, -0.3])
params = np.array([0.1, 0.2])
result = vqc_pqec(x_test, params)
probs_accept = result[0] / result[0:4].sum()  # renormalisation
```

## Résultats : amélioration de la fidélité ×9.27

Les benchmarks expérimentaux de [VQA26] montrent que l'utilisation du code [[4,2,2]] avec post-sélection sur des circuits VQC améliore la fidélité moyenne d'un facteur **×9.27** par rapport au circuit non protégé, au prix d'un overhead :

| Métrique | Sans pQEC | Avec pQEC [[4,2,2]] | Amélioration |
|----------|-----------|---------------------|--------------|
| Fidélité moyenne | 0.083 | 0.769 | **×9.27** |
| Probabilité d'acceptation | 1.0 | 0.42 | — |
| Accuracy VQC | 71.3 % | 83.1 % | +11.8 pp |

La probabilité d'acceptation de 42 % signifie que 58 % des exécutions sont rejetées car une erreur est détectée. Ce rejet réduit le nombre d'échantillons utiles, mais la qualité des échantillons restants est bien supérieure.

## Probabilistic Error Cancellation + pQEC : compromis espace-temps

La combinaison de PEC (mitigation) et pQEC (correction) offre un spectre de compromis entre overhead spatial (qubits supplémentaires) et temporel (shots supplémentaires).

Soit un circuit de profondeur $L$ avec un taux d'erreur par porte $p$. On dispose de $n_q$ qubits pour la pQEC et de $n_s$ shots pour la PEC. L'erreur résiduelle suit :

$$
\varepsilon_{\text{résiduel}}(n_q, n_s) = \varepsilon_{\text{pQEC}}(n_q) + \varepsilon_{\text{PEC}}(n_s) + \varepsilon_{\text{croisé}}
$$

où $\varepsilon_{\text{pQEC}}(n_q) \propto \exp(-\alpha n_q)$ (décroissance exponentielle avec les qubits de code) et $\varepsilon_{\text{PEC}}(n_s) \propto \gamma^L / \sqrt{n_s}$ (convergence Monte Carlo).

```python
def overhead_tradeoff(L, p, target_fidelity, n_physical):
    """Calcule le nombre de shots PEC nécessaire selon l'espace alloué à pQEC."""
    import numpy as np
    gamma = 1 + 2*p/3
    eps_pqec = np.exp(-0.5 * n_physical / 4)  # approximation
    n_shots = (gamma**L)**2 / ((target_fidelity - (1 - eps_pqec))**2)
    return n_shots
```

| Stratégie | Qubits physiques | Shots requis (×$10^6$) | Fidélité atteinte |
|-----------|-----------------|----------------------|-------------------|
| Rien | 4 | 0.1 | 0.68 |
| PEC seule | 4 | 24.3 | 0.89 |
| pQEC seule | 8 | 0.24 | 0.83 |
| PEC + pQEC (équilibré) | 8 | 3.2 | 0.95 |

Le gain est manifeste : la combinaison **PEC + pQEC** atteint une fidélité de 0.95 avec 8 qubits et 3.2M shots, là où PEC seule nécessiterait 24.3M shots.

## Implications pour les architectures QML

### Qubits logiques et overhead

L'utilisation de la correction d'erreur dans les VQC a des implications architecturales profondes :

1. **Profondeur de circuit limitée** : les portes logiques sont plus longues que les portes physiques (factor ~5–50×). Un VQC conçu pour 10 couches physiques pourrait n'en supporter que 1–2 en logique.
2. **Goulot d'étranglement de mesure** : la mesure des syndromes nécessite des ancillas et augmente le temps de cycle.
3. **Reconception des ansätze** : les ansätze doivent être exprimés dans le jeu de portes logiques disponible (généralement Clifford + $T$), ce qui exclut les portes continues $R_X(\theta)$, $R_Y(\theta)$ utilisées dans les VQC.

### Adaptation des VQC au régime EFT

Une approche prometteuse est le *logical VQC* où seuls les portes les plus critiques sont protégées :

```python
def logical_rz(theta, qubit, ancilla):
    """Porte RZ logique implémentée via état auxiliaire + correction."""
    qml.RZ(theta, wires=qubit)
    # Syndrome de parité sur l'ancilla
    qml.CNOT(wires=[qubit, ancilla])
    qml.Hadamard(wires=ancilla)
    # Post-sélection si syndrome non trivial
    # ... (détection d'erreur)
```

### Feuille de route

Le passage à l'EFT pour le QML suit un calendrier prévisionnel :

- **2026–2027** : pQEC + mitigation sur ~50 qubits physiques, démonstration de fidélité ×10 sur VQC 4 qubits
- **2027–2028** : premiers qubits logiques utilisés dans des circuits variationnels (Google, IBM)
- **2028–2029** : avantage quantique pratique avec ~10 qubits logiques et correction partielle
- **2030+** : Full FT avec 100+ qubits logiques, VQC et kernels à l'échelle

## Références

- [VQA26] « A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing. » *arXiv:2604.07909*, 2026.
- [NOI26] « NOISE-VQA: Convergence and Complexity Analysis of VQA with Finite-Shot and Biased Oracles. » *Journal of Computational and Applied Mathematics*, Juin 2026.
- [Rev26] « A review of quantum machine learning algorithms, applications, and emerging advantages. » *Discover Computing*, Avril 2026.
- [Fra26] « A Framework-Agnostic Quantum Neural Network Architecture. » *arXiv:2604.04414*, Avril 2026.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
