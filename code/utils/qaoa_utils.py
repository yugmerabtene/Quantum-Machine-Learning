import pennylane as qml
import networkx as nx
import numpy as np
from typing import Tuple, List, Optional, Dict
from itertools import product
import matplotlib.pyplot as plt


def maxcut_hamiltonian(graph: nx.Graph) -> qml.Hamiltonian:
    """Construit l'hamiltonien MaxCut à partir d'un graphe.

    L'hamiltonien MaxCut est H = Σ_{(i,j)∈E} ½(I - Z_i Z_j).

    Args:
        graph: Graphe NetworkX non orienté.

    Returns:
        Hamiltonien PennyLane pour le problème MaxCut.
    """
    coeffs = []
    obs = []
    for edge in graph.edges:
        coeffs.append(0.5)
        obs.append(qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
    H = qml.Hamiltonian(coeffs, obs)
    return H


def qaoa_circuit(params, graph: nx.Graph, n_layers: int) -> float:
    """Circuit QAOA pour le problème MaxCut.

    Args:
        params: Paramètres QAOA (2*n_layers : gamma_1, beta_1, gamma_2, beta_2, ...).
        graph: Graphe du problème.
        n_layers: Nombre de couches QAOA (profondeur p).

    Returns:
        Valeur moyenne de l'hamiltonien MaxCut.
    """
    n_qubits = len(graph.nodes)
    H = maxcut_hamiltonian(graph)
    for w in range(n_qubits):
        qml.Hadamard(wires=w)
    for layer in range(n_layers):
        gamma, beta = params[2 * layer], params[2 * layer + 1]
        for edge in graph.edges:
            qml.CNOT(wires=[edge[0], edge[1]])
            qml.RZ(2 * gamma, wires=edge[1])
            qml.CNOT(wires=[edge[0], edge[1]])
        for w in range(n_qubits):
            qml.RX(2 * beta, wires=w)
    return qml.expval(H)


def run_qaoa(
    graph: nx.Graph,
    n_layers: int = 2,
    steps: int = 100,
    stepsize: float = 0.1,
    verbose: bool = False
) -> Tuple[np.ndarray, float, List[float]]:
    """Exécute l'optimisation QAOA complète pour le problème MaxCut.

    Args:
        graph: Graphe du problème MaxCut.
        n_layers: Profondeur du circuit QAOA (nombre de couches).
        steps: Nombre de pas d'optimisation.
        stepsize: Pas d'apprentissage de l'optimiseur.
        verbose: Afficher la progression.

    Returns:
        Tuple (paramètres_optimaux, énergie_finale, historique_énergie).
    """
    n_qubits = len(graph.nodes)
    dev = qml.device('default.qubit', wires=n_qubits)
    H = maxcut_hamiltonian(graph)

    @qml.qnode(dev)
    def cost_fn(params):
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for layer in range(n_layers):
            gamma, beta = params[2 * layer], params[2 * layer + 1]
            for edge in graph.edges:
                qml.CNOT(wires=[edge[0], edge[1]])
                qml.RZ(2 * gamma, wires=edge[1])
                qml.CNOT(wires=[edge[0], edge[1]])
            for w in range(n_qubits):
                qml.RX(2 * beta, wires=w)
        return qml.expval(H)

    init_params = np.random.uniform(0, np.pi, 2 * n_layers, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=stepsize)

    energy_history = []
    params = init_params

    for i in range(steps):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy_history.append(energy)
        if verbose and (i + 1) % 10 == 0:
            print(f"  Étape {i+1}/{steps} — Énergie: {energy:.6f}")

    return params, energy, energy_history


def brute_force_maxcut(graph: nx.Graph) -> Tuple[float, Dict[str, int]]:
    """Résout MaxCut par force brute pour les petits graphes.

    Énumère toutes les coupes possibles et retourne la valeur maximale.
    La taille maximale recommandée est ~20 nœuds.

    Args:
        graph: Graphe NetworkX.

    Returns:
        Tuple (valeur_max_coupure, meilleure_partition) où la partition
        est un dictionnaire {nœud: groupe (0 ou 1)}.
    """
    n = len(graph.nodes)
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    max_cut_value = -1
    best_partition = {}

    for assignment in product([0, 1], repeat=n):
        cut_value = 0
        partition = {nodes[i]: assignment[i] for i in range(n)}
        for u, v in edges:
            if partition[u] != partition[v]:
                cut_value += 1
        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_partition = partition

    return float(max_cut_value), best_partition


def compute_approximation_ratio(cut_value: float, max_cut: float) -> float:
    """Calcule le ratio d'approximation QAOA vs optimal.

    Args:
        cut_value: Valeur de la coupure obtenue par QAOA.
        max_cut: Valeur de la coupure optimale (force brute).

    Returns:
        Ratio d'approximation dans [0, 1].
    """
    return cut_value / max_cut if max_cut > 0 else 0.0


def sample_qaoa_solution(
    graph: nx.Graph,
    params: np.ndarray,
    n_layers: int,
    n_shots: int = 1000
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Échantillonne les solutions QAOA et retourne la meilleure coupure.

    Args:
        graph: Graphe du problème.
        params: Paramètres QAOA optimisés.
        n_layers: Profondeur QAOA.
        n_shots: Nombre de mesures.

    Returns:
        Tuple (comptes_par_bitstring, meilleure_solution).
    """
    n_qubits = len(graph.nodes)
    dev = qml.device('default.qubit', wires=n_qubits, shots=n_shots)

    H = maxcut_hamiltonian(graph)

    @qml.qnode(dev)
    def sample_circuit(p):
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for layer in range(n_layers):
            gamma, beta = p[2 * layer], p[2 * layer + 1]
            for edge in graph.edges:
                qml.CNOT(wires=[edge[0], edge[1]])
                qml.RZ(2 * gamma, wires=edge[1])
                qml.CNOT(wires=[edge[0], edge[1]])
            for w in range(n_qubits):
                qml.RX(2 * beta, wires=w)
        return qml.counts(wires=range(n_qubits))

    counts = sample_circuit(params)
    edges = list(graph.edges)
    best_string = max(counts, key=lambda s: sum(
        1 for u, v in edges if s[list(graph.nodes).index(u)] != s[list(graph.nodes).index(v)]
    ))
    best_partition = {node: int(best_string[i]) for i, node in enumerate(graph.nodes)}

    return counts, best_partition


def plot_results(
    energy_history: List[float],
    graph: nx.Graph,
    best_partition: Optional[Dict] = None,
    figsize: Tuple[int, int] = (16, 5)
) -> None:
    """Visualise les résultats de l'optimisation QAOA.

    Affiche la convergence de l'énergie, le graphe coloré (si partition fournie),
    et un histogramme de la distribution des coupes.

    Args:
        energy_history: Historique des valeurs d'énergie pendant l'optimisation.
        graph: Graphe du problème.
        best_partition: Meilleure partition trouvée (optionnel).
        figsize: Taille de la figure.
    """
    n_plots = 3 if best_partition else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(energy_history, color='#2ecc71', linewidth=2)
    axes[0].set_xlabel("Itération")
    axes[0].set_ylabel("Énergie")
    axes[0].set_title("Convergence QAOA")
    axes[0].grid(True, alpha=0.3)

    if best_partition and n_plots >= 3:
        colors = ['#e74c3c' if best_partition[n] == 0 else '#3498db' for n in graph.nodes]
        nx.draw(graph, ax=axes[1], node_color=colors, with_labels=True,
                node_size=500, font_color='white', font_weight='bold')
        axes[1].set_title("Partition MaxCut")

        cut_sizes = []
        n = len(graph.nodes)
        nodes = list(graph.nodes)
        for bstr in product([0, 1], repeat=n):
            cut = sum(1 for u, v in graph.edges if bstr[nodes.index(u)] != bstr[nodes.index(v)])
            cut_sizes.append(cut)
        axes[2].hist(cut_sizes, bins=range(min(cut_sizes), max(cut_sizes) + 2),
                     color='#9b59b6', alpha=0.7, edgecolor='white')
        axes[2].set_xlabel("Taille de la coupure")
        axes[2].set_ylabel("Nombre de partitions")
        axes[2].set_title("Distribution des coupures")

    plt.tight_layout()
    plt.show()