import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, List, Optional, Callable
import matplotlib.pyplot as plt


def h2_hamiltonian(bond_length: float = 0.6614) -> Tuple[qml.Hamiltonian, int]:
    """Construit l'hamiltonien de la molécule H₂.

    Args:
        bond_length: Distance inter-nuléaire en angströms (défaut 0.6614).

    Returns:
        Tuple (H, n_qubits) où H est l'hamiltonien et n_qubits le nombre de qubits.
    """
    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -bond_length, 0.0, 0.0, bond_length])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=0)
    return H, qubits


def vqe_circuit(params, wires, H, ansatz: str = 'basic') -> float:
    """Circuit VQE avec choix d'ansatz.

    Args:
        params: Paramètres du circuit.
        wires: Fils quantiques.
        H: Hamiltonien à minimiser.
        ansatz: Type d'ansatz parmi 'basic', 'uccsd', 'hardware_efficient'.

    Returns:
        Valeur moyenne de l'hamiltonien.
    """
    if ansatz == 'basic':
        qml.BasicEntanglerLayers(params, wires=wires)
    elif ansatz == 'uccsd':
        qml.UCCSD(params, wires=wires)
    elif ansatz == 'hardware_efficient':
        qml.HardwareEfficientAnsatz(
            wires=wires,
            rot=qml.RY
        )(params)
    else:
        raise ValueError(f"Ansatz '{ansatz}' non supporté. Choisir parmi 'basic', 'uccsd', 'hardware_efficient'.")
    return qml.expval(H)


def compute_ground_state_energy(
    H: qml.Hamiltonian,
    n_qubits: int,
    n_layers: int = 2,
    steps: int = 100,
    ansatz: str = 'basic',
    stepsize: float = 0.1,
    verbose: bool = False
) -> Tuple[float, np.ndarray, List[float]]:
    """Calcule l'énergie de l'état fondamental par VQE.

    Args:
        H: Hamiltonien du système.
        n_qubits: Nombre de qubits.
        n_layers: Nombre de couches de l'ansatz.
        steps: Nombre de pas d'optimisation.
        ansatz: Type d'ansatz ('basic', 'uccsd', 'hardware_efficient').
        stepsize: Pas d'apprentissage de l'optimiseur.
        verbose: Afficher la progression.

    Returns:
        Tuple (energy, optimal_params, energy_history).
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def cost_fn(params):
        return vqe_circuit(params, wires=range(n_qubits), H=H, ansatz=ansatz)

    if ansatz == 'basic':
        shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_qubits=n_qubits)
    elif ansatz == 'uccsd':
        shape = (n_layers, n_qubits * 2)
    elif ansatz == 'hardware_efficient':
        shape = qml.HardwareEfficientAnsatz.shape(n_layers=n_layers, n_qubits=n_qubits)
    else:
        shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_qubits=n_qubits)

    params = np.random.random(shape, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=stepsize)

    energy_history = []
    for i in range(steps):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy_history.append(energy)
        if verbose and (i + 1) % 10 == 0:
            print(f"  Étape {i+1}/{steps} — Énergie: {energy:.6f} Ha")

    return energy, params, energy_history


def plot_dissociation_curve(
    bond_lengths: Optional[List[float]] = None,
    n_layers: int = 2,
    steps: int = 100,
    ansatz: str = 'basic',
    stepsize: float = 0.05
) -> None:
    """Trace la courbe de dissociation H₂ (énergie vs distance).

    Calcule l'énergie VQE pour différentes longueurs de liaison et affiche
    la courbe de dissociation comparée à la solution exacte.

    Args:
        bond_lengths: Liste des distances inter-nuléaires en Å.
        n_layers: Nombre de couches de l'ansatz.
        steps: Nombre de pas d'optimisation par point.
        ansatz: Type d'ansatz.
        stepsize: Pas d'apprentissage.
    """
    if bond_lengths is None:
        bond_lengths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    vqe_energies = []
    exact_energies = []

    for bl in bond_lengths:
        H, n_qubits = h2_hamiltonian(bond_length=bl)
        energy_vqe, _, _ = compute_ground_state_energy(
            H, n_qubits, n_layers=n_layers, steps=steps,
            ansatz=ansatz, stepsize=stepsize, verbose=False
        )
        vqe_energies.append(energy_vqe)
        exact_energies.append(qml.eigvals(H)[0].real)

    plt.figure(figsize=(10, 6))
    plt.plot(bond_lengths, vqe_energies, 'o-', label='VQE', color='#e74c3c', markersize=6)
    plt.plot(bond_lengths, exact_energies, 's--', label='Exact', color='#3498db', markersize=4)
    plt.xlabel('Distance inter-nuléaire (Å)')
    plt.ylabel('Énergie (Ha)')
    plt.title(f'Courbe de dissociation H₂ — Ansatz: {ansatz}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    idx_min = np.argmin(vqe_energies)
    print(f"Distance d'équilibre VQE : {bond_lengths[idx_min]:.3f} Å")
    print(f"Énergie minimale VQE    : {vqe_energies[idx_min]:.6f} Ha")
    idx_exact = np.argmin(exact_energies)
    print(f"Distance d'équilibre exacte : {bond_lengths[idx_exact]:.3f} Å")
    print(f"Énergie minimale exacte    : {exact_energies[idx_exact]:.6f} Ha")


def hardware_efficient_ansatz(
    n_qubits: int,
    n_layers: int = 2,
    rot_gate: str = 'RY',
    entangling_gate: str = 'CNOT'
) -> Tuple[int, Callable]:
    """Crée un ansatz hardware-efficient personnalisé.

    Args:
        n_qubits: Nombre de qubits.
        n_layers: Nombre de couches.
        rot_gate: Porte de rotation ('RY', 'RZ', 'RX').
        entangling_gate: Porte d'intrication ('CNOT', 'CZ').

    Returns:
        Tuple (nombre_de_paramètres, fonction_du_circuit).
    """
    rot_gates = {'RY': qml.RY, 'RZ': qml.RZ, 'RX': qml.RX}
    ent_gates = {'CNOT': qml.CNOT, 'CZ': qml.CZ}

    if rot_gate not in rot_gates:
        raise ValueError(f"Porte de rotation '{rot_gate}' non supportée.")
    if entangling_gate not in ent_gates:
        raise ValueError(f"Porte d'intrication '{entangling_gate}' non supportée.")

    R = rot_gates[rot_gate]
    ENT = ent_gates[entangling_gate]

    n_params = n_layers * n_qubits * 2
    dev = qml.device('default.qubit', wires=n_qubits)

    def circuit(params, H):
        @qml.qnode(dev)
        def _circuit(p):
            idx = 0
            for _ in range(n_layers):
                for w in range(n_qubits):
                    R(p[idx], wires=w)
                    idx += 1
                for w in range(n_qubits - 1):
                    ENT(wires=[w, w + 1])
                for w in range(n_qubits):
                    R(p[idx], wires=w)
                    idx += 1
            return qml.expval(H)
        return _circuit(params)

    return n_params, circuit