import pennylane as qml
from pennylane import numpy as np


def h2_hamiltonian():
    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=0)
    return H, qubits


def vqe_circuit(params, wires, H, ansatz='basic'):
    if ansatz == 'basic':
        qml.BasicEntanglerLayers(params, wires=wires)
    elif ansatz == 'uccsd':
        qml.UCCSD(params, wires=wires,)
    return qml.expval(H)


def compute_ground_state_energy(H, n_qubits, n_layers=2, steps=100):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def cost_fn(params):
        return vqe_circuit(params, wires=range(n_qubits), H=H)

    shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_qubits=n_qubits)
    params = np.random.random(shape)
    opt = qml.AdamOptimizer(stepsize=0.1)

    for _ in range(steps):
        params, energy = opt.step_and_cost(cost_fn, params)

    return energy, params
