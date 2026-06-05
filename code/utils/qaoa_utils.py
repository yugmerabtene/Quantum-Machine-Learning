import pennylane as qml
import networkx as nx


def maxcut_hamiltonian(graph):
    coeffs = []
    obs = []
    for edge in graph.edges:
        coeffs.append(0.5)
        obs.append(qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
    H = qml.Hamiltonian(coeffs, obs)
    return H


def qaoa_circuit(params, graph, n_layers):
    n_qubits = len(graph.nodes)
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
    return qml.expval(maxcut_hamiltonian(graph))


def compute_approximation_ratio(cut_value, max_cut):
    return cut_value / max_cut if max_cut > 0 else 0
