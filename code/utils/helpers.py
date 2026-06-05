import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(losses, title="Courbe d'apprentissage", xlabel="It\u00e9ration", ylabel="Perte"):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def accuracy(predictions, targets):
    return np.mean(predictions == targets)


def plot_confusion_matrix(cm, classes, title="Matrice de confusion"):
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Vrai')
    plt.xlabel('Pr\u00e9dit')
    plt.show()


def plot_comparison(classical_acc, quantum_acc, labels=None, title="Comparaison Classique vs Quantique"):
    if labels is None:
        labels = ['Classique', 'Quantique']
    plt.figure(figsize=(6, 4))
    plt.bar(labels, [classical_acc, quantum_acc], color=['#3498db', '#e74c3c'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1)
    for i, v in enumerate([classical_acc, quantum_acc]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.show()


def plot_encoding_comparison(encodings, qubits, title="Comparaison des encodages"):
    plt.figure(figsize=(8, 5))
    plt.bar(encodings, qubits, color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22'], alpha=0.7)
    plt.ylabel('Nombre de qubits')
    plt.title(title)
    for i, v in enumerate(qubits):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.show()


def generate_maxcut_graph(n_nodes=6, seed=42):
    import networkx as nx
    np.random.seed(seed)
    G = nx.random_regular_graph(3, n_nodes, seed=seed)
    return G


def hamming_weight(s):
    return sum(c == '1' for c in s)


def plot_quantum_circuit(circuit, title="Circuit quantique"):
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        circuit.draw(output='mpl', ax=ax)
        plt.title(title)
        plt.show()
    except Exception:
        print(circuit)
