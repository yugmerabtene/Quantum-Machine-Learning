import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
import warnings


def plot_training_curve(losses: List[float], title: str = "Courbe d'apprentissage",
                        xlabel: str = "Itération", ylabel: str = "Perte") -> None:
    """Trace la courbe d'apprentissage (loss vs itérations).

    Args:
        losses: Liste des valeurs de perte par itération.
        title: Titre du graphique.
        xlabel: Étiquette de l'axe x.
        ylabel: Étiquette de l'axe y.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calcule l'accuracy (proportion de prédictions correctes).

    Args:
        predictions: Prédictions du modèle.
        targets: Étiquettes réelles.

    Returns:
        Accuracy entre 0 et 1.
    """
    return float(np.mean(predictions == targets))


def plot_confusion_matrix(cm: np.ndarray, classes: List[str],
                          title: str = "Matrice de confusion") -> None:
    """Affiche une matrice de confusion avec annotations.

    Args:
        cm: Matrice de confusion de forme (n_classes, n_classes).
        classes: Noms des classes.
        title: Titre du graphique.
    """
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.show()


def plot_comparison(classical_acc: float, quantum_acc: float,
                    labels: Optional[List[str]] = None,
                    title: str = "Comparaison Classique vs Quantique") -> None:
    """Barre de comparaison classique vs quantique.

    Args:
        classical_acc: Accuracy du modèle classique.
        quantum_acc: Accuracy du modèle quantique.
        labels: Étiquettes personnalisées.
        title: Titre du graphique.
    """
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


def plot_encoding_comparison(encodings: List[str], qubits: List[int],
                             title: str = "Comparaison des encodages") -> None:
    """Compare le nombre de qubits requis par différents encodages.

    Args:
        encodings: Noms des encodages.
        qubits: Nombre de qubits correspondants.
        title: Titre du graphique.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(encodings, qubits, color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22'], alpha=0.7)
    plt.ylabel('Nombre de qubits')
    plt.title(title)
    for i, v in enumerate(qubits):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.show()


def generate_maxcut_graph(n_nodes: int = 6, seed: int = 42) -> Any:
    """Génère un graphe régulier aléatoire pour MaxCut.

    Args:
        n_nodes: Nombre de nœuds.
        seed: Graine aléatoire.

    Returns:
        Graphe NetworkX.
    """
    import networkx as nx
    np.random.seed(seed)
    G = nx.random_regular_graph(3, n_nodes, seed=seed)
    return G


def hamming_weight(s: str) -> int:
    """Calcule le poids de Hamming d'une chaîne binaire.

    Args:
        s: Chaîne binaire (e.g. '10110').

    Returns:
        Nombre de '1' dans la chaîne.
    """
    return sum(c == '1' for c in s)


def plot_quantum_circuit(circuit, title: str = "Circuit quantique") -> None:
    """Affiche le circuit quantique.

    Args:
        circuit: Circuit PennyLane ou Qiskit à dessiner.
        title: Titre du graphique.
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        circuit.draw(output='mpl', ax=ax)
        plt.title(title)
        plt.show()
    except Exception:
        print(circuit)


def plot_kernel_matrix(
    kernel_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Matrice de kernel quantique",
    figsize: Tuple[int, int] = (8, 7)
) -> None:
    """Visualise une matrice de kernel quantique avec heatmap.

    Affiche la matrice de kernel avec un code couleur et optionnellement
    les frontières entre les classes.

    Args:
        kernel_matrix: Matrice de kernel de forme (n, n).
        labels: Étiquettes des échantillons (optionnel, pour séparer les classes).
        title: Titre du graphique.
        figsize: Taille de la figure.
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        order = np.argsort(labels)
        K_sorted = kernel_matrix[np.ix_(order, order)]
        labels_sorted = labels[order]

        sns.heatmap(K_sorted, cmap='viridis', ax=ax, vmin=0, vmax=1)

        unique_labels = np.unique(labels_sorted)
        boundaries = []
        count = 0
        for lbl in unique_labels:
            count += np.sum(labels_sorted == lbl)
            boundaries.append(count - 0.5)
        for b in boundaries[:-1]:
            ax.axhline(y=b, color='white', linewidth=2)
            ax.axvline(x=b, color='white', linewidth=2)
    else:
        sns.heatmap(kernel_matrix, cmap='viridis', ax=ax, vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xlabel("Échantillon")
    ax.set_ylabel("Échantillon")
    plt.tight_layout()
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_accuracies: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    title: str = "Historique d'entraînement",
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """Trace les courbes de loss et d'accuracy par epoch.

    Affiche la loss d'entraînement et optionnellement les accuracy
    d'entraînement et de validation sur des sous-graphiques.

    Args:
        train_losses: Liste des losses d'entraînement.
        val_accuracies: Liste des accuracy de validation (optionnel).
        train_accuracies: Liste des accuracy d'entraînement (optionnel).
        title: Titre global du graphique.
        figsize: Taille de la figure.
    """
    has_accuracy = val_accuracies is not None or train_accuracies is not None
    n_plots = 2 if has_accuracy else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(train_losses, color='#e74c3c', linewidth=2, label='Train Loss')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if has_accuracy:
        if train_accuracies is not None:
            axes[1].plot(train_accuracies, color='#3498db', linewidth=2, label='Train Acc')
        if val_accuracies is not None:
            axes[1].plot(val_accuracies, color='#2ecc71', linewidth=2, label='Val Acc')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_classical_quantum(
    classical_results: Dict[str, float],
    quantum_results: Dict[str, float],
    metrics: Optional[List[str]] = None,
    title: str = "Comparaison Classique vs Quantique"
) -> None:
    """Affiche un tableau comparatif détaillé et un graphique radar.

    Compare les résultats classiques et quantiques avec un tableau
    formaté et une visualisation radar.

    Args:
        classical_results: Dictionnaire {métrique: valeur} classique.
        quantum_results: Dictionnaire {métrique: valeur} quantique.
        metrics: Liste des métriques à comparer (défaut : toutes les clés communes).
        title: Titre du graphique.
    """
    if metrics is None:
        metrics = list(set(classical_results.keys()) & set(quantum_results.keys()))

    print("=" * 65)
    print(f"{'Métrique':<25} {'Classique':<20} {'Quantique':<20}")
    print("=" * 65)
    for m in metrics:
        c_val = classical_results.get(m, float('nan'))
        q_val = quantum_results.get(m, float('nan'))
        if isinstance(c_val, float) and isinstance(q_val, float):
            print(f"{m:<25} {c_val:<20.4f} {q_val:<20.4f}")
        else:
            print(f"{m:<25} {str(c_val):<20} {str(q_val):<20}")
    print("=" * 65)

    n_metrics = len(metrics)
    if n_metrics >= 3:
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        c_vals = [classical_results.get(m, 0) for m in metrics]
        q_vals = [quantum_results.get(m, 0) for m in metrics]
        c_vals += c_vals[:1]
        q_vals += q_vals[:1]

        max_vals = [max(c, q) if max(c, q) > 0 else 1 for c, q in zip(c_vals[:-1], q_vals[:-1])]
        c_norm = [c / m for c, m in zip(c_vals[:-1], max_vals)] + [c_vals[0] / max_vals[0]]
        q_norm = [q / m for q, m in zip(q_vals[:-1], max_vals)] + [q_vals[0] / max_vals[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.fill(angles, c_norm, alpha=0.25, color='#3498db', label='Classique')
        ax.plot(angles, c_norm, 'o-', color='#3498db', linewidth=2)
        ax.fill(angles, q_norm, alpha=0.25, color='#e74c3c', label='Quantique')
        ax.plot(angles, q_norm, 'o-', color='#e74c3c', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(title, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(n_metrics)
        width = 0.35
        c_vals = [classical_results.get(m, 0) for m in metrics]
        q_vals = [quantum_results.get(m, 0) for m in metrics]
        ax.bar(x - width / 2, c_vals, width, label='Classique', color='#3498db')
        ax.bar(x + width / 2, q_vals, width, label='Quantique', color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()


def calculate_fisher_information(
    model: Any,
    data: np.ndarray,
    param_names: Optional[List[str]] = None,
    n_samples: int = 100
) -> np.ndarray:
    """Calcule la matrice d'information de Fisher empirique d'un modèle quantique.

    L'information de Fisher quantifie la sensibilité du modèle par rapport
    aux variations de ses paramètres. Utile pour l'analyse de paysage
    de loss et les méthodes LCU/NOM.

    Args:
        model: Modèle avec méthode .forward() et paramètres accessibles.
        data: Données d'entrée de forme (n_samples, n_features).
        param_names: Noms des paramètres à analyser (défaut : tous).
        n_samples: Nombre d'échantillons pour l'estimation.

    Returns:
        Matrice d'information de Fisher de forme (n_params, n_params).
    """
    import pennylane as qml

    if hasattr(model, 'parameters'):
        all_params = list(model.parameters())
    elif hasattr(model, 'qnode'):
        all_params = list(model.parameters())
    else:
        all_params = []

    if param_names is not None:
        params = [p for p in all_params if hasattr(p, 'name') and p.name in param_names]
    else:
        params = all_params

    n_params = len(params)
    fisher_matrix = np.zeros((n_params, n_params))

    eps = 1e-5
    for i in range(n_params):
        for j in range(i, n_params):
            grad_i_plus = []
            grad_i_minus = []
            for k in range(min(n_samples, len(data))):
                x = data[k] if k < len(data) else data[k % len(data)]

                p_plus_i = params[i] + eps if hasattr(params[i], '__add__') else params[i]
                p_minus_i = params[i] - eps if hasattr(params[i], '__sub__') else params[i]

                try:
                    out_plus = model(x)
                    out_minus = model(x)
                    grad = (out_plus - out_minus) / (2 * eps)
                    grad_i_plus.append(grad if isinstance(grad, (int, float)) else grad.item())
                except Exception:
                    grad_i_plus.append(0.0)

            fisher_matrix[i, j] = np.var(grad_i_plus) if grad_i_plus else 0.0
            fisher_matrix[j, i] = fisher_matrix[i, j]

    return fisher_matrix


def print_model_summary(
    model: Any,
    input_shape: Optional[Tuple[int, ...]] = None,
    detailed: bool = False
) -> None:
    """Affiche un résumé des paramètres du modèle.

    Affiche le nombre total de paramètres, les paramètres entraînables,
    et optionnellement la structure détaillée du modèle.

    Args:
        model: Modèle PyTorch ou PennyLane.
        input_shape: Forme d'entrée attendue (pour l'estimation de la taille).
        detailed: Afficher les détails par couche.
    """
    total_params = 0
    trainable_params = 0

    if hasattr(model, 'parameters'):
        for p in model.parameters():
            n = p.numel() if hasattr(p, 'numel') else 0
            total_params += n
            if not hasattr(p, 'requires_grad') or p.requires_grad:
                trainable_params += n
    else:
        try:
            total_params = len(list(model.parameters()))
            trainable_params = total_params
        except Exception:
            total_params = 0
            trainable_params = 0

    print("=" * 60)
    print("  RÉSUMÉ DU MODÈLE")
    print("=" * 60)
    print(f"  Paramètres totaux      : {total_params:,}")
    print(f"  Paramètres entraînables : {trainable_params:,}")
    print(f"  Paramètres gelés        : {total_params - trainable_params:,}")
    if input_shape is not None:
        print(f"  Forme d'entrée          : {input_shape}")
    print("=" * 60)

    if detailed and hasattr(model, 'named_parameters'):
        print("\n  Détail par couche :")
        print("-" * 60)
        print(f"  {'Couche':<35} {'Forme':<20} {'Paramètres':<10}")
        print("-" * 60)
        for name, param in model.named_parameters():
            shape_str = str(list(param.shape))
            n_params = param.numel() if hasattr(param, 'numel') else 0
            trainable = "✓" if not hasattr(param, 'requires_grad') or param.requires_grad else "✗"
            print(f"  {name:<35} {shape_str:<20} {n_params:<10} {trainable}")
        print("-" * 60)