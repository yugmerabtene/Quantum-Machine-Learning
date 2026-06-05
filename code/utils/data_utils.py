import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional, List
import os


def load_iris_data(test_size: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge le dataset Iris avec normalisation StandardScaler.

    Args:
        test_size: Proportion du jeu de test.
        random_state: Graine aléatoire pour la reproductibilité.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    data = load_iris()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_wine_data(test_size: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge le dataset Wine avec normalisation StandardScaler.

    Args:
        test_size: Proportion du jeu de test.
        random_state: Graine aléatoire pour la reproductibilité.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    data = load_wine()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_mnist_subset(n_classes: int = 2, n_samples: int = 100, test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge un sous-ensemble du dataset Digits (MNIST small).

    Args:
        n_classes: Nombre de classes à sélectionner (0 à n_classes-1).
        n_samples: Nombre maximum d'échantillons.
        test_size: Proportion du jeu de test.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    data = load_digits()
    mask = data.target < n_classes
    X, y = data.data[mask], data.target[mask]
    indices = np.random.RandomState(42).choice(len(X), min(n_samples, len(X)), replace=False)
    X, y = X[indices], y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=42)


def load_hymenoptera(
    root: str = './data',
    batch_size: int = 32,
    image_size: int = 224,
    n_classes: int = 2,
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, dict]:
    """Charge le dataset Hymenoptera (fourmis/abeilles) pour le transfer learning.

    Télécharge les images si nécessaire et applique les transformations
    standards d'ImageNet (normalisation, redimensionnement).

    Args:
        root: Répertoire racine pour le dataset.
        batch_size: Taille des batchs.
        image_size: Taille des images redimensionnées.
        n_classes: Nombre de classes (défaut 2 : fourmis/abeilles).
        subset_size: Limiter le nombre d'images par split (None = tout).

    Returns:
        Tuple (train_loader, val_loader, class_names).
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    hymenoptera_dir = os.path.join(root, 'hymenoptera_data')
    if os.path.exists(hymenoptera_dir):
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(hymenoptera_dir, x), data_transforms[x])
            for x in ['train', 'val']
        }
    else:
        import urllib.request
        import zipfile
        url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
        zip_path = os.path.join(root, 'hymenoptera_data.zip')
        os.makedirs(root, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(hymenoptera_dir, x), data_transforms[x])
            for x in ['train', 'val']
        }

    if subset_size is not None:
        for split in ['train', 'val']:
            n = min(subset_size, len(image_datasets[split]))
            indices = np.random.RandomState(42).choice(len(image_datasets[split]), n, replace=False)
            image_datasets[split] = Subset(image_datasets[split], indices)

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'val']
    }

    class_names = image_datasets['train'].dataset.classes if hasattr(image_datasets['train'], 'dataset') else image_datasets['train'].classes

    return dataloaders['train'], dataloaders['val'], class_names


def load_quantum_dot(
    n_samples: int = 500,
    n_features: int = 4,
    n_classes: int = 3,
    noise: float = 0.1,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Génère un dataset synthétique de boîtes quantiques pour la classification.

    Simule des mesures de conductance et d'énergie de boîtes quantiques
    avec des centres de cluster correspondant aux niveaux d'énergie caractéristiques.

    Args:
        n_samples: Nombre total d'échantillons.
        n_features: Nombre de caractéristiques (par défaut 4 : conductance, énergie, spin, température).
        n_classes: Nombre de classes (niveaux d'énergie).
        noise: Niveau de bruit gaussien.
        test_size: Proportion du jeu de test.
        random_state: Graine aléatoire.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    rng = np.random.RandomState(random_state)
    samples_per_class = n_samples // n_classes

    centers = rng.randn(n_classes, n_features) * 3.0
    centers[:, 0] *= 5.0
    centers[:, 1] = np.linspace(0.5, 3.0, n_classes) + rng.randn(n_classes) * 0.2

    X_list, y_list = [], []
    for cls in range(n_classes):
        X_cls = rng.randn(samples_per_class, n_features) * noise + centers[cls]
        X_list.append(X_cls)
        y_list.append(np.full(samples_per_class, cls))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    X[:, 0] = np.abs(X[:, 0]) * 7.75
    X[:, 1] = np.abs(X[:, 1]) * 1.6e-19
    X[:, 2] = np.clip(X[:, 2], -0.5, 0.5)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_synthetic_data(
    dataset_type: str = 'binary',
    n_samples: int = 400,
    n_features: int = 4,
    n_classes: int = 2,
    n_informative: Optional[int] = None,
    noise: float = 0.1,
    test_size: float = 0.3,
    random_state: int = 42,
    normalize: bool = True,
    scale_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Génère des datasets synthétiques variés pour les tests quantiques.

    Args:
        dataset_type: Type de dataset parmi 'binary', 'multiclass', 'circles', 'moons', 'blobs'.
        n_samples: Nombre d'échantillons.
        n_features: Nombre de caractéristiques.
        n_classes: Nombre de classes.
        n_informative: Nombre de features informatives (défaut n_features).
        noise: Niveau de bruit.
        test_size: Proportion du jeu de test.
        random_state: Graine aléatoire.
        normalize: Appliquer StandardScaler.
        scale_range: Si fourni (min, max), normaliser dans cette plage après StandardScaler.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    from sklearn.datasets import make_classification, make_circles, make_moons, make_blobs

    if n_informative is None:
        n_informative = n_features

    if dataset_type == 'binary':
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=n_informative, n_classes=2, random_state=random_state
        )
    elif dataset_type == 'multiclass':
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=n_informative, n_classes=n_classes, random_state=random_state
        )
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        if n_features > 2:
            rng = np.random.RandomState(random_state)
            extra = rng.randn(n_samples, n_features - 2) * noise
            X = np.hstack([X, extra])
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        if n_features > 2:
            rng = np.random.RandomState(random_state)
            extra = rng.randn(n_samples, n_features - 2) * noise
            X = np.hstack([X, extra])
    elif dataset_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=n_classes, random_state=random_state)
    else:
        raise ValueError(f"Type '{dataset_type}' non supporté. Choisir parmi "
                         "'binary', 'multiclass', 'circles', 'moons', 'blobs'.")

    if normalize:
        X = StandardScaler().fit_transform(X)
    if scale_range is not None:
        X = MinMaxScaler(feature_range=scale_range).fit_transform(X)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def normalize_to_01(X: np.ndarray) -> np.ndarray:
    """Normalise les données dans [0, 1].

    Args:
        X: Matrice de forme (n_samples, n_features).

    Returns:
        Matrice normalisée dans [0, 1].
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def normalize_to_angle_range(X: np.ndarray) -> np.ndarray:
    """Normalise les données dans [0, π] pour l'encodage angulaire quantique.

    Args:
        X: Matrice de forme (n_samples, n_features).

    Returns:
        Matrice normalisée dans [0, π].
    """
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    return scaler.fit_transform(X)


def reduce_dimensionality(
    X: np.ndarray,
    n_components: int = 4,
    method: str = 'pca'
) -> Tuple[np.ndarray, object]:
    """Réduit la dimensionnalité des données avec PCA.

    Args:
        X: Données de forme (n_samples, n_features).
        n_components: Nombre de dimensions cibles.
        method: Méthode de réduction ('pca').

    Returns:
        Tuple (X_reduit, transformateur).
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_.sum()
        print(f"PCA : {n_components} composantes expliquent {explained_var:.2%} de la variance")
        return X_reduced, reducer
    else:
        raise ValueError(f"Méthode '{method}' non supportée. Choisir 'pca'.")


def to_tensor(X: np.ndarray, y: np.ndarray, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Convertit des arrays numpy en tensors PyTorch.

    Args:
        X: Features.
        y: Étiquettes.
        device: Appareil cible ('cpu' ou 'cuda').

    Returns:
        Tuple (X_tensor, y_tensor).
    """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return X_t, y_t


def get_torchvision_dataset(name: str = 'hymenoptera', root: str = './data') -> Optional[object]:
    """Charge un dataset torchvision avec transformations standards.

    Args:
        name: Nom du dataset ('hymenoptera').
        root: Répertoire racine.

    Returns:
        Dataset torchvision ou None.
    """
    if name == 'hymenoptera':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return None
    return None