import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torchvision import datasets, transforms


def load_iris_data(test_size=0.3, random_state=42):
    data = load_iris()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_wine_data(test_size=0.3, random_state=42):
    data = load_wine()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_mnist_subset(n_classes=2, n_samples=100, test_size=0.3):
    data = load_digits()
    mask = data.target < n_classes
    X, y = data.data[mask], data.target[mask]
    indices = np.random.RandomState(42).choice(len(X), min(n_samples, len(X)), replace=False)
    X, y = X[indices], y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=42)


def normalize_to_01(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def to_tensor(X, y, device='cpu'):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return X_t, y_t


def get_torchvision_dataset(name='hymenoptera', root='./data'):
    if name == 'hymenoptera':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return None
    return None
