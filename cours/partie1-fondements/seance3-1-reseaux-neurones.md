# Séance 3.1 — Réseaux de neurones classiques

## Perceptron et MLP

Le **perceptron** (Rosenblatt, 1958) est le modèle fondateur des réseaux de neurones. Il calcule une combinaison linéaire des entrées suivie d'une non-linéarité :

$$
y = f\left( \sum_{i=1}^d w_i x_i + b \right)
$$

où $f$ est une fonction d'activation, $w_i$ les poids, et $b$ le biais.

Le **Multi-Layer Perceptron** (MLP) généralise le perceptron par l'empilement de couches :

$$
h^{(l)} = \sigma^{(l)}\left( W^{(l)} h^{(l-1)} + b^{(l)} \right)
$$

avec $h^{(0)} = x$ et $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$.

### Fonctions d'activation

Les fonctions d'activation introduisent la non-linéarité indispensable à l'expressivité :

- Sigmoid : $\sigma(x) = 1 / (1 + e^{-x})$
- Tanh : $\tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})$
- ReLU : $\text{ReLU}(x) = \max(0, x)$
- Softmax : $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(784, 256, 10)
x = torch.randn(32, 784)
out = model(x)
print(f"Sortie : {out.shape}")
```

## Rétropropagation et optimisation

La **rétropropagation** (backpropagation) calcule le gradient de la fonction de perte par rapport à tous les paramètres via la règle de dérivation en chaîne. Pour une perte $\mathcal{L}$ et un paramètre $w^{(l)}_{ij}$ :

$$
\frac{\partial \mathcal{L}}{\partial w^{(l)}_{ij}} = \delta^{(l)}_j \, h^{(l-1)}_i
$$

où $\delta^{(l)}_j$ est l'erreur rétropropagée à la couche $l$.

### Stochastic Gradient Descent (SGD)

SGD met à jour les paramètres sur un mini-batch :

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_B(\theta_t)
$$

où $\eta$ est le taux d'apprentissage et $B$ un mini-batch.

### Adam (Adaptive Moment Estimation)

Adam combine momentum adaptatif et taux d'apprentissage par paramètre :

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    x_batch = torch.randn(64, 784)
    y_batch = torch.randint(0, 10, (64,))

    optimizer.zero_grad()
    out = model(x_batch)
    loss = criterion(out, y_batch)
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Réseaux convolutifs (CNN)

Les CNN exploitent la structure spatiale des données via l'opération de **convolution** :

$$
(f * g)(i, j) = \sum_{m=-k}^{k} \sum_{n=-k}^{k} f(m, n) \, g(i - m, j - n)
$$

En pratique, un noyau de convolution $W \in \mathbb{R}^{k \times k \times c_{\text{in}} \times c_{\text{out}}}$ est appliqué avec un stride $s$ et un padding $p$.

```python
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 7 * 7, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### Pooling

Le **pooling** réduit la dimension spatiale :

- Max pooling : $p_{ij} = \max_{m,n \in \text{fenêtre}} a_{i+m, j+n}$
- Average pooling : $p_{ij} = \frac{1}{k^2} \sum_{m,n} a_{i+m, j+n}$

### Architectures modernes : ResNet

ResNet (He et al., 2015) introduit les **connexions résiduelles** :

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

où $\mathcal{F}$ est une ou plusieurs couches convolutives. Cette identité de chemin court (*skip connection*) permet d'entraîner des réseaux très profonds (50, 101, 152 couches) en atténuant le problème de gradient vanishing.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)
```

## Transformers

Le **Transformer** (Vaswani et al., 2017) est devenu l'architecture dominante en NLP et vision, grâce au mécanisme d'**attention**.

### Attention et self-attention

L'attention calcule une combinaison pondérée des valeurs $V$ en fonction de la compatibilité entre requêtes $Q$ et clés $K$ :

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

En **self-attention**, $Q$, $K$, $V$ sont toutes issues de la même séquence d'entrée.

### Mécanisme QKV

Chaque token $x_i$ est projeté en trois vecteurs :

$$
q_i = W_Q x_i, \quad k_i = W_K x_i, \quad v_i = W_V x_i
$$

Le score d'attention entre $x_i$ et $x_j$ est :

$$
\alpha_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{t} \exp(q_i^T k_t / \sqrt{d_k})}
$$

La sortie pour le token $i$ est :

$$
\text{output}_i = \sum_j \alpha_{ij} v_j
$$

```python
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return attn @ V

attn = SelfAttention(64, 16)
x = torch.randn(10, 64)  # 10 tokens
out = attn(x)
print(f"Sortie attention : {out.shape}")
```

### Multi-head attention

L'attention multi-têtes exécute $h$ attentions en parallèle :

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

où $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$.

## Pertinence pour le QML

Les réseaux de neurones classiques sont la contrepartie naturelle des circuits quantiques variationnels. Les analogies sont nombreuses :

- Les couches d'un MLP correspondent aux couches variationnelles d'un VQC
- La rétropropagation classique a son analogue quantique dans la *parameter-shift rule*
- Les CNN classiques inspirent les QCNN (séance 9.2)
- Les Transformers quantiques (séance 10.1) intègrent des VQC dans l'attention multi-têtes

Ces connexions seront explorées en détail dans les séances 9 à 11.

## Références

- [GBC16] Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning.* MIT Press, 2016. Ch. 6, 9.
- [Bis06] Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 5.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021. §7.3.
- [Tra26] « Quantum Transformers for Image Classification. » *Quantum Machine Intelligence* 8, 43 (2026).
