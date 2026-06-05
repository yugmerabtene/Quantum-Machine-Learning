# Séance 1.2 — Théorie de l'apprentissage automatique

## Apprentissage supervisé, non-supervisé, par renforcement

La théorie de l'apprentissage automatique fournit le cadre formel dans lequel s'inscrivent les modèles classiques et quantiques. On distingue trois paradigmes principaux :

**Apprentissage supervisé** : on dispose d'un ensemble d'entraînement $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^m$ où $x_i \in \mathcal{X}$ sont les données et $y_i \in \mathcal{Y}$ les étiquettes. L'objectif est d'apprendre une fonction $h : \mathcal{X} \to \mathcal{Y}$ minimisant l'erreur de prédiction sur des données non vues.

**Apprentissage non-supervisé** : seules les données $\{x_i\}$ sont disponibles. On cherche à découvrir une structure sous-jacente — clustering, réduction de dimension, estimation de densité.

**Apprentissage par renforcement** : un agent interagit avec un environnement, recevant une récompense $r_t$ à chaque étape $t$, et apprend une politique $\pi(a|s)$ maximisant la récompense cumulée $\mathbb{E}[\sum_t \gamma^t r_t]$.

## Fonction de perte, risque empirique et risque structurel

Soit $\mathcal{H}$ un espace de hypothèses (modèles). On définit la **fonction de perte** $\ell : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$ mesurant l'écart entre la prédiction $\hat{y} = h(x)$ et la vraie valeur $y$. Les choix courants incluent :

- Perte quadratique : $\ell(y, \hat{y}) = (y - \hat{y})^2$
- Perte 0-1 : $\ell(y, \hat{y}) = \mathbb{I}[y \neq \hat{y}]$
- Perte hinge (SVM) : $\ell(y, \hat{y}) = \max(0, 1 - y\hat{y})$
- Cross-entropie : $\ell(y, \hat{y}) = -\sum_k y_k \log \hat{y}_k$

Le **risque réel** (ou risque attendu) d'une hypothèse $h$ est :

$$
R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(y, h(x))]
$$

Mais comme la distribution $\mathcal{D}$ est inconnue, on minimise le **risque empirique** sur l'échantillon d'entraînement :

$$
\hat{R}(h) = \frac{1}{m} \sum_{i=1}^m \ell(y_i, h(x_i))
$$

Le principe de **minimisation du risque empirique** (ERM) consiste à sélectionner $\hat{h} = \arg\min_{h \in \mathcal{H}} \hat{R}(h)$. Cependant, ERM seul peut conduire au sur-apprentissage. La **minimisation du risque structurel** (SRM) ajoute un terme de régularisation qui pénalise la complexité du modèle :

$$
h^* = \arg\min_{h \in \mathcal{H}} \left[ \hat{R}(h) + \lambda \, \Omega(h) \right]
$$

où $\Omega(h)$ est une mesure de complexité (norme L1, L2, etc.) et $\lambda > 0$ un hyperparamètre.

## Généralisation et sur-apprentissage

Le **sur-apprentissage** (overfitting) se produit lorsque $\hat{R}(h)$ est faible mais $R(h)$ est élevée : le modèle mémorise le bruit des données plutôt que la structure sous-jacente. L'**erreur de généralisation** est définie comme :

$$
\epsilon_{\text{gen}} = |R(h) - \hat{R}(h)|
$$

Un résultat fondamental de la théorie de l'apprentissage est que, avec probabilité $1 - \delta$ :

$$
R(h) \leq \hat{R}(h) + O\left(\sqrt{\frac{\text{Complexité}(\mathcal{H})}{m}} \right)
$$

Cette borne dépend de la complexité de $\mathcal{H}$ et de la taille de l'échantillon $m$.

## Théorème de No-Free-Lunch

Le théorème de No-Free-Lunch (NFL) stipule qu'aucun algorithme d'apprentissage n'est universellement meilleur qu'un autre sur toutes les distributions de données possibles. Formellement, pour deux algorithmes $A_1$ et $A_2$, il existe des distributions où $A_1$ surpasse $A_2$ et vice versa, à performance moyenne égale sur l'ensemble des problèmes. Ce résultat est crucial pour le QML : il implique que l'avantage quantique, s'il existe, est nécessairement circonscrit à certaines classes de problèmes et ne peut être universel [MRT18].

## Compromis biais-variance

L'erreur de généralisation se décompose en trois termes :

$$
\mathbb{E}[(\hat{h}(x) - y)^2 | x] = \text{Biais}(\hat{h}(x))^2 + \text{Variance}(\hat{h}(x)) + \sigma^2
$$

où :
- $\text{Biais}(h)^2 = (\mathbb{E}[\hat{h}(x)] - h^*(x))^2$ mesure l'erreur due aux approximations du modèle
- $\text{Variance}(h) = \mathbb{E}[(\hat{h}(x) - \mathbb{E}[\hat{h}(x)])^2]$ mesure la sensibilité aux fluctuations des données
- $\sigma^2$ est le bruit irréductible

Un modèle trop simple a un biais élevé (sous-apprentissage), tandis qu'un modèle trop complexe a une variance élevée (sur-apprentissage). L'objectif est de trouver le point d'équilibre minimisant l'erreur totale.

## Complexité de Rademacher

La complexité de Rademacher est une mesure de la capacité d'une classe $\mathcal{H}$ à s'adapter au bruit aléatoire. Soit $\{\sigma_i\}_{i=1}^m$ des variables de Rademacher indépendantes ($\mathbb{P}(\sigma_i = +1) = \mathbb{P}(\sigma_i = -1) = 1/2$). La complexité de Rademacher empirique est :

$$
\hat{\mathfrak{R}}_m(\mathcal{H}) = \mathbb{E}_\sigma \left[ \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i h(x_i) \right]
$$

Une borne de généralisation utilisant Rademacher est :

$$
R(h) \leq \hat{R}(h) + \hat{\mathfrak{R}}_m(\mathcal{H}) + O\left(\sqrt{\frac{\log(1/\delta)}{m}}\right)
$$

```python
import numpy as np

def rademacher_complexity(H_preds, n_samples):
    """
    Calcule la complexité de Rademacher empirique.
    
    H_preds : array (n_models, n_samples) — prédictions de chaque modèle
    """
    n_models, m = H_preds.shape
    sigma = np.random.choice([-1, 1], size=m)
    correlations = H_preds @ sigma / m
    return np.max(correlations)

# Exemple : classe des fonctions linéaires bornées
m, d = 100, 5
X = np.random.randn(m, d)
w = np.random.randn(1000, d)  # 1000 hypothèses
H_preds = X @ w.T  # (100, 1000)
R_hat = rademacher_complexity(H_preds.T, m)
print(f"Complexité de Rademacher estimée : {R_hat:.4f}")
```

## VC-dimension

La VC-dimension (Vapnik-Chervonenkis) est une mesure de complexité combinatoire. C'est la taille du plus grand ensemble de points que $\mathcal{H}$ peut *pulvériser* (shatter), c'est-à-dire pour lequel toutes les $2^m$ étiquetages possibles sont réalisables par un élément de $\mathcal{H}$.

Pour les classifieurs linéaires dans $\mathbb{R}^d$, la VC-dimension est $d+1$. Pour les réseaux de neurones, elle peut être beaucoup plus grande. Un résultat fondamental relie VC-dimension et généralisation : avec probabilité $1-\delta$,

$$
R(h) \leq \hat{R}(h) + O\left(\sqrt{\frac{\text{VCdim}(\mathcal{H})}{m}} \right)
$$

Dans le contexte QML, la VC-dimension des circuits variationnels est un sujet actif de recherche. On sait que certains circuits peuvent avoir une VC-dimension exponentiellement grande en $n$, suggérant une capacité d'expression potentiellement très élevée — mais aussi un risque accru de sur-apprentissage et de barren plateaux [SP21].

## Pertinence pour le QML

La théorie de l'apprentissage fournit les outils pour analyser rigoureusement les modèles quantiques. Les questions clés incluent :

- Quelle est la capacité d'expression des circuits quantiques paramétrés ?
- Comment la complexité de Rademacher d'un VQC évolue-t-elle avec $n$ et la profondeur ?
- Les bornes de généralisation des quantum kernels sont-elles plus serrées que leurs équivalents classiques ?
- Le sur-apprentissage est-il plus ou moins probable dans l'espace de Hilbert exponentiel ?

Ces questions seront abordées dans les séances 5-6 (barren plateaux) et 7-8 (kernels quantiques), où nous verrons que la grande expressivité des modèles quantiques est une épée à double tranchant.

## Références

- [MRT18] Mohri, M., Rostamizadeh, A. & Talwalkar, A. *Foundations of Machine Learning.* 2nd ed., MIT Press, 2018. Ch. 2.
- [Bis06] Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 1.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
- [GBC16] Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning.* MIT Press, 2016. Ch. 5-7.
