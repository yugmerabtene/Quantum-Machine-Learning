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

### Démonstration formelle

Considérons un ensemble de fonctions cibles $f : \mathcal{X} \to \mathcal{Y}$ et un ensemble d'algorithmes d'apprentissage $\mathcal{A}$. Notons $R(A, f)$ le risque de l'algorithme $A$ sur la cible $f$. La version forte du théorème NFL [WM97] établit que, pour tout ensemble fini $\mathcal{X}$ avec $|\mathcal{X}| = d$ et $\mathcal{Y} = \{-1, +1\}$, la performance moyenne sur toutes les fonctions cibles possibles est identique pour tout algorithme :

$$
\sum_{f : \mathcal{X} \to \mathcal{Y}} R(A, f) = \text{constante}, \quad \forall A \in \mathcal{A}
$$

Plus précisément, soient $m$ exemples d'entraînement distincts et $\mathcal{C} \subseteq \mathcal{X}$ l'ensemble des points non observés ($|\mathcal{C}| = d - m$). Pour tout algorithme déterministe $A$ et toute fonction de perte $\ell$,

$$
\frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} R(A, f \mid \mathcal{C}) = \frac{1}{2^{|\mathcal{C}|}} \sum_{f \in \mathcal{F}} \frac{1}{|\mathcal{C}|} \sum_{x \in \mathcal{C}} \ell(f(x), A(x)) = \text{constante}
$$

où $\mathcal{F}$ est l'ensemble de toutes les fonctions $\mathcal{X} \to \mathcal{Y}$ (il y en a $2^d$). Cette constante ne dépend pas de l'algorithme $A$.

**Conséquence pour le QML** : Si un algorithme quantique $A_Q$ surpasse un algorithme classique $A_C$ sur une classe de problèmes $\mathcal{F}_0$, nécessairement $A_C$ surpasse $A_Q$ sur $\mathcal{F} \setminus \mathcal{F}_0$. L'avantage quantique est donc toujours relatif à une classe de problèmes. L'enjeu est d'identifier les $\mathcal{F}_0$ où le QML excelle — par exemple, les données avec structure quantique, ou les problèmes où l'espace de feature quantique est plus adapté.

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

### Estimation complète de la complexité de Rademacher

La complexité de Rademacher doit être estimée sur plusieurs tirages des variables $\sigma_i$ pour obtenir une estimation stable. L'implémentation suivante calcul la borne de généralisation complète :

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC

def rademacher_complexity_mc(H_preds, n_trials=500):
    """
    Estimation Monte Carlo de la complexité de Rademacher.
    
    H_preds : array (n_models, n_samples)
    n_trials : nombre de tirages aléatoires de sigma
    
    Retourne la moyenne sur n_trials de sup_h (1/m) sum(sigma_i h(x_i)).
    """
    n_models, m = H_preds.shape
    total = 0.0
    for _ in range(n_trials):
        sigma = np.random.choice([-1, 1], size=m)
        correlations = H_preds @ sigma / m
        total += np.max(correlations)
    return total / n_trials

def generalization_bound(R_hat, m, delta=0.05, loss_upper_bound=1.0):
    """
    Borne de généralisation avec complexité de Rademacher.
    R(h) <= R_hat(h) + 2 * R_hat(H) + loss_bound * sqrt(log(1/delta) / (2m))
    """
    rademacher_term = 2 * R_hat
    confidence_term = loss_upper_bound * np.sqrt(np.log(1 / delta) / (2 * m))
    return rademacher_term + confidence_term

# --- Démonstration sur un jeu de données synthétique ---
X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                           random_state=42)
y_transformed = 2 * y - 1  # Conversion en {-1, +1}

# Générer les prédictions de 500 classifieurs linéaires bornés sur [-1, 1]
n_hypotheses = 500
predictions = np.zeros((n_hypotheses, len(y_transformed)))
for i in range(n_hypotheses):
    w = np.random.randn(10)
    w /= np.linalg.norm(w)  # Normalisation
    raw = X @ w
    predictions[i] = np.clip(raw, -1, 1)

# Calcul de la complexité de Rademacher
R_hat = rademacher_complexity_mc(predictions, n_trials=500)
bound = generalization_bound(R_hat, len(y_transformed), delta=0.05)

print(f"Complexité de Rademacher estimée : {R_hat:.4f}")
print(f"Borne de généralisation (delta=0.05) : {bound:.4f}")
print(f"Avec m={len(y_transformed)} échantillons, l'excès de risque est borné par {bound:.4f}")

# --- Comparaison : effet de la taille de l'échantillon ---
for m_val in [50, 100, 200, 500, 1000]:
    X_sub, y_sub = make_classification(n_samples=m_val, n_features=10, 
                                        n_informative=5, random_state=42)
    preds = np.zeros((n_hypotheses, m_val))
    for i in range(n_hypotheses):
        w = np.random.randn(10)
        w /= np.linalg.norm(w)
        raw = X_sub @ w
        preds[i] = np.clip(raw, -1, 1)
    R = rademacher_complexity_mc(preds, n_trials=300)
    b = generalization_bound(R, m_val, delta=0.05)
    print(f"m={m_val:4d} | Rademacher={R:.4f} | Borne={b:.4f}")
```

On observe que la borne de généralisation diminue en $O(1/\sqrt{m})$ et que la complexité de Rademacher capture la capacité de la classe d'hypothèses.

## VC-dimension

La VC-dimension (Vapnik-Chervonenkis) est une mesure de complexité combinatoire. C'est la taille du plus grand ensemble de points que $\mathcal{H}$ peut *pulvériser* (shatter), c'est-à-dire pour lequel toutes les $2^m$ étiquetages possibles sont réalisables par un élément de $\mathcal{H}$.

Pour les classifieurs linéaires dans $\mathbb{R}^d$, la VC-dimension est $d+1$. Pour les réseaux de neurones, elle peut être beaucoup plus grande. Un résultat fondamental relie VC-dimension et généralisation : avec probabilité $1-\delta$,

$$
R(h) \leq \hat{R}(h) + O\left(\sqrt{\frac{\text{VCdim}(\mathcal{H})}{m}} \right)
$$

Dans le contexte QML, la VC-dimension des circuits variationnels est un sujet actif de recherche. On sait que certains circuits peuvent avoir une VC-dimension exponentiellement grande en $n$, suggérant une capacité d'expression potentiellement très élevée — mais aussi un risque accru de sur-apprentissage et de barren plateaux [SP21].

### Calcul de la VC-dimension : exemples

**Perceptron dans $\mathbb{R}^d$** : On montre que la VC-dimension est $d+1$. Pour $d$ points en position générale, le perceptron peut réaliser les $2^d$ étiquetages possibles (par le théorème de séparabilité). Mais pour $d+1$ points, il existe des étiquetages non réalisables (ex : XOR dans $\mathbb{R}^2$). Donc $\text{VCdim}(\text{perceptron}) = d+1$.

**Preuve détaillée (cas $d=2$)** : Considérons 3 points non colinéaires $x_1, x_2, x_3 \in \mathbb{R}^2$. Pour tout étiquetage $(y_1, y_2, y_3) \in \{-1,+1\}^3$, il existe un hyperplan séparant les $+1$ des $-1$ (car les points sont en position générale). Donc $\text{VCdim} \geq 3$. Pour 4 points, l'étiquetage XOR-esque $\{(0,0,+1), (1,0,-1), (0,1,-1), (1,1,+1)\}$ n'est pas réalisable par un classifieur linéaire. Donc $\text{VCdim} = 3 = d+1$.

**Séparateur linéaire avec marge** : Si la donnée est bornée dans une boule de rayon $R$ et que la marge est $\gamma$, alors la VC-dimension effective est bornée par :

$$
\text{VCdim}_{\text{marge}} \leq \min\left(d+1, \left\lceil \frac{R^2}{\gamma^2} \right\rceil \right)
$$

C'est le fondement théorique du SVM à marge : maximiser la marge $\gamma$ réduit la VC-dimension effective, améliorant la généralisation.

**Réseaux de neurones** : Pour un réseau à $L$ couches avec $W$ poids et $U$ unités, la borne est :

$$
\text{VCdim} \leq O(W \cdot U)
$$

Ce qui signifie que la VC-dimension croît avec le nombre de paramètres, justifiant empiriquement que les grands réseaux nécessitent plus de données d'entraînement.

**Circuits quantiques variationnels** : Pour un VQC avec $n$ qubits et $p$ paramètres, des bornes récentes donnent :

$$
\text{VCdim} \leq O(p \cdot 2^n)
$$

Cette borne exponentielle en $n$ a des implications ambivalentes : haute expressivité mais risque de sur-apprentissage, et les bornes de généralisation deviennent lâches.

## Pertinence pour le QML

La théorie de l'apprentissage fournit les outils pour analyser rigoureusement les modèles quantiques. Les questions clés incluent :

- Quelle est la capacité d'expression des circuits quantiques paramétrés ?
- Comment la complexité de Rademacher d'un VQC évolue-t-elle avec $n$ et la profondeur ?
- Les bornes de généralisation des quantum kernels sont-elles plus serrées que leurs équivalents classiques ?
- Le sur-apprentissage est-il plus ou moins probable dans l'espace de Hilbert exponentiel ?

Ces questions seront abordées dans les séances 5-6 (barren plateaux) et 7-8 (kernels quantiques), où nous verrons que la grande expressivité des modèles quantiques est une épée à double tranchant.

## Tableau récapitulatif : modèles ML et leur complexité

| Modèle | VC-dim (borne) | Rademacher $\hat{\mathfrak{R}}_m$ | Nombre de paramètres | Risque de sur-apprentissage | Lien QML |
|---|---|---|---|---|---|
| Perceptron ($\mathbb{R}^d$) | $d+1$ | $O\left(\frac{B}{\sqrt{m}}\right)$ | $d+1$ | Faible | Ansatz à 1 couche |
| SVM (marge $\gamma$) | $\min\left(d+1, \lceil R^2/\gamma^2 \rceil\right)$ | $O\left(\frac{R}{\gamma\sqrt{m}}\right)$ | $m$ (vecteurs de support) | Faible (marge large) | QSVM |
| Réseau de neurones (1 couche cachée, $U$ unités) | $O(WU)$ | $O\left(\frac{W}{\sqrt{m}}\right)$ | $W$ (poids) | Modéré–élevé | VQC peu profond |
| Réseau profond ($L$ couches) | $O(WL)$ | $O\left(\frac{WL}{\sqrt{m}}\right)$ | $W$ | Élevé | VQC profond |
| KRR (noyau $k$, régularisation $\lambda$) | $\infty$ (RBF) | $O\left(\kappa\sqrt{\frac{\text{Tr}(K)}{m^2}}\right)$ | $m$ (coefficients $\alpha$) | Faible ($\lambda$ grand) | Quantum KRR |
| Ensemble (Random Forest, $T$ arbres) | $O(T \cdot \text{VC-dim(arbre)})$ | $O\left(\frac{T}{\sqrt{m}}\right)$ | $O(T \cdot 2^d)$ | Modéré | AdaBoost quantique |
| VQC ($n$ qubits, $p$ paramètres) | $O(p \cdot 2^n)$ (borne sup.) | $\Theta\left(\frac{2^n}{m}\right)$ (concentration) | $p$ | Potentiellement élevé | Direct |

**Interprétation** : Les modèles quantiques variationnels (VQC) ont une VC-dimension et une complexité de Rademacher qui croissent exponentiellement avec $n$. Cela signifie à la fois une grande expressivité (capacité d'approximer des fonctions complexes) et un risque de sur-apprentissage si $m \ll 2^n$. La régularisation explicite (pénalisation $\|\theta\|^2$) et implicite (architecture, barren plateaux) sont les mécanismes principaux pour contrôler la généralisation.

## Références

- [MRT18] Mohri, M., Rostamizadeh, A. & Talwalkar, A. *Foundations of Machine Learning.* 2nd ed., MIT Press, 2018. Ch. 2.
- [Bis06] Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 1.
- [SP21] Schuld, M. & Petruccione, F. *Machine Learning with Quantum Computers.* 2nd ed., Springer, 2021.
- [GBC16] Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning.* MIT Press, 2016. Ch. 5-7.
- [WM97] Wolpert, D. H. & Macready, W. G. « No free lunch theorems for optimization. » *IEEE Transactions on Evolutionary Computation* 1(1), 67–82 (1997).
- [Vap98] Vapnik, V. N. *Statistical Learning Theory.* Wiley, 1998.
- [Mc18] McClean, J. R. et al. « Barren plateaus in quantum neural network training landscapes. » *Nature Communications* 9, 4812 (2018).
