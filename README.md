<div align="center">

# 🫁 ThoraxAI
### Classification d'images medicales par Deep Learning

<p><em>Detection de pneumonie sur radiographies thoraciques par CNN, Transfer Learning et explicabilite visuelle</em></p>

![Status](https://img.shields.io/badge/status-operational-success?style=flat)
![Best AUC](https://img.shields.io/badge/AUC--ROC-0.971-blue?style=flat)
![Best Recall](https://img.shields.io/badge/Recall-98.97%25-blue?style=flat)
![Python](https://img.shields.io/badge/python-3.10+-brightgreen?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

<p><em>Built with:</em></p>

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![UV](https://img.shields.io/badge/UV-package%20manager-5C4EE5?style=flat)

**Models:**
![DenseNet121](https://img.shields.io/badge/DenseNet121-Best%20Model-success?style=flat)
![ResNet18](https://img.shields.io/badge/ResNet18-Transfer%20Learning-orange?style=flat)
![EfficientNet](https://img.shields.io/badge/EfficientNet--B0-Transfer%20Learning-orange?style=flat)
![CNN](https://img.shields.io/badge/CNN-Baseline-blue?style=flat)

**Interpretability:**
![Grad-CAM](https://img.shields.io/badge/Grad--CAM-implemented-purple?style=flat)
![Grad-CAM++](https://img.shields.io/badge/Grad--CAM++-implemented-purple?style=flat)
![IG](https://img.shields.io/badge/Integrated%20Gradients-implemented-purple?style=flat)

---

</div>

## 🎯 Objectif

Developper un pipeline complet de Deep Learning capable de distinguer des radiographies thoraciques **normales** de celles presentant une **pneumonie**, avec des performances robustes (validees par K-fold) et une interpretation visuelle des decisions (Grad-CAM, Grad-CAM++, Integrated Gradients).

> ⚕️ Ce projet s'inscrit dans le cadre d'un **outil d'aide a la decision medicale** — il ne vise en aucun cas a remplacer le diagnostic d'un medecin.

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Source : Guangzhou Women and Children's Medical Center. Labels valides par 2 medecins experts + 1 verificateur.

| Set | NORMAL | PNEUMONIA | Total | Ratio P/N |
|-----|--------|-----------|-------|-----------|
| Train | 1,341 | 3,875 | 5,216 | 2.89 |
| Val (original) | 8 | 8 | 16 | 1.00 |
| Test | 234 | 390 | 624 | 1.67 |

**⚠️ Problemes identifies :**
- Val set original inutilisable (16 images) → re-split stratifie 85/15 du train set
- Desequilibre 2.89:1 → gere par WeightedRandomSampler + pos_weight
- 170 patients partages entre train et test (risque de data leakage)
- Population exclusivement pediatrique (un seul centre)

## 🏗️ Architectures

### CNN Baseline (from scratch)
```
Conv(32) + BatchNorm + ReLU + MaxPool   (224 → 112)
Conv(64) + BatchNorm + ReLU + MaxPool   (112 → 56)
Conv(128) + BatchNorm + ReLU + MaxPool  (56  → 28)
Flatten → Dense(128) + Dropout(0.5) → Dense(1)
```

### Transfer Learning
Trois modeles pre-entraines sur ImageNet, avec fine-tuning des dernieres couches et classifieur enrichi `Linear(256) + ReLU + Dropout + Linear(1)` :

| Modele | Pre-entraine sur | Fine-tuning | Reference |
|--------|-----------------|-------------|-----------|
| **ResNet18** | ImageNet (1.4M images) | layer4 | He et al., 2016 |
| **DenseNet121** | ImageNet (1.4M images) | denseblock4 | Huang et al., 2017 / CheXNet |
| **EfficientNet-B0** | ImageNet (1.4M images) | derniers blocs | Tan & Le, 2019 |

## 📈 Resultats

### Comparaison des modeles (test set, seuil = 0.5)

| Metrique | CNN Baseline | ResNet18 | DenseNet121 | EfficientNet | 
|----------|:----------:|:--------:|:-----------:|:------------:|
| **Accuracy** | 82.53% | 85.74% | **88.78%** | 87.98% |
| **Recall** | 98.72% | **99.49%** | 98.97% | 97.69% |
| **Specificite** | 55.56% | 62.82% | **71.79%** | **71.79%** |
| **F1-score** | 87.60% | 89.71% | **91.69%** | 91.04% |
| **AUC-ROC** | 0.9457 | 0.9666 | **0.9710** | 0.9703 |
| **AUC-PR** | 0.9648 | 0.9751 | 0.9792 | **0.9829** |

🏆 Meilleur modele global : **DenseNet121** (AUC-ROC = 0.9710, seulement 4 pneumonies manquees sur 390).

### Matrice de confusion — DenseNet121

|  | Predit NORMAL | Predit PNEUMONIA |
|--|:------------:|:----------------:|
| **Vrai NORMAL** | 168 (TN) | 66 (FP) |
| **Vrai PNEUMONIA** | 4 (FN) | 386 (TP) |

### 🎚️ Seuil optimal

Le seuil par defaut de 0.5 n'est pas optimal. Seuil optimal a **0.89** (F1 passe de 0.9169 → 0.9328).

### 🔄 Validation croisee K-fold (5 folds, CNN baseline)

| Metrique | Moyenne | Ecart-type |
|----------|:-------:|:----------:|
| Accuracy | 97.60% | ± 0.54% |
| F1-score | 98.38% | ± 0.37% |
| Recall | 97.96% | ± 0.88% |
| AUC-ROC | 0.9974 | ± 0.0009 |

Ecarts-types < 1% → le modele est stable et les performances ne dependent pas du split.

## 🔍 Interpretabilite

Trois methodes complementaires implementees from scratch :

| Methode | Principe | Avantage |
|---------|----------|----------|
| **Grad-CAM** | Gradients moyens sur la derniere couche conv | Vue globale des zones influentes |
| **Grad-CAM++** | Derivees d'ordre superieur | Meilleure localisation, lesions multifocales |
| **Integrated Gradients** | Interpolation baseline → input (50 steps) | Attributions au niveau pixel |

L'analyse des erreurs par Grad-CAM revele que les faux negatifs sont lies a une attention trop localisee du modele, et les faux positifs a une activation sur des structures anatomiques ambigues.

> ⚠️ Les heatmaps ne sont pas des preuves cliniques. Elles analysent le comportement du modele, pas les lesions.

## 💻 Application Streamlit — ThoraxAI

Interface de demonstration avec :
- 🔬 Selection parmi les 4 modeles
- 🎚️ Seuil de decision ajustable
- ⚖️ Comparaison multi-modeles avec vote de consensus
- 🔥 Visualisation Grad-CAM (original / heatmap / superposition)
- 📂 Exemples pre-charges du dataset
- 📋 Historique des analyses

```bash
uv run streamlit run app.py
```

## 📁 Structure du projet

```
ThoraxAI/
├── 📂 app/                      # Application Streamlit (modulaire)
│   ├── main.py
│   ├── styles.py
│   ├── components.py
│   ├── inference.py
│   └── data.py
├── 📂 src/                      # Pipeline ML
│   ├── config.py                # Constantes et configuration
│   ├── dataset.py               # Transforms, DataLoaders, split stratifie
│   ├── model.py                 # CNN baseline + ResNet18 + DenseNet121 + EfficientNet
│   ├── train.py                 # EarlyStopping, boucle d'entrainement, K-fold
│   └── eval/                    # Evaluation (modulaire)
│       ├── metrics.py           # Metriques, comparaison, export JSON
│       ├── plots.py             # Visualisations (confusion matrix, ROC, etc.)
│       ├── gradcam.py           # Grad-CAM, Grad-CAM++, Integrated Gradients
│       └── visualize.py         # Visualisations Grad-CAM
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── 📂 outputs/
│   ├── checkpoints/             # Modeles sauvegardes (.pt)
│   ├── figures/                 # Graphiques generes
│   ├── results.json
│   ├── training_summary.json
│   └── eda_summary.json
├── 📂 reports/                  # Rapport technique
├── 📂 samples/                  # Images exemples pour la demo
├── 📂 scripts/                  # Scripts utilitaires
├── 📄 app.py                    # Point d'entree Streamlit
├── 📄 pyproject.toml
└── 📄 README.md
```

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/Exowz/ThoraxAI.git
cd ThoraxAI

# Installer les dependances avec UV
uv sync --extra demo

# Telecharger le dataset Kaggle
uv sync --extra kaggle
uv run kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

## 🧪 Usage

```bash
# 1. Exploration des donnees
uv run jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Entrainement (4 modeles + ablation study + K-fold)
uv run jupyter notebook notebooks/02_training.ipynb

# 3. Evaluation + interpretabilite
uv run jupyter notebook notebooks/03_evaluation.ipynb

# 4. Application Streamlit
uv run streamlit run app.py
```

## 🔒 Reproductibilite

| Composant | Valeur |
|-----------|--------|
| Python | >= 3.10 |
| PyTorch | >= 2.0 |
| Gestionnaire | UV |
| Seed | 42 |
| Split validation | 15% stratifie |
| Device | CUDA / MPS / CPU |

## ⚠️ Limites

- **Data leakage** : 170 patients partages entre train et test
- **Source unique** : population pediatrique, un seul centre (Guangzhou)
- **Specificite limitee** : 66 faux positifs sur le meilleur modele
- **Pas de calibration des probabilites** : les scores ne refletent pas les vraies probabilites

## 🔭 Perspectives

- Split par patient pour eliminer le data leakage
- Validation sur des datasets multi-centres (CheXpert, MIMIC-CXR)
- Ensemble de modeles (voting entre les 4 architectures)
- Augmentations avancees (cutout, mixup)

## 📚 References

- Kaggle, Chest X-Ray Images (Pneumonia)
- He et al., Deep Residual Learning for Image Recognition, CVPR 2016
- Huang et al., Densely Connected Convolutional Networks, CVPR 2017
- Tan & Le, EfficientNet: Rethinking Model Scaling, ICML 2019
- Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks, ICCV 2017
- Chattopadhay et al., Grad-CAM++, WACV 2018
- Sundararajan et al., Axiomatic Attribution for Deep Networks, ICML 2017
- Rajpurkar et al., CheXNet: Radiologist-Level Pneumonia Detection, arXiv 2017

---

<div align="center">

**Projet B3 Deep Learning** — ECE Paris 2026

M.K.E. Kapoor & T.M. Rakotomalala — Pr. F. Derraz

Made with ❤️ and PyTorch

</div>
