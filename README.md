<h1 align="center">
# 🛡️ Adversarial Attacks Explained & Implemented 🎯  
</h1>

![GitHub Repo Stars](https://img.shields.io/github/stars/Quinet-Maxence/adversarial-attacks?style=social)  
![GitHub Repo Forks](https://img.shields.io/github/forks/Quinet-Maxence/adversarial-attacks?style=social)  
![GitHub Last Commit](https://img.shields.io/github/last-commit/Quinet-Maxence/adversarial-attacks)  

📌 **Welcome to the most comprehensive collection of Adversarial Attacks!**  
This repository provides **detailed explanations, implementations, and visualizations** of **20 different adversarial attacks** using **Python, TensorFlow, and the Adversarial Robustness Toolbox (ART)**.  

🔥 **Why this repository?**  
The official ART documentation can be **difficult to follow**, and many implementations lack detailed explanations.  
This repository **bridges the gap** by providing:  
✔ **Well-documented Jupyter Notebooks** 📒  
✔ **Step-by-step attack implementations** ⚡  
✔ **Examples on multiple datasets** 📊  
✔ **Pre-trained models for easy testing** 🏆  

---

---
## 📚 Table of Contents  

- [📖 What are Adversarial Attacks?](#-what-are-adversarial-attacks)  
- [🚀 How to Use This Repository?](#-how-to-use-this-repository)  
- [⚡ Implemented Attacks](#-implemented-attacks)  
  - [🎯 White-Box Attacks](#-white-box-attacks-full-model-access)  
  - [🕵️‍♂️ Black-Box Attacks](#-black-box-attacks-no-access-to-model-weights)  
  - [🎭 Adversarial Patch Attacks](#-adversarial-patch-attacks)  
  - [📌 Universal & Feature-Space Attacks](#-universal--feature-space-attacks)  
- [🏆 Datasets Used](#-datasets-used)  
- [🛠 Requirements](#-requirements)  
- [🤝 Contributing](#-contributing)  
- [🛠 How to Contribute?](#-how-to-contribute)  
- [📜 References & Credits](#-references--credits)  


---

## 📖 What are Adversarial Attacks?  

Adversarial Attacks are **techniques that manipulate machine learning models** by applying **carefully crafted perturbations** to input data. These attacks can:  
- **Mislead classifiers** into making incorrect predictions.  
- **Bypass security models** (e.g., fool facial recognition or self-driving car algorithms).  
- **Expose weaknesses** in Deep Learning architectures.  

📌 **Example of an adversarial attack:**  
A classifier originally predicts **"Panda"** with **57.7% confidence**.  
After a **tiny perturbation**, it predicts **"Gibbon"** with **99.3% confidence!**  

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQihyCOg3MeVDoOHNU7ept_fNaikPaJ01yHfQ&s" width="600"/>
</p>

---

## 🚀 How to Use This Repository?  

### 📌 1. Clone the Repository  
```bash
git clone https://github.com/Quinet-Maxence/adversarial-attacks.git
```
### 📌 2. Create conda environment
```bash
conda create ....
```
### 📌 3. Install Dependencies
```bash
cd adversarial-attacks
```
### 📌 4. Open Jupyter notebook
```bash
jupyter notebook
```

Note: A step-by-step tutorial is available in the ```environment_configuration.txt``` file !

---

## ⚡ Implemented Attacks  

This repository covers **both White-Box and Black-Box** adversarial attacks, structured as **one Jupyter Notebook per attack**.  

### 🎯 White-Box Attacks (Full Model Access)  
| Attack Name | Description | Notebook |
|------------|-------------|-----------|
| **FGSM** (Fast Gradient Sign Method) | Classic one-step gradient-based attack. | [`FGSM_AdversarialAttack.ipynb`](FGSM_AdversarialAttack.ipynb) |
| **PGD** (Projected Gradient Descent) | Iterative version of FGSM with stronger perturbations. | [`PGD_AdversarialAttack.ipynb`](PGD_AdversarialAttack.ipynb) |
| **DeepFool** | Minimal perturbation attack based on decision boundaries. | [`DeepFool_AdversarialAttack.ipynb`](DeepFool_AdversarialAttack.ipynb) |
| **Carlini & Wagner (C&W)** | One of the strongest optimization-based attacks. | [`CW_AdversarialAttack.ipynb`](CW_AdversarialAttack.ipynb) |
| **Elastic Net Attack (EAD)** | A variation of C&W using Elastic Net regularization. | [`EAD_AdversarialAttack.ipynb`](EAD_AdversarialAttack.ipynb) |
| **NewtonFool** | Attack that reduces confidence in the true class. | [`NewtonFool_AdversarialAttack.ipynb`](NewtonFool_AdversarialAttack.ipynb) |
| **Jacobian-Based Saliency Map Attack (JSMA)** | Perturbation focused on important pixels. | [`JSMA_AdversarialAttack.ipynb`](JSMA_AdversarialAttack.ipynb) |

### 🕵️‍♂️ Black-Box Attacks (No Access to Model Weights)  
| Attack Name | Description | Notebook |
|------------|-------------|-----------|
| **Boundary Attack** | Queries the model iteratively to fool it. | [`Boundary_AdversarialAttack.ipynb`](Boundary_AdversarialAttack.ipynb) |
| **HopSkipJump Attack** | A more efficient version of Boundary Attack. | [`HopSkipJump_AdversarialAttack.ipynb`](HopSkipJump_AdversarialAttack.ipynb) |
| **ZOO (Zero Order Optimization)** | Optimization-based black-box attack. | [`ZOO_AdversarialAttack.ipynb`](ZOO_AdversarialAttack.ipynb) |
| **Square Attack** | Query-efficient adversarial attack. | [`Square_AdversarialAttack.ipynb`](Square_AdversarialAttack.ipynb) |

### 🎭 Adversarial Patch Attacks  
| Attack Name | Description | Notebook |
|------------|-------------|-----------|
| **Adversarial Patch** | Fooling models with a patch visible in the image. | [`AdversarialPatch_AdversarialAttack.ipynb`](AdversarialPatch_AdversarialAttack.ipynb) |
| **DPatch** | A variation of Adversarial Patch for object detection. | [`DPatch_AdversarialAttack.ipynb`](DPatch_AdversarialAttack.ipynb) |
| **Robust DPatch** | An improved version of DPatch. | [`RobustDPatch_AdversarialAttack.ipynb`](RobustDPatch_AdversarialAttack.ipynb) |

### 📌 Universal & Feature-Space Attacks  
| Attack Name | Description | Notebook |
|------------|-------------|-----------|
| **Targeted Universal Perturbations** | A single perturbation fools many images. | [`UniversalPerturbation_AdversarialAttack.ipynb`](UniversalPerturbation_AdversarialAttack.ipynb) |
| **Feature Adversaries** | Attacks focused on high-level feature representations. | [`FeatureAdversaries_AdversarialAttack.ipynb`](FeatureAdversaries_AdversarialAttack.ipynb) |
| **Virtual Adversarial Method (VAT)** | Regularization-based adversarial method. | [`VAT_AdversarialAttack.ipynb`](VAT_AdversarialAttack.ipynb) |

---

## 🏆 Datasets Used  

Each attack is tested on various datasets:  

| Dataset | Description | Number of Classes | Image Shape |
|---------|-------------|------------------|-------------|
| **MNIST** | Handwritten digit recognition | 10 | (28, 28, 1) |
| **CIFAR-10** | Small object classification | 10 | (32, 32, 3) |
| **CIFAR-100** | 100-class object classification | 100 | (32, 32, 3) |
| **ImageNet (Subset)** | Large-scale image classification | 1,000 | (299, 299, 3) |

📌 **Note:** You can choose your dataset inside each notebook! 📂  

---

## 🛠 Requirements  

To run the notebooks, ensure you have **Python 3.8+** and install the required libraries using:  

```bash
pip install -r requirements.txt
```

or install them manually:

```bash
pip install tensorflow torch torchvision adversarial-robustness-toolbox numpy matplotlib tqdm scipy
```

### 📌 Additional dependencies:

* Jupyter Notebook → pip install notebook
* tqdm (for progress bars) → pip install tqdm
* scipy (for rotations & transformations) → pip install scipy

✅ Ensure you have CUDA installed for GPU acceleration:
```bash
!nvidia-smi
```
If no GPU is detected, consider updating TensorFlow and PyTorch.

---

## 🤝 Contributing  

🔥 This repository is **open for contributions**! If you:  
✔ Found an **error** in an implementation? **Create an issue!**  
✔ Have a **new attack** to add? **Submit a pull request!**  
✔ Want to **discuss adversarial robustness**? **Join the discussions!**  

---

## 🛠 How to Contribute?  

We welcome all contributions! Follow these steps to add your improvements:  

### 📌 1. Fork the repository  
Click on the **Fork** button at the top right corner of this page.  

### 📌 2. Clone your fork  
```bash
git clone https://github.com/Quinet-Maxence/adversarial-attacks.git
cd adversarial-attacks
```

### 📌 3. Create a new branch
```bash
git checkout -b my-new-feature
```  

### 📌 4. Make your changes
* Add a new adversarial attack 🛡️
* Improve documentation 📝
* Fix bugs 🐞

### 📌 5. Commit your changes
```bash
git add .
git commit -m "Added XYZ adversarial attack"
```

### 📌 6. Push to your fork
```bash
git push origin my-new-feature
```

### 📌 7. Submit a pull request 🚀
Go to **Pull Requests** on this repository and submit your PR for review!
✅ Once approved, your contributions will be added to the project!

---

### ✅ **📌 Section "References & Credits"**  

## 📜 References & Credits  

This repository is based on **ART (Adversarial Robustness Toolbox)** and research papers, including:  

📖 **Ian Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)**  
🔗 [https://arxiv.org/abs/1412.6572](https://arxiv.org/abs/1412.6572)  

📖 **Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)**  
🔗 [https://arxiv.org/abs/1608.04644](https://arxiv.org/abs/1608.04644)  

📖 **Su et al., "One Pixel Attack for Fooling Deep Neural Networks" (2019)**  
🔗 [https://arxiv.org/abs/1710.08864](https://arxiv.org/abs/1710.08864)  

📌 **Libraries Used:**  
- **Adversarial Robustness Toolbox (ART)** → [https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)  
- **TensorFlow** → [https://www.tensorflow.org/](https://www.tensorflow.org/)  
- **PyTorch** → [https://pytorch.org/](https://pytorch.org/)  

