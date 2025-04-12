# genepriority 🚀

[![codecov](https://codecov.io/gh/azimgivron/genepriority/branch/main/graph/badge.svg?token=QSAKYRC4EH)](https://codecov.io/gh/azimgivron/genepriority)

<p align="center">
  <img width="40%" src=".images/genepriority-logo.png" alt="genepriority logo">
</p>

Hey there! Welcome to **genepriority** – your go-to repo for rocking matrix completion algorithms on the *Online Mendelian Inheritance in Man* (OMIM) dataset. Our mission? To hunt down disease-associated genes and level up accuracy by throwing in extra genomic and phenotypic info. Let’s dive in! 😎

---

## Algorithms in Action

We’re using two awesome methods:

1. **Non-Euclidean Gradient Algorithm (NEGA2)**  
   Check out the paper *"Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications"* for all the math magic.
2. **GeneHound**  
   Based on *"Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information"* ([Zakeri et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29949967/)).

### Cool References
- Ghaderi, S., Moreau, Y., & Ahookhosh, M. (2022). *Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications*. JMLR, 23(2022):1-44.
- Zakeri, P., Simm, J., Arany, A., ElShal, S., & Moreau, Y. (2018). *Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.* Bioinformatics, 34(13), i447–i456. doi:10.1093/bioinformatics/bty289.

---

## Overview 🔍

### What’s the Deal?

We tackle gene prioritization as a **matrix completion** challenge. In simple terms, we fill in the missing pieces in a giant gene-disease puzzle using side information. Neat, right?

### Objective Function 🎯

Our goal is to optimize:

$$
\min_{W, H} \quad \frac{1}{2}\bigl\|B \odot (R - W H^T)\bigr\|^2_2 + \lambda_1 \|W\|_F^2 + \lambda_2 \|H\|_F^2
$$

where:  
- **W** and **H** are the learnable matrices.  
- **B** is a binary mask.  
- **R** is our gene-disease association matrix.

This problem is nonconvex (yup, it's a tough nut to crack) due to the quartic term from $W H^T$, and the gradient might not be Lipschitz continuous (so regular step sizes can be tricky).

### NEGA (Non-Euclidean Gradient Algorithm) 😎

To overcome these issues, NEGA uses **relative smoothness**. That means we compare our function to a distance-generating kernel that vibes with the problem’s geometry.

#### Bregman Distance & Relative Smoothness

We say a function $f$ is $L_f$-smooth relative to a kernel $h$ if:

$$
\left| f(x) - f(y) - \langle \nabla f(y), x-y \rangle \right| \leq L_f \mathcal{D}_h(x,y)
$$

with the Bregman distance defined as:

$$
\mathcal{D}_h(x,y) = h(x) - h(y) - \langle \nabla h(y), x-y \rangle
$$

#### How It Works 🤖

The NEGA update rule is:

$$
x^{k+1} = \arg\min_x  \{  \langle \nabla f(x^k), x - x^k \rangle + \frac{1}{\alpha_k} \mathcal{D}_h(x,x^k)  \}
$$

This formulation keeps the update step convex and ensures we gradually converge to a local optimum—even when the gradient isn’t exactly well-behaved.

---

## NEGA with Side Info

We even have a cool twist where we toss in additional genomic and phenotypic data to boost prioritization performance. Next-level gene hunting! 🔬

---

## Experimental Setup 🧪

### Datasets & Info Sources

- **Gene-Disease Association Matrix**:  
  Extracted from OMIM 2013 and post-processed as per the GeneHound paper.  
  - **Genes:** 14,196  
  - **Diseases:** 315  
  - **Known Positives:** 2,625 (backed by experimental evidence)  
  - **Association Density:** 0.06%

- **Side Info Sources**:
  
  | **Dataset** | **Number of Positive Entries** |
  |-------------|--------------------------------|
  | Phen Text   | 66,883                         |
  | UniProt     | 164,125                        |
  | InterPro    | 51,499                         |
  | GO          | 1,365,393                      |

*Note:* Before merging, each genomic data matrix is normalized by its Frobenius norm so no single source dominates.

### Evaluation Strategy 📊

We split the data into:
- **90%** for on cross-validation
- **10%** for hyperparameter fine tuning  

Then use five-fold cross-validation on the main split (72% training, 18% testing per fold). We monitor metrics like **RMSE**, **ROC/AUROC**, **BEDROC**, and **PR Curve/AUPRC**. Also, negative sampling (5 negatives per positive) is used to mirror the real-life imbalance in biology.

---

## Installation 🚀

### Option 1: Quick Install with pip

```bash
pip install genepriority git+https://github.com/azimgivron/genepriority.git@main
```

**Requirements:** [SMURFF](https://smurff.readthedocs.io/en/release-0.17/INSTALL.html)

### Option 2: Docker for ARM64 🐳

1. **Build the Docker Image:**
   ```bash
   docker compose build
   ```
2. **Start Up the Containers:**
   ```bash
   docker compose up -d
   ```
3. **Access the Environment:**
   ```bash
   docker exec -it $(docker ps -q -f "name=nega") zsh
   ```
4. **Launch TensorBoard:**
   Visit [http://localhost:6006](http://localhost:6006)
5. **Stop the Containers:**
   ```bash
   docker compose down
   ```

*Note:* Docker is preconfigured to handle permission issues, and TensorBoard logs live at `/home/TheGreatestCoder/code/logs` inside the container.

---

## Requirements

- **Python 3.11**  
- **pip**  
- **venv** module (for virtual environments)

---

## Repository Structure 🗂️

### What's Inside?

- **`genepriority/`**: Main modules for preprocessing, matrix operations, and evaluation.
- **`requirements.txt`**: Lists all project dependencies.
- **`Dockerfile`**: For containerized deployment.
- **`pyproject.toml`**: Build and dependency info.
- **`.gitignore`**: Files and directories to ignore.
- **`LICENSE`**: MIT License details.

### Folder Breakdown

```bash
genepriority/
├── models/                # NEGA2 algorithm implementation
├── evaluation/            # Tools for evaluation metrics
├── postprocessing/        # Aggregates & analyzes model results
├── preprocessing/         # DataLoader for gene-disease data
├── scripts/
│   ├── genehound          # Reproducing GeneHound results
│   ├── nega               # NEGA fine-tuning & cross-validation scripts
│   └── post               # Post-processing scripts
├── trainer/               # Training and evaluation framework
├── utils/                 # Utility functions
├── README.md              # This documentation!
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Build & project metadata
```

---

## How to Rock the Scripts 🎸

### Command Line Usage

```bash
usage: genepriority [-h] {genehound,nega-tuning,nega,post} ...
```

**Subcommands:**
- **genehound:** Run GeneHound with cross-validation on the OMIM dataset.
- **nega-tuning:** Find the best hyperparameters for NEGA.
- **nega:** Train and evaluate the NEGA model using cross-validation.
- **post:** Post-process and analyze the evaluation results.

For more details, run:
```bash
genepriority --help
```

---

## License

This project is MIT licensed. See the [LICENSE](LICENSE) file for details.

---

## Got Questions? 🤔

If you have any questions, suggestions, or issues, feel free to open an issue on GitHub. We love hearing from you!

---

Enjoy the ride and happy gene prioritizing! 🎉  
