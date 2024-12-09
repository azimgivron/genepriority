# NEGradient_GenePriority

[![codecov](https://codecov.io/gh/azimgivron/NEGradient_GenePriority/branch/main/graph/badge.svg?token=QSAKYRC4EH)](https://codecov.io/gh/azimgivron/NEGradient_GenePriority)

<p align="center" width="100%">
    <img width="40%" src=".images/NEGradient_GenePriority-logo.png" >
</p>

The repository **NEGradient_GenePriority** (short for "Non-Euclidean Gradient Methods for Matrix Completion in Gene Prioritization") provides the code and data to reproduce the results presented in the paper "[Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.](https://pubmed.ncbi.nlm.nih.gov/29949967/)" This study introduces a novel method for gene prioritization by combining Bayesian matrix factorization (BMF) with genomic and phenotypic side information, enabling robust predictions and improved identification of disease-associated genes.

---

## Overview

1. **Problem Formulation**:  
   - **Input**: Matrix $M$ of gene-disease associations ($M_{ij} = 1$ for known associations, $M_{ij} = 0$ otherwise).  
   - **Objective**: Predict potential associations $\hat{M}$, where $\hat{M}_{ij}$ are floating-point scores.

2. **Data Preparation**:  
   - **Positive Class**: Known gene-disease associations ($M_{ij} = 1$).  
   - **Negative Class**: Randomly sampled negatives from $M_{ij} = 0$, avoiding overlap with positives.
   - **Unknown Class**: Represented as missing values in the sparse encoding.  
   - **Procedure**:  
     - Split the Positive Class data.  
     - Split the Negative Class data.  
     - Merge the splits.

3. **Model Training**:  
   - Train a matrix completion model on $M$.

4. **Model Testing**:  
   - Complete $M$ using the trained model to yield $\hat{M}$ with predicted scores.  
   - Rank genes for each disease based on $\hat{M}$.  
   - Use only positive associations from the test set. For each disease, highly ranked genes in $\hat{M}$ should correspond to true positive associations from the test set.

**Note**:  
- The procedure is repeated across six random splits using different matrices. Metrics are aggregated across splits for robustness.

---

## Dataset Generation

### OMIM1 Dataset Construction

**Input**:  
- Gene-disease DataFrame: $d$  
- Number of splits: $N$  
- Sparsity factor $\alpha$, the ratio of 0s to 1s.  

**Output**:  
- Unified sparse matrices: $M_{1_{i}} \quad \forall i \in \{1, N\}$  
- Combined splits: $S_{1_{i}} \quad \forall i \in \{1, N\}$  

**Procedure**:  
1. Convert $d$ into a sparse matrix $M_{1_{1s}}$.  
2. Sample $N$ zero matrices $M_{1_{0s, i}}$ from $M_{1_{1s}}$ using $\alpha \quad \forall i \in \{1, N\}$.  
3. Combine $M_{1_{1s}}$ and $M_{1_{0s, i}}$ into unified matrices $M_{1_{i}} \quad \forall i \in \{1, N\}$.  
4. Split $M_{1_{1s}}$ indices randomly into subsets $S_{1_{1s}}$.  
5. Split $M_{1_{0s, i}}$ indices randomly into subsets $S_{1_{0s, i}} \quad \forall i \in \{1, N\}$.  
6. Merge splits from positive ($S_{1_{1s}}$) and zero ($S_{1_{0s, i}}$) samples into $S_{1_{i}} \quad \forall i \in \{1, N\}$.  

---

### OMIM2 Dataset Construction

**Input**:  
- Gene-disease DataFrame: $d$  
- Number of splits: $N$  
- Association threshold: $T$  
- Sparsity factor $\alpha$, the ratio of 0s to 1s.  

**Output**:  
- Unified sparse matrix: $M_2$  
- Combined folds: $S_2$  

**Procedure**:  
1. Filter diseases with fewer than $T$ associations.  
2. Convert $d$ into a sparse matrix $M_{2_{1s}}$.  
3. Sample zeros from $M_{2_{1s}}$ using $\alpha$.  
4. Combine $M_{2_{1s}}$ and $M_{2_{0s}}$ into a unified matrix $M_2$.  
5. Split $M_{2_{1s}}$ indices randomly into subsets $S_{2_{1s}}$.  
6. Split $M_{2_{0s}}$ indices randomly into subsets $S_{2_{0s}}$.  
7. Merge splits from positive ($S_{2_{1s}}$) and zero ($S_{2_{0s}}$) samples into $S_2$.  

---

## Installation

**Steps**:

1. Clone the repository:
   ```bash
   git clone https://github.com/azimgivron/NEGradient_GenePriority.git
   cd NEGradient_GenePriority
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Upgrade pip and install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Requirements

- Python 3.11  
- pip (Python package installer)  
- Virtual environment (venv) module  

---

## Usage

Run the main script to reproduce the results:

```bash
python main.py
```

This script performs gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.

---

## Repository Structure

- **`NEGradient_GenePriority/`**: Main modules for preprocessing, matrix operations, and evaluation.
- **`main.py`**: Main script for running the gene prioritization pipeline.
- **`requirements.txt`**: List of dependencies for the project.
- **`Dockerfile`**: Configuration for containerized deployment.
- **`pyproject.toml`**: Project and dependency configuration for building and distribution.
- **`.gitignore`**: Specifies files to ignore in the repository.
- **`LICENSE`**: Project license.

---

## Key Features

1. **Matrix Completion with Bayesian Matrix Factorization (BMF)**:  
   - Predicts gene-disease associations based on sparse data.

2. **Support for Genomic and Phenotypic Side Information**:  
   - Combines genomic and phenotypic data to improve prioritization accuracy.

3. **Flexible Preprocessing and Evaluation**:  
   - Tools for data preprocessing, creating folds/splits, and computing metrics like ROC-AUC and BEDROC.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or issues, please open an issue on the repository.