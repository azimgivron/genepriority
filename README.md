# NEGradient-GenePriority

<p align="center" width="100%">
    <img width="40%" src=".images/NEGradient-GenePriority-logo.png" >
</p>

The repository **NEGradient-GenePriority** (short for "Non-Euclidean Gradient Methods for Matrix Completion in Gene Prioritization") provides code and data to reproduce the results presented in the paper "[Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.](https://pubmed.ncbi.nlm.nih.gov/29949967/)" This study introduces a novel method for gene prioritization by combining Bayesian matrix factorization (BMF) with genomic and phenotypic side information, enabling robust predictions and improved identification of disease-associated genes.

## Datasets Generation

### OMIM1 Dataset Construction

>
**Input**: 
- Gene-disease DataFrame: $d$
- Number of splits: $N$
- Sparsity factor $\alpha$ that is the ratio of 0s over 1s.

**Output**:
- Unified sparse matrices: $O1_i \quad \forall i \in \set{1, N}$
- Combined splits: $S1_i \quad \forall i \in \set{1, N}$

**Execution**:
1. Convert $d$ to sparse matrix $O1_{1s}$.
2. Sample $N$ zero matrices $O1_{0s, i}$ from $O1_{1s}$ using $\alpha$. $\forall i \in \set{1, N}$
3. Combine $O1_{1s}$ and $O1_{0s, i}$ into unified matrices $O1_i$. $\forall i \in \set{1, N}$
4. Split $O1_{1s}$ indices randomly into multiple subsets $S1_{1s}$.
5. Split $O1_{0s, i}$ indices randomly into multiple subsets $S1_{0s,i}$. $\forall i \in \set{1, N}$
6. Merge splits from positive ($S1_{1s}$) and zero ($S1_{0s,i}$) samples into $S1_i$. $\forall i \in \set{1, N}$
>

### OMIM2 Dataset Construction
>
**Input**: 
- Gene-disease DataFrame: $d$
- Number of splits: $N$
- Association threshold: $T$
- Sparsity factor $\alpha$ that is the ratio of 0s over 1s.

**Output**:
- Unified sparse matricx: $O2$
- Combined folds: $S2$

**Execution**:
1. Filter diseases with fewer than $T$.
2. Convert $d$ to sparse matrix $O2_{1s}$.
3. Sample zeros from $O2_{1s}$ using $\alpha$.
4. Combine $O2_{1s}$ and $O2_{0s}$ into a unified matrix $O2$.
5. Split $O2_{1s}$ indices randomly into multiple subsets $S2_{1s}$.
6. Split $O2_{0s}$ indices randomly into multiple subsets $S2_{0s}$.
7. Merge splits from positive ($S2_{1s}$) and zero ($S2_{0s}$) samples into $S2$.
>

---

## Installation

To set up the environment and install necessary dependencies, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/azimgivron/NEGradient-GenePriority.git
   cd NEGradient-GenePriority
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Upgrade pip and Install Dependencies**:
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

After installation, you can reproduce the results by running the main script:

```bash
python main.py
```

This script performs gene prioritization using Bayesian matrix factorization, leveraging the genomic and phenotypic side information.

---

## Repository Structure

- **`NEGradient_GenePriority/`**: Contains the main modules and functions for preprocessing, matrix operations, and evaluation.
- **`main.py`**: Main script to run the gene prioritization pipeline.
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`Dockerfile`**: Configuration for containerized deployment of the pipeline.
- **`pyproject.toml`**: Project configuration and dependencies for building and distribution.
- **`.gitignore`**: Specifies files to ignore in the repository.
- **`LICENSE`**: MIT License for the project.

---

## Key Features

1. **Matrix Completion with Bayesian Matrix Factorization (BMF)**:
   - Implements BMF to predict gene-disease associations based on sparse data.

2. **Support for Genomic and Phenotypic Side Information**:
   - Combines genomic and phenotypic information to improve prioritization accuracy.

3. **Flexible Preprocessing and Evaluation**:
   - Includes tools for data preprocessing, creating folds and splits, and computing evaluation metrics like ROC-AUC and BEDROC.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions, suggestions, or issues, please open an issue.
