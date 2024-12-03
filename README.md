# NEGradient-GenePriority

The repository **NEGradient-GenePriority** (short for "Non-Euclidean Gradient Methods for Matrix Completion in Gene Prioritization") provides code and data to reproduce the results presented in the paper "[Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.](https://pubmed.ncbi.nlm.nih.gov/29949967/)" This study introduces a novel method for gene prioritization by combining Bayesian matrix factorization (BMF) with genomic and phenotypic side information, enabling robust predictions and improved identification of disease-associated genes.

## Datasets Generation

### OMIM1 Dataset Construction
```
Input: Gene-disease DataFrame
Output: Unified sparse matrix and combined splits

1. Convert gene-disease DataFrame to sparse matrix `omim1_1s`.
2. Sample zeros from `omim1_1s` using sparsity factor.
3. Combine `omim1_1s` and `omim1_0s` into a unified matrix.
4. Split `omim1_1s` indices randomly into multiple subsets.
5. Split `omim1_0s` indices randomly into multiple subsets.
6. Merge splits from positive (`omim1_1s`) and zero (`omim1_0s`) samples.
```

### OMIM2 Dataset Construction
```
Input: Gene-disease DataFrame
Output: Unified sparse matrix and combined folds

1. Filter diseases with fewer than the specified association threshold.
2. Convert gene-disease DataFrame to sparse matrix `omim2_1s`.
3. Sample zeros from `omim2_1s` using sparsity factor.
4. Combine `omim2_1s` and `omim2_0s` into a unified matrix.
5. Split `omim2_1s` indices randomly into multiple subsets.
6. Split `omim2_0s` indices randomly into multiple subsets.
7. Merge splits from positive (`omim2_1s`) and zero (`omim2_0s`) samples.
```

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
