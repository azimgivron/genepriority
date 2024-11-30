# NEGradient-GenePriority

The repository **NEGradient-GenePriority** (short for "Non-Euclidean Gradient Methods for Matrix Completion in Gene Prioritization") provides code and data to reproduce the results presented in the paper "Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information." This study introduces a novel method for gene prioritization by combining Bayesian matrix factorization (BMF) with genomic and phenotypic side information, enabling robust predictions and improved identification of disease-associated genes.

---

## Installation

To set up the environment and install necessary dependencies, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/azimgivron/Non-Euclidean-Gradient-Methods-for-Matrix-Completion-in-Gene-Prioritization.git
   cd Non-Euclidean-Gradient-Methods-for-Matrix-Completion-in-Gene-Prioritization
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Upgrade pip and Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Requirements

- Python 3.11 or higher
- pip (Python package installer)
- Virtual environment (venv) module

---

## Usage

After installation, you can reproduce the results by running the main script:

```bash
python main.py
```

This script performs gene prioritization using Bayesian matrix factorization, leveraging the genomic and phenotypic data provided in the repository.

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
