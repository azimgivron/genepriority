The repository `NEGradient-GenePriority`, short for `Non-Euclidean Gradient Methods for Matrix Completion in Gene Prioritization`, provides code and data to reproduce the results presented in the paper "Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information." This study introduces a novel method for gene prioritization by combining Bayesian matrix factorization (BMF) with additional genomic and phenotypic side information, enabling robust predictions and improved identification of disease-associated genes.

**Installation**

To install the necessary dependencies, run the following command from the root directory:

```bash
pip install .
```

**Requirements**

- Python 3.11 or higher
- pip
- requirements.txt

**Usage**

After installation, you can reproduce the results by executing the `main.py` script:

```bash
python main.py
```

This will perform gene prioritization using the Bayesian matrix factorization method with the provided genomic and phenotypic data.

**Repository Structure**

- `matrix_completion/`: Contains modules related to matrix completion algorithms.
- `main.py`: Main script to run the gene prioritization analysis.
- `Dockerfile`: Docker configuration for containerized deployment.
- `pyproject.toml`: Project configuration and dependencies.
- `.gitignore`: Specifies files to ignore in the repository.
- `LICENSE`: MIT License for the project.

**License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

**Contact**

For questions or issues, please open an issue in this repository.

This enhanced README provides a clearer understanding of the repository's purpose, installation process, usage, structure, and licensing, facilitating easier engagement for users and contributors. 