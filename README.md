# NEGradient-GenePriority

<p align="center" width="100%">
    <img width="40%" src=".images/NEGradient-GenePriority-logo.png" >
</p>

The repository **NEGradient-GenePriority** (short for "Non-Euclidean Gradient Methods for Matrix Completion in Gene Prioritization") provides code and data to reproduce the results presented in the paper "[Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.](https://pubmed.ncbi.nlm.nih.gov/29949967/)" This study introduces a novel method for gene prioritization by combining Bayesian matrix factorization (BMF) with genomic and phenotypic side information, enabling robust predictions and improved identification of disease-associated genes.

1. **Problem Formulation**:  
   - Input: Matrix $M$ of gene-disease associations ($M_{ij} = 1$ for known associations, $M_{ij} = 0$ otherwise).  
   - Objective: Predict potential associations $\hat{M}$, with $\hat{M}_{ij}$ as floating-point scores.

2. **Data Preparation**:  
   - Positive Class: Use known gene-disease associations ($M_{ij} = 1$).  
   - Negative Class: Sample negatives from $M_{ij} = 0$, avoiding overlap with positives.
   - Unknown Class: Missing values in a sparse encoding. 
   - Split the data into train and test sets:
     - Split the Positive Class data.
     - Split the Negative Class data.
     - Merge the splits.

3. **Model Training**:  
   - Train a matrix completion model on $M$.

4. **Model Testing**:  
   - Use the trained model to complete $M$, yielding $\hat{M}$ with predicted scores.
   - Rank genes for each disease based on $\hat{M}$. 
   - Use only positive associations from the test set. For each disease, high-ranking
   genes in $\hat{M}$ should correspond to true positive associations from the test set. 

1. **Robustness Testing**:  
   - Perform the procedure on six random splits with different matrices.  
   - Aggregate metrics across splits for robustness.

2. **Outcome**:  
     

## Datasets Generation
#### OMIM1 Dataset Construction

>
**Input**: 
- Gene-disease DataFrame: $d$
- Number of splits: $N$
- Sparsity factor $\alpha$ that is the ratio of 0s over 1s.

**Output**:
- Unified sparse matrices: $M_{1_{i}} \quad \forall i \in \set{1, N}$
- Combined splits: $S_{1_{i}} \quad \forall i \in \set{1, N}$

**Execution**:
1. Convert $d$ to sparse matrix $M_{1_{1s}}$.
2. Sample $N$ zero matrices $M_{1_{0s, i}}$ from $M_{1_{1s}}$ using $\alpha$. $\quad \forall i \in \set{1, N}$
3. Combine $M_{1_{1s}}$ and $M_{1_{0s, i}}$ into unified matrices $M_{1_{i}}$. $\quad \forall i \in \set{1, N}$
4. Split $M_{1_{1s}}$ indices randomly into multiple subsets $S_{1_{1s}}$.
5. Split $M_{1_{0s, i}}$ indices randomly into multiple subsets $S_{1_{0s,i}}$. $\quad \forall i \in \set{1, N}$
6. Merge splits from positive ($S_{1_{1s}}$) and zero ($S_{1_{0s,i}}$) samples into $S_{1_{i}}$. $\quad \forall i \in \set{1, N}$
>

#### OMIM2 Dataset Construction
>
**Input**: 
- Gene-disease DataFrame: $d$
- Number of splits: $N$
- Association threshold: $T$
- Sparsity factor $\alpha$ that is the ratio of 0s over 1s.

**Output**:
- Unified sparse matricx: $M_2$
- Combined folds: $S_2$

**Execution**:
1. Filter diseases with fewer than $T$.
2. Convert $d$ to sparse matrix $M_{2_{1s}}$.
3. Sample zeros from $M_{2_{1s}}$ using $\alpha$.
4. Combine $M_{2_{1s}}$ and $M2_{0s}$ into a unified matrix $M_2$.
5. Split $M_{2_{1s}}$ indices randomly into multiple subsets $S_{2_{1s}}$.
6. Split $M_{2_{0s}}$ indices randomly into multiple subsets $S_{2_{0s}}$.
7. Merge splits from positive ($S_{2_{1s}}$) and zero ($S_{2_{0s}}$) samples into $S_2$.
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
