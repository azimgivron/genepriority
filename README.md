# genepriority

[![codecov](https://codecov.io/gh/azimgivron/genepriority/branch/main/graph/badge.svg?token=QSAKYRC4EH)](https://codecov.io/gh/azimgivron/genepriority)

<p align="center" width="100%">
    <img width="40%" src=".images/genepriority-logo.png" >
</p>

The repository **genepriority** is designed to implement and evaluate matrix completion algorithms on the *Online Mendelian Inheritance in Man* (OMIM) dataset for the task of gene prioritization. The goal is to identify disease-associated genes. Additional genomic and phenotypic side information can optionally be leveraged to increase accuracy. 

Two algorithms are used/implemented:
1. Non-Euclidean Gradient Algorithm, NEGA, more specifically **NEGA2 algorithm**, as described in the paper *"Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications"*.
2. Genehound from *"Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information"* ([Zakeri et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29949967/)). 

**References**

1. Ghaderi, S., Moreau, Y., & Ahookhosh, M. (2022). *Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications*. Journal of Machine Learning Research, 23(2022):1-44.
2. Zakeri, P., Simm, J., Arany, A., ElShal, S., & Moreau, Y. (2018). *Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.* Bioinformatics, 34(13), i447–i456. doi:10.1093/bioinformatics/bty289. PMID: 29949967; PMCID: PMC6022676.

## Overview

[![](https://mermaid.ink/img/pako:eNptlE1P4zAQhv_KyOeC2nxCDiuVlu1WaqhEOW2ag2mG1KJxItvZBVr-O44duoZsDpE9z8wkr2fGR7KrCyQJKQVt9rC633LQzzRbp8s0h4uLH6cNrZoDwlie4MaYJ7l1ku2jjbrHBqmC6MXab7owmGUPgjIOm-bAVO6SefaAUn0ByAu7sO-Z8Vtks7pqWoVwn25uYd4KxkswWfWij5xbV7u5zVY1LWDDCoQlf6pFRRWrOVBewF23O7A3hL9M7eGnqB-Rs1YaYJUuuUShBuGnz_S_spnQQlGHMImQ6pM75O73F2aztMqxsB4yd6WdD80ya-yepYlNJ5kB8HnIDvJ65A2R3yN_iIIeBUMU9igcoqhH0f8LdFbRVeiAqtNKlWA7dBSlE5NrrTW5bWOQ1yPPIFdR6vfIN8hVlAY9CgxyFaVhj0KDXEVp1KPIoC-KvqmZlqXAsqvww17UbamrhJTDukFhOuFf0nUvbmVywvSP9iixyF0Pq3HlmvyhKRiawqEpck3firEyzTuVsq10UzIpuzGZU0WBCju3d9kCeadBjxLlzxJ0b8Mt3e1hrhuZ6l7uJuQ8bdqlS5FiV1GZkxGpUM8CK_Q9cey-uSVqjxVuSaKXBRXPW7Ll79qPtqrevPIdSZRocUTMIZLkiR6k3rVNoX9hzqg-7epsbSj_XdfVZwgWTNUitbeSuZyMC0mO5IUkF17gX0aRF0-uwtAL4nHsjcgrSWL_0pvEkziIAz_249B_H5E3k9W79Pxr7R9ej4Pgyg8i7_0D2xRmHg?type=png)](https://mermaid.live/edit#pako:eNptlE1P4zAQhv_KyOeC2nxCDiuVlu1WaqhEOW2ag2mG1KJxItvZBVr-O44duoZsDpE9z8wkr2fGR7KrCyQJKQVt9rC633LQzzRbp8s0h4uLH6cNrZoDwlie4MaYJ7l1ku2jjbrHBqmC6MXab7owmGUPgjIOm-bAVO6SefaAUn0ByAu7sO-Z8Vtks7pqWoVwn25uYd4KxkswWfWij5xbV7u5zVY1LWDDCoQlf6pFRRWrOVBewF23O7A3hL9M7eGnqB-Rs1YaYJUuuUShBuGnz_S_spnQQlGHMImQ6pM75O73F2aztMqxsB4yd6WdD80ya-yepYlNJ5kB8HnIDvJ65A2R3yN_iIIeBUMU9igcoqhH0f8LdFbRVeiAqtNKlWA7dBSlE5NrrTW5bWOQ1yPPIFdR6vfIN8hVlAY9CgxyFaVhj0KDXEVp1KPIoC-KvqmZlqXAsqvww17UbamrhJTDukFhOuFf0nUvbmVywvSP9iixyF0Pq3HlmvyhKRiawqEpck3firEyzTuVsq10UzIpuzGZU0WBCju3d9kCeadBjxLlzxJ0b8Mt3e1hrhuZ6l7uJuQ8bdqlS5FiV1GZkxGpUM8CK_Q9cey-uSVqjxVuSaKXBRXPW7Ll79qPtqrevPIdSZRocUTMIZLkiR6k3rVNoX9hzqg-7epsbSj_XdfVZwgWTNUitbeSuZyMC0mO5IUkF17gX0aRF0-uwtAL4nHsjcgrSWL_0pvEkziIAz_249B_H5E3k9W79Pxr7R9ej4Pgyg8i7_0D2xRmHg)

1. **Problem Formulation**:  
   - **Input**: Matrix $O$ of gene-disease associations ($O_{ij} = 1$ for known associations, $O_{ij} = 0$ for unknown or missing associations).  
   - **Objective**: Predict potential associations $\hat{O}$, where $\hat{O}_{ij}$ are floating-point scores indicating the likelihood of gene-disease associations.

2. **Data Preparation**:  
   - **Positive Class**: Known gene-disease associations ($O_{ij} = 1$).  
   - **Negative Class**: Randomly sampled negatives ($O_{ij} = 0$), ensuring no overlap with positives.  
   - **Unknown Class**: Represented as missing values in the sparse encoding.  
   - **Procedure**:  
     - **Generate Negative Samples**: Randomly sample negatives and merge with the positive class to form $O_1$.  
     - **Create Train-Test Splits**: Split $O_1$ into train and test sets across six random iterations for cross-validation.

3. **Incorporating Side Information**:  
   - Load side information (e.g., gene or disease features) and normalize using the Frobenius norm.  
   - Integrate side information into the training process to enhance predictions.

4. **Model Training**:  
   - Train matrix completion models on the train split of $O_1$, using side information.  
   - Compute the Root Mean Squared Error (RMSE) during training as a measure of model performance.  
   - Repeat training across six random splits to yield six independent models, $M_i \quad \forall i \in \set{1,2,3,4,5,6}$.

5. **Matrix Completion**:  
   - Use the trained models to generate the completed matrix, generating six predicted matrices ($\hat{O_i} \quad \forall i \in \set{1,2,3,4,5,6}$) with floating-point scores.  
   - Aggregate the six matrices using a mean operation to produce $\bar{O}$ (averaged matrix).

6. **Model Testing and Ranking**:  
   - Flatten the matrix and rank genes based on the predicted scores.  
   - Assume missing values are zeros for ranking purposes.  
   - Use only positive associations from the test set for evaluation. Highly ranked genes in $\bar{O}$ should correspond to true positive associations.

7. **Evaluation and Metrics**:  
   - Compute ranking metrics to assess the quality of predictions, focusing on the ability of the model to rank true positive associations higher.  
   - Metrics are aggregated across the six splits to ensure robustness.

---

## Dataset Generation

The above procedure describes the general approach. However, the reference paper uses two datasets, OMIM1 and OMIM2, respectively named  $O_1$ and $O_2$. The procedure for $O_2$ differs in the following way:

- Subset Relationship: $O_2$ is a subset of $O_1$.
- Model Generation: For $O_2$, models are generated using an N-Fold cross-validation approach, rather than the N random splits used for $O_1$.  

### OMIM1 Dataset Construction

**Input**:  
- Gene-disease DataFrame: $d$  
- Number of splits: $N$  
- Sparsity factor $\alpha$, the ratio of 0s to 1s.  

**Output**:  
- Sparse matrix: $O_{1}$  
- Splits: $S_{1_{i}} \quad \forall i \in \{1, N\}$  

**Procedure**:  
1. Convert $d$ into a sparse matrix $O_{1_{1s}}$.  
2. Sample $N$ zeros to get $O_{1}$ from $O_{1_{1s}}$ using $\alpha$.  
3. Split $O_{1}$ indices randomly into subsets $S_{1_{i}} \quad \forall i \in \{1, N\}$.  

---

### OMIM2 Dataset Construction

**Input**:  
- Gene-disease DataFrame: $d$  
- Number of folds: $N$  
- Association threshold: $T$  
- Sparsity factor $\alpha$, the ratio of 0s to 1s.  

**Output**:  
- Sparse matrix: $O_2$
- Folds: $F_{2_{i}} \quad \forall i \in \{1, N\}$

**Procedure**:  
1. Filter diseases with fewer than $T$ associations.  
2. Convert $d$ into a sparse matrix $O_{2_{1s}}$.  
5. Sample $N$ zeros to get $O_{2}$ from $O_{2_{1s}}$ using $\alpha$.  
6. Split $O_{2}$ indices randomly into $N$ folds $F_{2_{i}} \quad \forall i \in \{1, N\}$.  

---

## Installation

You can set up the environment using one of the following methods:

### **Option 1: Using `pip`**

```bash
pip install genepriority git+https://github.com/azimgivron/genepriority.git@main
```

**Requirements**: SMURFF[https://smurff.readthedocs.io/en/release-0.17/INSTALL.html]

4. **Run Scripts**:
   Use the `genepriority` scripts as described in the [Scripts Usage](#scripts-usage) section.

### **Option 2: Using Docker for ARM64 Platform**

Using Docker, two containers are launched: one for the working environment (`nega`) and another for TensorBoard.

1. **Build the Docker Image**:
   Build the image and set up the containers using the provided `docker-compose.yml`:
   ```bash
   docker compose build
   ```

2. **Start the Containers**:
   Launch both containers (working environment and TensorBoard):
   ```bash
   docker compose up -d
   ```

3. **Access the Working Environment**:
   Enter the `nega` container interactively:
   ```bash
   docker exec -it $(docker ps -q -f "name=nega") zsh
   ```
   This opens a `zsh` shell in the `nega` container, ready for running scripts and experiments.

4. **Access TensorBoard**:
   View TensorBoard logs and visualizations by navigating to:
   [http://localhost:6006](http://localhost:6006)

5. **Stopping the Containers**:
  To stop the running containers, use:
  ```bash
  docker compose down
  ```

### **Notes**:
- The Docker image and `docker-compose.yml` are preconfigured to handle permissions issues when working with mounted volumes.
- The TensorBoard logs are located at `/home/TheGreatestCoder/code/logs` in the container and are accessible to both services.

---

## Requirements

- Python 3.11  
- pip (Python package installer)  
- Virtual environment (venv) module  

---

## Repository Structure

### Root Directory

- **`genepriority/`**: Main modules for preprocessing, matrix operations, and evaluation.
- **`requirements.txt`**: List of dependencies for the project.
- **`Dockerfile`**: Configuration for containerized deployment.
- **`pyproject.toml`**: Project and dependency configuration for building and distribution.
- **`.gitignore`**: Specifies files to ignore in the repository.
- **`LICENSE`**: Project license.

### Folder Structure

```bash
genepriority/
├── compute_models/
│   └── # NEGA2 algorithm implementation
├── evaluation/
│   └── # Defines `Evaluation` class for managing evaluation metrics
├── postprocessing/
│   └── # `ModelEvaluationCollection` class for aggregating and analyzing results
├── preprocessing/
│   └── # `DataLoader` class for preprocessing gene-disease association data
├── scripts/
│   ├── genehound
│   │   └── # Script for reproducing GeneHound results
│   ├── nega
│   │   └── # Script for NEGA2 cross-validation and evaluation
│   ├── post
│       └── # Script for post processing
├── trainer/
│   └── # Facilitates training and evaluation of predictive models
├── utils/
│   └── # Utility functions for supporting operations across the repository
├── README.md
│   └── # Documentation for the repository
├── requirements.txt
│   └── # Python dependencies for the project
├── pyproject.toml
│   └── # Build system and project metadata
```

---

Here’s how you can add a **Scripts Usage** section to your README file, providing detailed guidance on how to use the `genepriority` script:

---

### Scripts Usage
#### `genepriority`
**Usage**:
```bash
usage: genepriority [-h] {genehound-omim1,genehound-omim2,nega-cv,nega,post} ...

Gene Prioritization Tool

This tool implements matrix completion algorithms for gene prioritization on the OMIM gene-disease matrix. It supports integration of side information, such as gene features and disease phenotypes, to improve prediction accuracy.

Implemented subcommands:
  1. GeneHound (based on MACAU)
  2. Non-Euclidean Gradient Algorithm (NEGA)
  3. Post-processing of evaluation results

positional arguments:
  {genehound-omim1,genehound-omim2,nega-cv,nega,post}
    genehound-omim1     Run GeneHound with the OMIM1 dataset (multiple splits).
    genehound-omim2     Run GeneHound with the filtered OMIM2 dataset (cross-validation).
    nega-cv             Perform cross-validation for hyperparameter tuning of NEGA.
    nega                Train and evaluate the NEGA model.
    post                Perform post-processing of evaluation results.

options:
  -h, --help            show this help message and exit
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or issues, please open an issue on the repository.