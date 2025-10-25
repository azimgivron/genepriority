# genepriority 🚀

## Algorithm

We’re using a family of matrix completion models:

1. **Non-Euclidean Gradient Algorithm (NEGA)** – the original optimizer for standard matrix completion ([Ghaderi et al., 2022]()).
2. **NEGA-GPFS** – NEGA with factorization in the feature spaces.
3. **NEGA-GPR** – NEGA wit side information in regularization.
4. **GeneHound** – A Bayesian matrix factorization method ([Zakeri et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29949967/)).

### Complete References
- Ghaderi, S., Moreau, Y., & Ahookhosh, M. (2022). *Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications*. JMLR, 23(2022):1-44.
- Zakeri, P., Simm, J., Arany, A., ElShal, S., & Moreau, Y. (2018). *Gene prioritization using Bayesian matrix factorization with genomic and phenotypic side information.* Bioinformatics, 34(13), i447–i456. doi:10.1093/bioinformatics/bty289.

---

## Overview 🔍


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

*Note:* TensorBoard logs live at `/home/TheGreatestCoder/code/logs` inside the container.

---

## Requirements

- **Python 3.11**  
- **pip**  
- **venv** module (for virtual environments)

---

## Repository Structure 🗂️

### Folder Structure

```bash
genepriority/
├── models/                # NEGA variants and GeneHound implementations
├── evaluation/            # Tools for evaluation metrics
├── preprocessing/         # DataLoader for gene-disease data
├── scripts/
│   ├── genehound          # Reproducing GeneHound results
│   └── nega               # NEGA fine-tuning & cross-validation scripts
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
- **nega:** Train and evaluate the appropriate NEGA variant (standard, IMC, or GeneHound) using cross-validation depending on the provided options.

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
