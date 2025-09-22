# primeSearch

**primeSearch** is a research-grade, multi-objective evolutionary framework for discovering formulas that generate prime numbers. It combines advanced symbolic regression, diagnostic metrics, and evolutionary strategies—including parallel, multi-core computation—to explore, evaluate, and evolve candidate prime-generating formulas.

---

## Key Features

- **Multi-objective optimization (NSGA-II):**  
  Simultaneously optimizes for prime prediction accuracy, closeness to target primes, sequence variance, novelty, and formula complexity, enabling the evolution of robust and innovative formulas.

- **Diversity and novelty preservation:**  
  Employs novelty search and diversity metrics to avoid stagnation and encourage exploration of non-trivial, varied formulas.

- **Parallelized evaluation:**  
  Utilizes multi-core CPUs for parallel fitness and diagnostics evaluation, dramatically accelerating large-scale experiments.

- **Extensible diagnostics and logging:**  
  Captures detailed metrics per formula, including:
  - `strict_hits`: number of exact primes generated
  - `closeness`: proximity of output to target primes
  - `variance`: variability in formula outputs
  - `novelty`: uniqueness of output sequences
  - `complexity`: formula length or symbolic complexity  
  All metrics are logged to CSV for post-analysis and visualization.

- **Configurable, reproducible experiments:**  
  All evolutionary parameters, random seeds, and experiment settings are managed via a `config.json` file for full reproducibility and easy experiment management.

- **Prime-focused symbolic evaluation:**  
  Evaluates symbolic formulas using integer-safe computations and penalizes invalid, negative, or non-integer outputs.

- **Modular and extensible architecture:**  
  Easily add new fitness or diagnostic metrics, experiment with different evolutionary algorithms, or integrate with neural guidance and other AI frameworks.

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Stuba-UI/primeSearch
cd primeSearch

2. Create and activate a Python virtual environment (recommended) 

Windows:

python -m venv venv
venv\Scripts\activate

Linux / macOS:

python -m venv venv
source venv/bin/activate

3. Install python dependencies
pip install sympy numpy matplotlib pandas

**Configuration**

Edit config.json to set parameters such as:

	-Population size

	-Number of generations

	-Mutation rates

	-Other evolutionary algorithm settings

This ensures experiments are fully reproducible.

**Run the evaluation:**
Windows:

py fitness_evaluation.py

Linux / macOS:

python3 fitness_evaluation.py

**Visualize evolution progress:**
Windows:

py plot_log.py evolution_log.csv

Linux / macOS:

python3 plot_log.py evolution_log.csv

---

## Diagnostics

`advanced_fitness.py` provides detailed evaluation of candidate formulas, allowing you to track:
- Exact matches to primes
- Proximity to primes for partially correct formulas
- Output variance to monitor diversity
- Novelty to encourage exploration of new sequences
- Formula complexity to balance simplicity and power

---

## Contribution

primeSearch is designed to be modular and extensible. You can:
- Add new fitness or diagnostic metrics
- Experiment with different evolutionary algorithms or neural guidance
- Integrate with other symbolic regression or AI frameworks

---

## Author

Tuomas Lehto

---

## License

This project is released under the MIT License.
