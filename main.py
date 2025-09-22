# main.py
import random
import csv
import numpy as np
import json
from deap import tools
from .prime_data import get_primes
from .evolution_engine import EvolutionEngine
from .neural_guidance import NeuralGuidance

# ----------------------------
# 1. Config & Parameters
# ----------------------------
with open("config.json") as f:
    config = json.load(f)

POP_SIZE = config.get("POP_SIZE", 100)
GENERATIONS = config.get("GENERATIONS", 200)
ELITISM_COUNT = config.get("ELITISM_COUNT", 4)
MAX_MUTATION = config.get("MAX_MUTATION", 15)
LOG_FILE = config.get("LOG_FILE", "evolution_log.csv")
RANDOM_NEW_INDIVIDUAL_CHANCE = config.get("RANDOM_NEW_INDIVIDUAL_CHANCE", 0.2)
TOURNAMENT_SIZE = config.get("TOURNAMENT_SIZE", 5)
NOVELTY_WEIGHT = config.get("NOVELTY_WEIGHT", 1.0)
FITNESS_WEIGHT = config.get("FITNESS_WEIGHT", 1.0)
COMPLEXITY_WEIGHT = config.get("COMPLEXITY_WEIGHT", 0.2)
SEED = config.get("SEED", 42)

random.seed(SEED)
np.random.seed(SEED)

def main():
    # ----------------------------
    # 2. Initialize population
    # ----------------------------
    population = [toolbox.individual() for _ in range(POP_SIZE)]

    # ----------------------------
    # 3. Setup CSV logging
    # ----------------------------
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation", "best_formula", "combined_fitness", "fitness", "novelty", "complexity", "strict_hits", "closeness", "variance"
        ])

    # ----------------------------
    # 4. Diversity & Novelty
    # ----------------------------
    def compute_novelty(population):
        # Novelty = average distance to k nearest neighbors in output space
        k = min(5, len(population) - 1)
        outputs = [np.array(diagnostics(ind.formula, primes)["outputs"]) for ind in population]
        novelties = []
        for i, out_i in enumerate(outputs):
            dists = [np.linalg.norm(out_i - out_j) for j, out_j in enumerate(outputs) if i != j]
            dists.sort()
            novelty = np.mean(dists[:k]) if dists else 0.0
            novelties.append(novelty)
        return novelties

    # ----------------------------
    # 5. Safe individual creation
    # ----------------------------
    def safe_individual(parent1=None, parent2=None):
        if random.random() < RANDOM_NEW_INDIVIDUAL_CHANCE:
            ind = toolbox.individual()
            ind = mutate_individual(ind, max_mutation=MAX_MUTATION)
            return ind
        elif parent1 and parent2:
            child = crossover_individuals(parent1, parent2)
            child = mutate_individual(child, max_mutation=MAX_MUTATION)
            return child
        else:
            ind = random.choice(population)
            ind = mutate_individual(ind, max_mutation=MAX_MUTATION)
            return ind

    # ----------------------------
    # 6. Evolution loop
    # ----------------------------
    best_individual = None
    best_fitness = -float("inf")

    for gen in range(1, GENERATIONS + 1):
        # Evaluate population
        for ind in population:
            diag = diagnostics(ind.formula, primes)
            ind.fitness.values = (
                FITNESS_WEIGHT * diag["combined_fitness"],
                NOVELTY_WEIGHT * diag["novelty"],
                -COMPLEXITY_WEIGHT * diag.get("complexity", 0)
            )
            ind._diag = diag  # Store for logging

        # NSGA-II selection for Pareto front
        population = tools.selNSGA2(population, POP_SIZE)
        population.sort(key=lambda x: x.fitness.values[0], reverse=True)
        elites = population[:ELITISM_COUNT]

        # Track best
        if elites[0].fitness.values[0] > best_fitness:
            best_fitness = elites[0].fitness.values[0]
            best_individual = elites[0]

        # Logging
        best_diag = best_individual._diag
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                gen,
                str(best_individual.formula),
                best_diag["combined_fitness"],
                best_diag["fitness"],
                best_diag["novelty"],
                best_diag.get("complexity", 0),
                best_diag.get("strict_hits", 0),
                best_diag.get("closeness", 0),
                best_diag.get("variance", 0)
            ])

        print(f"\n--- Generation {gen} ---")
        print(f"Best formula: {best_individual.formula}")
        print(f"Combined fitness: {best_diag['combined_fitness']}")
        print(f"Novelty: {best_diag['novelty']}, Complexity: {best_diag.get('complexity', 0)}")
        print("Diagnostics:", best_diag)

        # Next generation
        PARENT_POOL_SIZE = int(POP_SIZE * 0.5)
        parent_pool = population[:PARENT_POOL_SIZE]
        new_population = elites.copy()
        while len(new_population) < POP_SIZE:
            parent1, parent2 = random.sample(parent_pool, 2)
            child = safe_individual(parent1, parent2)
            new_population.append(child)
        population = new_population

    print("\n=== FINAL BEST FORMULA ===")
    print(f"{best_individual.formula} | Combined fitness: {best_diag['combined_fitness']}")
    print("Final diagnostics:", best_diag)
    print(f"Evolution log saved to {LOG_FILE}")

    primes = get_primes(config.get("num_primes", 100))
    neural_guidance = NeuralGuidance(config.get("neural_model_path", None))
    engine = EvolutionEngine(config, primes, neural_guidance)
    population, diagnostics = engine.evolve()
    # Optionally, log or visualize results here

if __name__ == "__main__":
    main()
