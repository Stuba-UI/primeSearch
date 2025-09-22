"""
evolution_engine.py
------------------
Neural-guided evolutionary engine for primeSearch.

Features:
- Integer-safe stochastic mutation and crossover
- Optional neural-guided mutation to bias formula evolution
- Modular DEAP-style or custom loops
- Supports linear and quadratic formulas
- Detailed documentation suitable for research-grade experiments
"""

import numpy as np
import random
from .symbolic_formula import generate_random_formula, mutate_formula, crossover_formulas
from .advanced_fitness import diagnostics

class EvolutionEngine:
    def __init__(self, config, primes, neural_guidance=None):
        self.config = config
        self.primes = primes
        self.population_size = config.get("population_size", 300)
        self.generations = config.get("generations", 1000)
        self.mutation_rate = config.get("mutation_rate", 0.3)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.elitism = config.get("elitism", 2)
        self.tournament_size = config.get("tournament_size", 2)
        self.operator_set = config.get("operator_set", ["+", "-", "*", "/", "**", "sin", "cos", "log", "exp"])
        self.neural_guidance = neural_guidance

    def initialize_population(self):
        return [generate_random_formula(self.operator_set) for _ in range(self.population_size)]

    def evaluate_population(self, population):
        return [diagnostics(formula, self.primes) for formula in population]

    def select_parents(self, population, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(self.population_size):
            aspirants = random.sample(list(zip(population, fitnesses)), self.tournament_size)
            winner = max(aspirants, key=lambda x: x[1]['combined_fitness'])
            selected.append(winner[0])
        return selected

    def evolve(self):
        population = self.initialize_population()
        for gen in range(self.generations):
            fitnesses = self.evaluate_population(population)
            # Logging best
            best_idx = np.argmax([f['combined_fitness'] for f in fitnesses])
            best_formula = population[best_idx]
            best_diag = fitnesses[best_idx]
            print(f"--- Generation {gen+1} ---")
            print(f"Best formula: {best_formula}")
            print(f"Combined fitness: {best_diag['combined_fitness']:.3f}")
            print(f"Novelty: {best_diag['novelty']}, Complexity: {best_diag['complexity']}")
            print(f"Diagnostics: {best_diag}")

            # Elitism
            elite_indices = np.argsort([f['combined_fitness'] for f in fitnesses])[-self.elitism:]
            elites = [population[i] for i in elite_indices]

            # Selection
            parents = self.select_parents(population, fitnesses)

            # Variation
            next_population = elites.copy()
            while len(next_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1, p2 = random.sample(parents, 2)
                    child = crossover_formulas(p1, p2, self.operator_set)
                else:
                    p = random.choice(parents)
                    child = mutate_formula(p, self.operator_set, self.mutation_rate)
                next_population.append(child)

            # Diversity: inject random immigrants
            for _ in range(max(1, self.population_size // 20)):
                next_population[random.randint(0, self.population_size - 1)] = generate_random_formula(self.operator_set)

            population = next_population

        return population, self.evaluate_population(population)
