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

import random
from deap import base, creator, tools
from symbolic_formula import random_formula, mutate_formula, crossover_formula
from fitness_evaluation import fitness

# ----------------------------
# 1. DEAP setup
# ----------------------------

# Fitness and individual definitions
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", object, fitness=creator.FitnessMax)

def make_individual():
    ind = creator.Individual()
    ind.formula = random_formula()
    return ind

toolbox = base.Toolbox()
toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness evaluation
def eval_individual(individual, primes):
    """
    Evaluate an individual formula over the provided prime sequence.
    Returns a tuple for DEAP compatibility.
    """
    fit = fitness(individual.formula, primes)
    return (fit,)

toolbox.register("evaluate", eval_individual)
toolbox.register("select", tools.selNSGA2)

# ----------------------------
# 2. Neural-Guided Mutation
# ----------------------------

def mutate_individual(ind, nn_model=None, features_tensor=None, max_mutation=5, guided_prob=0.5):
    """
    Mutate an individual formula.
    - Uses neural-guided mutation if model & features are provided
    - Otherwise applies integer-safe stochastic mutation
    - Returns a new Individual object
    """
    formula = ind.formula

    # Neural-guided mutation
    if nn_model is not None and features_tensor is not None and random.random() < guided_prob:
        try:
            suggested_delta = nn_model.predict(features_tensor)
            formula = formula + int(round(suggested_delta))
        except Exception:
            formula = mutate_formula(formula, max_mutation=max_mutation)
    else:
        formula = mutate_formula(formula, max_mutation=max_mutation)

    new_ind = creator.Individual()
    new_ind.formula = formula
    return new_ind

# ----------------------------
# 3. Neural-Guided Crossover
# ----------------------------

def crossover_individuals(ind1, ind2, nn_model=None, features_tensor=None):
    """
    Crossover two individuals:
    - Uses stochastic integer-safe crossover
    - Optional neural-guided adjustment
    """
    formula1 = ind1.formula
    formula2 = ind2.formula

    # Base stochastic crossover
    new_formula = crossover_formula(formula1, formula2)

    # Neural-guided adjustment
    if nn_model is not None and features_tensor is not None:
        try:
            delta = nn_model.predict(features_tensor)
            new_formula = new_formula + int(round(delta))
        except Exception:
            pass  # fallback: keep base crossover result

    new_ind = creator.Individual()
    new_ind.formula = new_formula
    return new_ind
