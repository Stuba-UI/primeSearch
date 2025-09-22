import sympy
import random

def generate_random_formula(operators, max_depth=3):
    x = sympy.Symbol('x')
    def build(depth):
        if depth == 0 or (depth < max_depth and random.random() < 0.3):
            return sympy.Integer(random.randint(1, 20)) * x + sympy.Integer(random.randint(-10, 10))
        op = random.choice(operators)
        if op in {"+", "-", "*", "/"}:
            return getattr(sympy, op)(build(depth-1), build(depth-1))
        elif op == "**":
            return build(depth-1) ** random.randint(2, 3)
        elif op in {"sin", "cos", "log", "exp"}:
            return getattr(sympy, op)(build(depth-1))
        else:
            return x
    return build(max_depth)

def mutate_formula(formula, operators, mutation_rate=0.3):
    # Simple mutation: randomly change a subexpression
    if random.random() > mutation_rate:
        return formula
    return generate_random_formula(operators)

def crossover_formulas(f1, f2, operators):
    # Simple crossover: randomly choose one of the parents
    return random.choice([f1, f2])
