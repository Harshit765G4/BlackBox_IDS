import random
import pandas as pd
from pathlib import Path
from deap import base, creator, tools

from .genome import BOUNDS
from .constraints import repair
from .fitness import evaluate

from ids.oracle_ensemble import reset_budget, get_query_count

# ===============================
# PATH
# ===============================
BASE_DIR = Path(__file__).resolve().parents[1]

# ===============================
# LOAD MALICIOUS SEEDS
# ===============================
seeds = pd.read_csv(
    BASE_DIR / "dataset" / "malicious_seeds.csv"
).values.tolist()

# ===============================
# DEAP SETUP
# ===============================
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def init_individual(seed):
    return [val * random.uniform(0.8, 1.2) for val in seed]


toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    lambda: init_individual(random.choice(seeds))
)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.4)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)


def evaluate_and_repair(individual):
    repaired = repair(individual, BOUNDS)
    individual[:] = repaired
    return evaluate(individual, individual)


toolbox.register("evaluate", evaluate_and_repair)

# ===============================
# MAIN GA ATTACK
# ===============================
def run_attack():

    reset_budget()

    POP_SIZE = 80
    MAX_GEN = 80

    pop = toolbox.population(n=POP_SIZE)

    for gen in range(MAX_GEN):

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

            if ind.fitness.values[0] == 1.0:
                return {
                    "success": True,
                    "generation": gen,
                    "queries": get_query_count(),
                    "adversarial_sample": list(ind)
                }

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)

        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)

        pop[:] = offspring

    return {
        "success": False,
        "generation": None,
        "queries": get_query_count(),
        "adversarial_sample": None
    }