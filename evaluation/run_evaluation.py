import numpy as np
from attack.ga_engine import run_attack
from ids.oracle_ensemble import reset_budget, get_query_count

print("Oracle = ENSEMBLE")

RUNS = 100
results = []

for i in range(RUNS):

    reset_budget()

    res = run_attack()

    res["queries"] = get_query_count()

    results.append(res)
    print(f"Run {i+1}/{RUNS} → {res}")

successes = [r for r in results if r["success"]]

success_rate = len(successes) / RUNS
avg_gen = np.mean([r["generation"] for r in successes])
avg_queries = np.mean([r["queries"] for r in successes])

print("\n========== ENSEMBLE ATTACK METRICS ==========")
print(f"Evasion Success Rate : {success_rate*100:.2f}%")
print(f"Avg Generations      : {avg_gen:.2f}")
print(f"Avg Queries          : {avg_queries:.2f}")
print("=============================================")