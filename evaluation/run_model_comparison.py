from attack.ga_engine import run_attack
from ids.oracle_manager import load_model

MODELS = ["rf", "lgbm", "lgbm_adv1", "lgbm_adv2"]
RUNS = 50

results = {}

for model in MODELS:
    print(f"\n🔍 Evaluating model: {model}")
    load_model(model)

    success = 0
    total_gen = 0
    total_queries = 0
    failures = 0

    for i in range(RUNS):
        res = run_attack()
        if res["success"]:
            success += 1
            total_gen += res["generation"]
            total_queries += res["queries"]
        else:
            failures += 1

    results[model] = {
        "evasion_rate": success / RUNS,
        "failure_rate": failures / RUNS,
        "avg_generations": total_gen / max(success, 1),
        "avg_queries": total_queries / max(success, 1)
    }

print("\n========== MODEL COMPARISON ==========")
for model, stats in results.items():
    print(f"\nModel: {model}")
    for k, v in stats.items():
        print(f"{k:20}: {v}")
print("====================================")
