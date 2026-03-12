import matplotlib.pyplot as plt

models = ["RF", "LGBM", "LGBM+Adv1", "LGBM+Adv2"]
evasion = [1.0, 0.0, 0.0, 0.0]
failure = [0.0, 1.0, 1.0, 1.0]

plt.figure(figsize=(8, 4))
plt.bar(models, evasion, label="Evasion Rate")
plt.bar(models, failure, bottom=evasion, label="Failure Rate")

plt.ylabel("Rate")
plt.title("Adversarial Robustness Comparison")
plt.legend()
plt.tight_layout()
plt.show()
