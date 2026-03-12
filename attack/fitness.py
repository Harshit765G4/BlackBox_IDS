from ids.oracle_ensemble import ids_oracle, QueryBudgetExceeded

def evaluate(individual, original):
    """
    Fitness = 1 if attack succeeds (misclassified as benign)
    """

    try:
        prediction = ids_oracle(individual)

    except QueryBudgetExceeded:
        return (0.0,)

    return (1.0,) if prediction == 0 else (0.0,)