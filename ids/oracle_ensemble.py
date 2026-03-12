from ids.oracle_rf import ids_oracle as rf_oracle
from ids.oracle_lgbm import ids_oracle as lgbm_oracle
from ids.oracle_fnn import ids_oracle as fnn_oracle

# ===============================
# GLOBAL QUERY BUDGET
# ===============================
QUERY_LIMIT = 1000
_query_count = 0


class QueryBudgetExceeded(Exception):
    pass


def reset_budget():
    global _query_count
    _query_count = 0


def get_query_count():
    return _query_count


# ===============================
# ENSEMBLE ORACLE (MAJORITY VOTE)
# ===============================
def ids_oracle(sample):
    global _query_count

    if _query_count >= QUERY_LIMIT:
        raise QueryBudgetExceeded()

    _query_count += 1

    preds = [
        rf_oracle(sample),
        lgbm_oracle(sample),
        fnn_oracle(sample)
    ]

    # attack = 1 , benign = 0
    return int(sum(preds) >= 2)