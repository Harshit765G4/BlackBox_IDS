def repair(individual, bounds):
    """
    Repair adversarial sample to stay within realistic feature bounds.
    """

    repaired = []

    keys = list(bounds.keys())

    for i, val in enumerate(individual):
        lim = bounds[keys[i]]

        # clip inside bounds
        val = float(val)
        val = max(lim["min"], min(val, lim["max"]))

        repaired.append(val)

    # ===== Realistic DDoS Constraints =====
    FLOW_PKTS_IDX = 3
    FLOW_BYTES_IDX = 4

    repaired[FLOW_PKTS_IDX] = max(repaired[FLOW_PKTS_IDX], 10)
    repaired[FLOW_BYTES_IDX] = max(repaired[FLOW_BYTES_IDX], 5000)

    return repaired