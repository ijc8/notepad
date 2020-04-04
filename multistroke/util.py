import numpy as np

def reject_outliers(points, acceptance_threshold=2, max_rounds=10, verbose=False):
    """Reject outliers from a dataset in rounds.

    Terminates after converging, or running max_rounds.

    points: array of input data; flattened in determining mean, stddev
    acceptance_threshold: threshold for including points in a round, expressed in standard deviations

    Returns: indices of selected points (subset of original array; maintains original order)
    """
    points = np.asarray(points)
    old_len = len(points)
    indices = np.arange(old_len)
    for i in range(max_rounds):
        if verbose:
            print(i, old_len)
        mean = points[indices].mean()
        dists = np.abs(points[indices] - mean)
        selector = dists <= np.mean(dists) * acceptance_threshold
        indices = indices[selector]
        if len(indices) == old_len:
            break
        old_len = len(indices)

    if verbose:
        print('final size:', old_len)

    return indices