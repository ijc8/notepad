import numpy as np
from dollarpy import Point

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

def convert_to_dollar(g_vectors):
    points = []
    for idx, stroke in enumerate(g_vectors):
        for point in stroke:
            points.append(Point(point[0], point[1], idx))

    return points

class ResultWrapper:
    def __init__(self, result):
        self.best = {
            'name': result[0],
            'score': result[1],
            'dist': result[1],
        }
        self._gesture_obj = None

        if result[0]:
            self.results = {
                result[0]: {
                    'name': result[0],
                    'score': result[1],
                    'dist': result[1],
                },
            }
        else:
            self.results = {}
