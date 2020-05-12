import numpy as np
from dollarpy import Point
from kivy.graphics import Color, Line

def get_bounds(points):
    return np.array([np.min(points, axis=0), np.max(points, axis=0)])

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

def convert_to_dollar(g_vectors, flip=False):
    points = []

    # Finding the maximum val so that when we turn 180 degrees, there's no negative values.
    # I am sure there's an efficient numpy method
    maximum_val = -np.inf
    for idx, stroke in enumerate(g_vectors):
        for point in stroke:
            maximum_val = max(maximum_val, point[0])
            maximum_val = max(maximum_val, point[1])

    for idx, stroke in enumerate(g_vectors):
        for point in stroke:
            if flip:
                points.append(Point(-1 * point[0] + maximum_val, -1 * point[1] + maximum_val, idx))
            else:
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


def align_note(points, is_note, value, line_spacing):
    "Normalize and translate notes so (0, 0) is where the note should be pinned on a staff."
    mins, maxs = get_bounds(points)
    width, height = maxs - mins
    if is_note:
        # Normalize notes - size of notehead should equal line spacing.
        normalization_factor = (line_spacing / (width / 1.5)) if value == 1/2 else (line_spacing / width)
        points = mins + (points - mins) * normalization_factor
        # Translate - set (0, 0) to center of notehead.
        points_without_outliers = points[reject_outliers(points[:, 1])]
        center = points_without_outliers.mean(axis=0)
        results = points - center

        pitch = is_note
        if (pitch > 72):
            print("pitch", pitch)
            pass

        return results

    normalization_factor = line_spacing / min(width, height)
    points = mins + (points - mins) * normalization_factor

    if value == 4:
        # For whole rests, top should be aligned to y = 0.
        return points - points.mean(axis=0) - line_spacing / 2
    elif value == 2:
        # For half rests, bottom should be aligned to y = 0.
        return points - points.mean(axis=0) + height / 2
    else:
        return points - points.mean(axis=0)


def is_bbox_intersecting_helper(bb, x, y, margin=0):
    minx, miny, maxx, maxy = bb
    minx -= margin
    miny -= margin
    maxx += margin
    maxy += margin
    return minx <= x <= maxx and miny <= y <= maxy

def is_unrecognized_gesture(name, gesture, surface):
    return name is None or too_far_away_from_staff(gesture, surface)


def too_far_away_from_staff(gesture, surface):

    miny = gesture.miny
    maxy = gesture.maxy

    dist = np.inf
    for y in [miny, maxy]:
        for staff_y in [surface.height_min, surface.height_max]:
            dist = min(dist, abs(y - staff_y))

    return (surface.size[1] / 22) < dist

def loaded_id(old_id):
    return f'loaded_id:{old_id}'

def set_color_rgba(surface, stroke_group, rgba):
    for id in stroke_group.ids:
        group = list(surface.canvas.get_group(id))
        for i0, i1 in zip(group, group[2:]):
            if isinstance(i0, Color) and isinstance(i1, Line):
                i0.rgba = rgba
