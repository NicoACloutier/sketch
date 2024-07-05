import numpy as np

def closest(stroke: np.ndarray, point: np.ndarray) -> int:
    """
    Find the index of the closest point to a given point on a curve.
    Arguments:
        `stroke: np.ndarray`: the stroke to search on.
        `point: np.ndarray`: the point to search with.
    Returns:
        `int`: the index of the closest point on the curve.
    """
    return np.argmin(np.linalg.norm(stroke - point, axis=1))

def best_buddies(stroke_i: np.ndarray, stroke_j: np.ndarray) -> list[tuple[int, int]]:
    """
    Find best buddy indeces in two strokes.
    Arguments:
        `stroke_i: np.ndarray`: first stroke to find best buddies in.
        `stroke_j: np.ndarray`: second stroke to find best buddies in.
    Returns:
        `list[tuple[int, int]]`: list of best buddy indeces.
    """
    indeces = [closest(stroke_j, point) for point in stroke_i]
    return [(index, i) for (i, index) in enumerate(indeces) if closest(stroke_i, stroke_j[index]) == i]

def aggregate(stroke_i: np.ndarray, stroke_j: np.ndarray) -> np.ndarray:
    """
    Aggregate two strokes to a common one using best buddies.
    Arguments:
        `stroke_i: np.ndarray`: first stroke to aggregate.
        `stroke_j: np.ndarray`: second stroke to aggregate.
    Returns:
        `np.ndarray`: aggregate stroke.
    """
    buddies = best_buddies(stroke_i, stroke_j)
    i_max, i_min = max(buddies, key=(lambda x: x[0])), min(buddies, key=(lambda x: x[0]))
    j_max, j_min = max(buddies, key=(lambda x: x[0])), min(buddies, key=(lambda x: x[0]))
    raise NotImplementedError()
