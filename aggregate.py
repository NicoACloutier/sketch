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

def buddy_point(buddy: tuple[int, int], stroke_i: np.ndarray, stroke_j: np.ndarray):
    """
    Find the mean point coordinates between two best buddy points.
    Arguments:
        `buddy: tuple[int, int]`: the index of the buddy points in the first and second stroke, respectively.
        `stroke_i: np.ndarray`: the first stroke.
        `stroke_j: np.ndarray`: the second stroke.
    Returns:
        `np.ndarray`: the average point.
    """
    return np.mean(np.array([stroke_i[buddy[0]], stroke_j[buddy[1]]]), axis=0)

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
    buddy_coords = np.array([buddy_point(buddy, stroke_i, stroke_j) for buddy in buddies])
    i_max, i_min = max(buddies, key=(lambda x: x[1]))[1], min(buddies, key=(lambda x: x[1]))[1]
    j_max, j_min = max(buddies, key=(lambda x: x[0]))[0], min(buddies, key=(lambda x: x[0]))[0]
    stroke_i = np.concat([stroke_i[:i_min], stroke_i[i_max+1:]])
    stroke_j = np.concat([stroke_j[:j_min], stroke_j[j_max+1:]])
    return np.concatenate([stroke_i, buddy_coords, stroke_j]) if i_min > j_min else np.concatenate([stroke_j, buddy_coords, stroke_i])
