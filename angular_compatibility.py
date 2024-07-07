import numpy as np
import math

SIGMA_I = 9/3.5
SIGMA_J = 7/3.5

def curve_map(point: np.ndarray, stroke: np.ndarray) -> np.ndarray:
    """
    Map a point `point` onto a curve ($M(p)$ in Liu et al. 2018).
    Arguments:
        `point: np.ndarray`: the point to map.
        `stroke: np.ndarray`: the curve to map it onto.
    Returns:
        `np.ndarray`: point of same dimensions as `point` mapped onto curve.
    """
    return stroke[np.argmin(np.linalg.norm(stroke - point, axis=1))]

def angular_difference(point: np.ndarray, stroke_i: np.ndarray, aggregate_stroke: np.ndarray) -> float:
    """
    Angular difference at point `p_prime` ($A_i$ in Liu et al. 2018).
    Arguments:
        `point: np.ndarray`: the 3D vector point to evaluate at.
        `stroke_i: np.ndarray`: the stroke to evaluate on.
        `aggregate_stroke: np.ndarray`: the aggregate stroke.
    Returns:
        `float`: angular difference score.
    """
    
    ## `point` should be a 3D vector
    assert len(point.shape) == 1
    assert point.shape[0] == 3
    ## `stroke_i` should be a collection of 3D vectors
    assert len(stroke_i.shape) == 2
    assert stroke_i.shape[1] == 3

    mapped_point = curve_map(point, aggregate_stroke) # $p'$ in Liu et al.
    t = unit_tangent(stroke_i, point)
    t_prime = unit_tangent(aggregate_stroke, mapped_point)

    return np.arccos(t @ t_prime)

def average_distance(aggregate_stroke: np.ndarray, stroke_i: np.ndarray, stroke_j: np.ndarray) -> float:
    """
    Average distance between strokes ($D_{i,j}(I_1)$ in Liu et al. 2018).
    Arguments:
        `stroke_i: np.ndarray`: stroke of shape `(n_points, 3)`.
        `stroke_j: np.ndarray`: stroke of shape `(n_points, 3)`.
        `aggregate_stroke: np.ndarray`: aggregate stroke of shape `(n_points, 3)`.
    Returns:
        `float`: Average distance score between strokes.
    """

    ## `stroke_i` and `stroke_j` should be 2D ndarrays
    assert len(stroke_i.shape) == 2
    assert len(stroke_j.shape) == 2
    assert len(aggregate_stroke.shape) == 2
    ## the second dimension should be 3 for 3D
    assert stroke_i.shape[1] == 3
    assert stroke_j.shape[1] == 3   
    assert aggregate_stroke.shape[1] == 3   
   
    distance = sum(np.linalg.norm(curve_map(point, stroke_i) - curve_map(point, stroke_j)) for point in aggregate_stroke)
    return distance / len(aggregate_stroke)  

def angular_distance(stroke_i: np.ndarray, aggregate_stroke: np.ndarray) -> float:
    """
    Angular distance between two strokes ($D_a$ in Liu et al. 2018).
    Arguments:
        `stroke_i: np.ndarray`: stroke of shape `(n_points, 3)`.
        `aggregate_stroke: np.ndarray`: aggregate stroke of shape `(n_points, 3)`.
    Returns:
        `float`: Angular distance `phi` between strokes.    
    """
    
    ## should be 2D ndarrays
    assert len(stroke_i.shape) == 2
    assert len(aggregate_stroke.shape) == 2
    ## the second dimension should be 3 for 3D
    assert stroke_i.shape[1] == 3
    assert aggregate_stroke.shape[1] == 3   

    return sum(angular_difference(point, stroke_i, aggregate_stroke) for point in stroke_i) / len(aggregate_stroke)

def angular_compatibility(stroke_i: np.ndarray, stroke_j: np.ndarray, aggregate_stroke: np.ndarray) -> float:
    """
    Angular compatibility between two strokes ($ComA(Si, S_j)$ in Liu et al. 2018).
    Arguments:
        `stroke_i: np.ndarray`: stroke of shape `(n_points, 3)`.
        `stroke_j: np.ndarray`: stroke of shape `(n_points, 3)`.
        `aggregate_stroke: np.ndarray`: aggregate stroke of shape `(n_points, 3)`.
    Returns:
        `float`: Angular compatibility score between strokes.
    """

    ## `stroke_i` and `stroke_j` should be 2D ndarrays
    assert len(stroke_i.shape) == 2
    assert len(stroke_j.shape) == 2
    assert len(aggregate_stroke.shape) == 2
    ## the second dimension should be 3 for 3D
    assert stroke_i.shape[1] == 3
    assert stroke_j.shape[1] == 3   
    assert aggregate_stroke.shape[1] == 3   

    phi = max(angular_distance(stroke_i, aggregate_stroke), angular_distance(stroke_j, aggregate_stroke))
    
    if phi < 8:
        return 1
    elif phi < 17:
        return math.exp(-1 * (phi - 8)**2 / 2 * SIGMA_I**2)
    elif phi < 23:
        return 0
    elif phi < 30:
        return math.exp(-1 * (phi - 30)**2 / 2 * SIGMA_J**2)
    return -1.5
