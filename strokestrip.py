import numpy as np

def closest_ortho_idx_on_curve(stroke_i, p_i, stroke_j) -> int:
    origin = stroke_i[p_i]
    ortho = normal(tangent(stroke_i, p_i))
    p1 = origin - 100 * ortho
    p2 = origin + 100 * ortho
    ints, indeces = intersections(stroke_j, p1, p2)
    return -1 if ints.empty() else indeces[np.argmin(np.linalg.norm(ints - origin, axis=1))]

def fill_overlaps(stroke_i: np.ndarray, stroke_j: np.ndarray) -> np.ndarray:
    """
    Create overlaps array.
    Arguments:
        `stroke_i: np.ndarray`: the first stroke.
        `stroke_j: np.ndarray`: the second stroke.
    Returns:
        `np.ndarray`: the overlaps.
    """
    overlaps = np.full(stroke_i.shape[0], -1)
    for i, point in enumerate(stroke_i):
        j = closest_ortho_idx_on_curve(stroke_i, point, stroke_j) 
        if j == -1: continue
        
        vec1 = point - stroke_j[j]
        dist1 = np.linalg.norm(vec1)
        vec1 /= dist1
        vec1_ok = True
        if dist1 > 5e-1:
            line1_i = point + 1e-1*vec1
            line2_i = stroke_j[j] - 1e-1*vec1
            if not intersections(stroke_i, line1_i, line2_i)[0].empty():
                vec1_ok = False
        
        k = closest_ortho_idx_on_curve(stroke_j, j, stroke_i)
        vec2_ok = False
        dist2 = float('inf')
        if k != -1:
            vec2_ok = True
            vec2 = stroke_j[j] - stroke_i[k]
            dist2 = np.linalg.norm(vec2)
            vec2 /= dist2
        
            if dist2 > 5e-1:
                line1_j = stroke_i[k] + 1e-1*vec1
                line2_j = stroke_j[j] - 1e-1*vec1
                if not intersections(stroke_j, line1_j, line2_j)[0].empty():
                    vec2_ok = False

        if not vec1_ok and not vec2_ok: continue
        dist = dist1
        used_i = i
        if not vec1_ok or (vec2_ok and dist2 < dist1):
            used_i = k
            dist = dist2

        if overlaps[used_i] == -1 or dist < np.linalg.norm(stroke_i[used_i] - stroke_j[overlaps[used_i]]):
            overlaps[used_i] = j
    
    return overlaps
    
def compatibility(stroke_i: np.ndarray, stroke_j: np.ndarray) -> float:
    """
    Find the pairwise orientation compatibility $o_{ij}$ as described in section 4.1 of Pagurek van Mossel et al. (2021).
    Arguments:
        `stroke_i: np.ndarray`: first stroke to judge.
        `stroke_j: np.ndarray`: second stroke to judge.
    Returns:
        `float`: pairwise orientation compatibility score $o_{ij}$.
    """
    # create orthogonal cross sections
    overlaps_i = fill_overlaps(stroke_i, stroke_j)
    overlaps_j = fill_overlaps(stroke_j, stroke_i)
 
    # check for a direct continuation
    endpoint_distance, min_i, min_j = float('inf'), 0, 0
    for i in range(1):
        for j in range(1):
            distance = np.linalg.norm(stroke_i[i] - stroke_j[j])
            if distance < endpoint_distance:
                endpoint_distance, min_i, min_j = distance, i, j
    endpoint = endpoint_distance < 1e-1

def find_stroke_orientations(vectors: np.ndarray) -> np.ndarray:
    """
    Find stroke orientation information on a series of raw strokes.
    Arguments:
        `vectors: np.ndarray`: 3D array, raw input strokes to find information on.
    Returns:
        `np.ndarray`: orientation information for the input strokes.
    """
    pass 

def flip_strokes(vectors: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """
    Flip improperly oriented strokes given orientation information.
    Arguments:
        `vectors: np.ndarray`: 3D array, raw input strokes to re-orient.
        `orientations: np.ndarray`: orientation information.
    Returns:
        `np.ndarray`: 3D array, correctly oriented strokes.
    """
    return np.array([vector[::orientations[i]] for i, vector in enumerate(vectors)])

def reorient_strokes(vectors: np.ndarray) -> np.ndarray:
    """
    Perform reorientation on improperly oriented strokes.
    Arguments:
        `vectors: np.ndarray`: 3D array, raw input strokes to re-orient.
    Returns:
        `np.ndarray`: 3D array, correctly oriented strokes.
    """
    orientations = find_stroke_orientations(vectors)
    return flip_strokes(vectors, orientations)

def parameterize(vectors: np.ndarray) -> np.ndarray:
    """
    Parameterize a series of correctly oriented vectors.
    Arguments:
        `vectors: np.ndarray`: correctly-oriented vectors, 3D array.
    Returns:
        `np.ndarray`: generated parameterization.
    """
    pass

def fit(parameters: np.ndarray) -> np.ndarray:
    """
    Fit an aggregate stroke on a series of parameterized strokes.
    Arguments:
        `parameters: np.ndarray`: fully parameterized strokes.
    Returns:
        `np.ndarray`: fitted aggregate stroke, 2D array.
    """
    pass

def stroke_strip(vectors: np.ndarray) -> np.ndarray:
    """
    Complete the entire stroke strip process.
    Arguments:
        `vectors: np.ndarray`: 3D array with a series of strokes to aggregate.
    Returns:
        `np.ndarray`: 2D array with aggregated stroke.
    """
    parameters = parameterize(reorient_strokes(vectors))
    return fit(parameters)
