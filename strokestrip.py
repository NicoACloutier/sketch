import numpy as np

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
    pass

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
