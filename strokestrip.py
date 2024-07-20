import numpy as np
import typing, pyoptinterface

def closest_ortho_idx_on_curve(stroke_i: np.ndarray, p_i: int, stroke_j: np.ndarray) -> int:
    """
    Find the closest orthogonal index on a curve to a point on another curve.
    Arguments:
        `stroke_i: np.ndarray`: the curve the original point is on.
        `p_i: int`: the index of the original point.
        `stroke_j: np.ndarray`: the curve to find a point on.
    Returns:
        `int`: the index of the closest point on the second curve.
    """
    origin = stroke_i[p_i]
    ortho = normal(tangent(stroke_i, p_i))
    p1 = origin - 100 * ortho
    p2 = origin + 100 * ortho
    ints, indeces = intersections(stroke_j, p1, p2)
    return -1 if ints.empty() else indeces[np.argmin(np.linalg.norm(ints - origin, axis=1))]

def weight_for_angle(angle: float) -> float:
    """
    Find weight given an angle in radians.
    Arguments:
        `angle: float`: angle in radians.
    Returns:
        `float`: the associated weight.
    """
    deg = angle / np.pi * 180
    sigma, low, high = 10 / 3, 10, 20
    if deg < low: return 1
    return np.exp(-(deg - low)**2 / (2*sigma**2)) if deg < high else 0

def compare_neighbors(n_i: dict[str, typing.Any], n_j: dict[str, typing.Any]) -> bool:
    """
    Compare neighboring points to see which is larger.
    Arguments:
        `n_i: dict[str, typing.Any]`: the first point to compare.
        `n_j: dict[str, typing.Any]`: the second point to compare.
    Returns:
        `bool`: `True` if `n_i` is larger, `False` otherwise.
    """
    return n_i['diff'] < n_j['diff'] if n_i['diff'] != n_j['diff'] else n_i['dist'] < n_j['dist']

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

def check_violations(stroke_i: np.ndarray, stroke_j: np.ndarray, over_i: np.ndarray, over_j: np.ndarray, policy: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float], list[float]]:
    """
    Check for violations in orientation policy.
    Arguments:
        `stroke_i: np.ndarray`: the first stroke.
        `stroke_j: np.ndarray`: the second stroke.
        `over_i: np.ndarray`: the first overlap array.
        `over_j: np.ndarray`: the second overlap array.
        `policy: int`: the predecided policy from the parent function.
    Returns:
        `list[tuple[np.ndarray, np.ndarray]]`: violations lists, with points from both arrays.
        `list[float]`: connection angles.
        `list[float]`: connection distances.
    """
    violations, result_violations, connection_angles, connection_dists = [], [], [], []
    for i, point in enumerate(over_i):
        if point != -1:
            connection_angles.append(np.arccos(policy * tangent(stroke_i, i) @ tangent(stroke_j, point)))
            connection_dists.append(np.linalg.norm(stroke_i[i] - stroke_j[point]))
            if len(violations) > len(result_violations):
                result_violations = violations
                violations = []
        else:
            if i in over_j:
                j = over_j.index(i)
                connection_angles.append(np.arccos(policy * tangent(stroke_i, i) @ tangent(stroke_j, point)))
                connection-dists.append(np.linalg.norm(stroke_i[i] - stroke_j[j]))
            if len(violations) > len(result_violations):
                result_violations = violations
                violations = []
                continue

            # left neighbor
            neighbors = []
            for other_i in range(i)[::-1]:
                if overlaps[other_i] != -1:
                    neighbors.append({'other_i': other_i, 'over': overlaps[other_i], 'diff': abs(i - other_i), 'dist': np.linalg.norm(stroke_i[other_i] - stroke_j[overlaps[other_i]])})
                    break

            # right neighbor
            for other_i, over in enumerate(over_i):
                if over != -1:
                    neighbors.append({'other_i': other_i, 'over': over, 'diff': abs(i - other_i), 'dist': np.linalg.norm(stroke_i[other_i] - stroke_j[other_i])})

            # other neighbor
            closest_j_left = -1
            closest_j_right = -1
            for other_j, over in enumerate(over_j):
                if over == -1: continue
                if over < i and (closest_j_left == -1 or over > over_j[closest_j_left]): closest_j_left = j
                if over > i and (closest_j_right) == -1 or over < over_j[closest_j_right]: closest_j_right = j

            if closest_j_left != -1:
                neighbors.append({'other_i': over_j[closest_j_left], 'over': closest_j_left, 'diff': abs(i - over_j[closest_j_left], 'dist': np.linalg.norm(stroke_i[over_j[closest_j_left]] - stroke_j[closest_j_left]))})
            
            if closest_j_right != -1:
                neighbors.append({'other_i': over_j[closest_j_right], 'over': closest_j_right, 'diff': abs(i - over_j[closest_j_right], 'dist': np.linalg.norm(stroke_i[over_j[closest_j_right]] - stroke_j[closest_j_right]))})
    
            if len(neighbors) == 0: continue
            neighbor = min(neighbors, key=compare_neighbors)
            dist = i - neighbor['other_i']
            j = neighbor['over'] + policy * dist
            if j < 0 or j >= len(stroke_j): continue
            if over_j[j] != -1: continue
            
            dot = policy * (tangent(stroke_i, i) @ tangent(stroke_j, j)) 
            if dot < 0:
                violations.append((stroke_i[i], stroke_j[j]))
            else:
                connection_angles.append(np.arccos(dot))
                connection_dists.append(np.linalg.norm(stroke_i[i] - stroke_j[j]))
    if len(violations) > len(result_violations):
        result_violations = violations

    return result_violations, connection_angles, connection_dists

def evaluate_policy(over_i: np.ndarray, over_j: np.ndarray, stroke_i: np.ndarray, stroke_j: np.ndarray, policy: int) -> dict[str, typing.Any]:
    """
    Evaluate an orientation policy.
    Arguments:
        `over_i: np.ndarray`: the overlap array for the first stroke.
        `over_j: np.ndarray`: the overlap array for the second stroke.
        `stroke_i: np.ndarray`: the first stroke.
        `stroke_j: np.ndarray`: the second stroke.
        `policy: int`: the decided policy.
    Returns:
        `dict[str, typing.Any]`: a dictionary containing the following keys:
            `"violations"`: corresponds to value of type `list[tuple[np.ndarray, np.ndarray]]`, list of violation points on both strokes.
            `"dists"`: corresponds to value of type `list[float]`, list of distances.
            `"angles"`: corresponds to value of type `list[float]`, list of angles.
            `"shortest"`: corresponds to value of type`int`, shortest distance.
    """
    
    # remove overlaps that don't work for this orientation
    shortest = float('inf')
    for i, point in enumerate(over_i):
        if point == -1: continue
        if policy * (tangent(stroke_i, i) @ tangent(stroke_j, point)) <= 0:
            shortest = min(shortest, np.linalg.norm(stroke_i[i] - stroke_j[point]))
            over_i[i] -= 1 
    
    for j, point in enumerate(over_j):
        if point == -1: continue
        if policy * (tangent(stroke_j, j) @ tangent(stroke_i, point)) <= 0:
            shortest = min(shortest, np.linalg.norm(stroke_j[j] - stroke_i[point]))
            over_j[j] -= 1

    viol_i, angles_i, dists_i = check_violations(stroke_i, stroke_j, over_i, over_j)
    viol_j, angles_j, dists_j = check_violations(stroke_j, stroke_i, over_j, over_i)
    return {'violations': viol_i + viol_j, 'angles': angles_i + angles_j, 'dists': dists_i + dists_j, 'shortest': shortest} 

def compatibility(stroke_i: np.ndarray, stroke_j: np.ndarray) -> tuple[float, float]:
    """
    Find the pairwise orientation compatibility $o_{ij}$ as described in section 4.1 of Pagurek van Mossel et al. (2021).
    Arguments:
        `stroke_i: np.ndarray`: first stroke to judge.
        `stroke_j: np.ndarray`: second stroke to judge.
    Returns:
        `float`: result weight.
        `float`: result orienation.
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

    # NOTE: not sure which of these is correct yet
    # has_overlap = any(overlaps_i[0], overlaps_i[-1], overlaps_j[0], overlaps_j[-1]
    has_overlap = -1 in overlaps_i + overlaps_j

    if has_overlap:
        policy_fwd = evaluate_policy(overlaps_i, overlaps_j, stroke_i, stroke_j, 1)
        policy_rev = evaluate_policy(overlaps_i, overlaps_j, stroke_i, stroke_j, -1)
        
        if policy_fwd['shortest'] == float('inf'): policy = 1
        elif policy_rev['shortest'] == float('inf'): policy = -1
        elif len(policy_fwd['violations']) < 2 != len(policy_rev['violations']) < 2:
            policy = 1 if len(policy_fwd['violations']) < 2 else -1
        elif len(policy_fwd['violations']) < MAX_VIOLATIONS and len(policy_rev['violations']) < MAX_VIOLATIONS:
            policy = 1 if policy_fwd['shortest'] > policy_rev['shortest'] else -1
        else:
            policy = 1 if len(policy_fwd['shortest']) < len(policy_rev['shortest']) else -1

        policy_result = policy_fwd if policy == 1 else policy_rev

    tiny_stroke, continuation = False, False
    result = {}

    if not has_overlap or len(policy_result['violations']) < MAX_VIOLATIONS or endpoint:
        len_i, len_j = len(stroke_i), len(stroke_j)
        tiny_stroke = min(len_i, len_j) < 2.5
        if tiny_stroke:
            policy_result['violations'] = []
            if len_i < len_j:
                closest_j = closest_idx_on_curve(stroke_i, 0, stroke_j)
                dot = tangent(stroke_i, 0) @ tangent(stroke_j, closest_j)
                result['orientation'] = 1 if dot > 0 else -1
                result['weight'] = 1e-1
                policy_result['angles'] = [np.arccos(result.orientation * dot)]
                policy_result['dists'] = [np.linalg.norm(stroke_i[0] - stroke_j[closest_j])]
            else:
                closest_i = closest_idx_on_curve(stroke_j, 0, stroke_i)
                dot = tangent(stroke_j, 0) @ tangent(stroke_i, closest_i)
                result['orientation'] = 1 if dot > 0 else -1
                result['weight'] = 1e-1
                policy_result['angles'] = [np.arccos(result['orientation'] * dot)]
                policy_result['dists'] = [np.linalg.norm(stroke_j[0] - stroke_i[closest_i])]
        else:
            result['orientation'] = 1 if min_i != min_j else -1
            endpoint_angle = np.arccos(result['orientation'] * (tangent(stroke_i, min_i * (len(stroke_i) - 1)) @ tangent(stroke_j, min_j * (len(stroke_j) - 1))))
            if not has_overlap or len(policy_result['violations']) > MAX_VIOLATIONS or len(policy_result['angles']) == 0:
                continuation = True
                policy_result['violations'] = []
                policy_result['angles'] = [endpoint_angle]
                policy_result['dists'] = [endpoint_dist]
                result['weight'] = 1e-2 if has_overlap else 1
            else:
                orig_angle_ok = endpoint_angle / np.pi * 180 < 180 - 40
                orig_angle_good = endpoint_angle / np.pi * 180 <= 90

                if orig_angle_ok:
                    continuation = True
                    policy_result['violations'] = []
                    policy_result['angles'] = [endpoint_angle]
                    policy_result['dists'] = [endpoint_dist]
                    result['weight'] = 0.8 if orig_angle_good else 0.1
    
    if not continuation:
        if not tiny_stroke:
            result['orientation'] = policy
        if len(policy_result['angles']) == 0 or (len(policy_result['dists']) <= 2 and min(policy_result['dists'] > 2)):
            result['weight'] = 2e-2
        else:
            angle = np.mean(policy_result['angles'])
            result['weight'] = weight_for_angle(angle) + 1e-1
            len_ratio = len(policy_result['angles']) / (0.95 * min(len(stroke_i), len(stroke_j)))
            len_ratio = min(1, max(0.5, len_ratio))
            result['weight'] *= len_ratio
    
    min_dist = 0
    if len(policy_result['dists']) != 0:
        num_dists = len(policy_result['dists'])
        off = int(0.025 * num_dists)
        min_dist = 3 * policy_result['dists'][off]
        result['weight'] /= 1 + min_dist * min_dist

    return result['weight'], result['orientation']

def find_stroke_orientations(vectors: np.ndarray) -> np.ndarray:
    """
    Find stroke orientation information on a series of raw strokes.
    Arguments:
        `vectors: np.ndarray`: 3D array, raw input strokes to find information on.
    Returns:
        `np.ndarray`: orientation information for the input strokes.
    """
    model = pyoptinterface.highs.Model()
    variables = [model.add_variable(lb=0, ub=1, domain=poi.VariableDomain.Binary, name=f"{i}") for (i, stroke) in enumerate(vectors)]

    obj = 0
    for i, stroke_i in enumerate(vectors):
        for j, stroke_j in enumerate(vectors):
            weight, orientation = compatibility(stroke_i, stroke_j) 
            obj += weight * orientation * (variables[i] - variables[j]) * (variables[i] - variables[j])
    
    model.add_linear_constraint(variables[0], pyoptinterface.Eq, 1.0)
    model.set_objective(obj, pyoptinterface.ObjectiveSense.Minimize)
    model.optimize()

    return np.array([(-1 if model.get_value(var) == 0 else 1) for var in variables])

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
