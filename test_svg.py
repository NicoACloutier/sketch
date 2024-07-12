from svgpathtools import *

import aggregate
import numpy as np
import trimesh

def sample_path( path, num_samples ):
    '''
    Given:
        path: A path as in an element of the sequence returned from `paths_from_path_to_svg()`
        num_samples: A positive integer which determines the number of times to sample the path.
    Returns:
        A 2D numpy array containing points along the path.
    '''
    
    result = np.zeros( ( num_samples, 2 ) )
    for index, t in enumerate( np.linspace( 0, 1, num_samples ) ):
        point = path.point( t )
        result[index] = point.real, point.imag
    
    return result

def paths_from_path_to_svg( path_to_svg ):
    '''
    Given:
        path: A path to an SVG file
        num_samples: An optional positive integer number of samples.
    '''
    
    try:
        doc = Document( path_to_svg )
        flatpaths = doc.flatten_all_paths()
        paths = [ path for ( path, _, _ ) in flatpaths ]
    except:
        paths, _ = svg2paths( path_to_svg )
    for i in range(len(paths)):
        if paths[i] == Path():
            paths.pop(i)
    
    return paths

def main():
    import sys
    path_to_svg = sys.argv[1]
    paths = paths_from_path_to_svg( path_to_svg )
    
    ## Make the first two paths 3D
    N = 100
    s_i_one = np.append( sample_path( paths[10], N ), np.zeros((N,1)), axis = 1 )
    s_j_one = np.append( sample_path( paths[20], N ), np.zeros((N,1)), axis = 1 )
    
    ## Visualize the input
    lines = [trimesh.load_path(line) for line in [s_i_one, s_j_one]]
    for line in lines:
        line.colors = [np.append(np.random.random(3), 1)]
    trimesh.Scene(lines).show()
    
    ## Visualize the aggregate, too
    s_a_one = aggregate.aggregate(s_i_one, s_j_one)
    lines = [trimesh.load_path(line) for line in [s_i_one, s_j_one, s_a_one]]
    for line in lines:
        line.colors = [np.append(np.random.random(3), 1)]
    trimesh.Scene(lines).show()

if __name__ == "__main__":
    main()
