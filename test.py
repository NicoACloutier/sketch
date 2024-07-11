import aggregate
import numpy as np
import trimesh

def main():
    
    ## strait line, hardcoded
    s_i_one = np.array( [ [0, 0, 0], [1, 1, 0], [2, 2.5, 0] ] )
    s_j_one = np.array( [ [2, 1.5, 0], [3, 3, 0], [4, 4, 0] ] )
    s_a_one = aggregate.aggregate(s_i_one, s_j_one)
    lines = [trimesh.load_path(line) for line in [s_i_one, s_j_one, s_a_one]]
    for line in lines:
        line.colors = [np.append(np.random.random(3), 1)]
    trimesh.Scene(lines[:-1]).show()
    trimesh.Scene(lines).show()

    ## curved line, perfect function 
    s_i_two = np.array( [ [x/1000, (x/1000)**1.05, 0] for x in range(70) ] ) 
    s_j_two = np.array( [ [x/1000, x/1000-.02, 0] for x in range(30, 100) ] ) 
    s_a_two = aggregate.aggregate(s_i_two, s_j_two)
    lines = [trimesh.load_path(line) for line in [s_i_two, s_j_two, s_a_two]]
    for line in lines:
        line.colors = [np.append(np.random.random(3), 1)]
    trimesh.Scene(lines[:-1]).show()
    trimesh.Scene(lines).show()

if __name__ == "__main__":
    main()
