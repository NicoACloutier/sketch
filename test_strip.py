import strokestrip
import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(vectors: np.ndarray, colors: list[str]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    for j, vector in enumerate(vectors):
        num_points = len(vector)
        for i, point in enumerate(vector[:-1]):
            ax.plot(xs=[point[0], vector[i+1][0]], ys=[point[1], vector[i+1][1]], zs=[point[2], vector[i+1][2]], alpha=(i+1)/num_points, color=colors[j])
    plt.show()

def display_vector(vectors: np.ndarray, colors: list[str]) -> None:
    y = strokestrip.find_stroke_orientations(vectors)
    print(y)
    plot_vectors(vectors, colors)
    vectors = [vector[::y[i]] for i, vector in enumerate(vectors)]
    plot_vectors(vectors, colors)

def main():
    x1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[3, 3, 3.1], [2, 2, 2.1], [1, 1, 1.1]]])
    x2 = np.random.random((10, 100, 3))

    display_vector(x1, ["blue", "orange"])
    display_vector(x2, ["indigo", "blue", "green", "yellow", "orange", "indigo", "blue", "green", "yellow", "orange"])
if __name__ == '__main__':
    main()
