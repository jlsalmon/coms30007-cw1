import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def generate_k(x, ls):
    """ Generate the covariance matrix K """
    return np.exp(-cdist(x, x)**2/(ls**2))


def main():
    theta = [1.0, 4.0, 0.0, 0.0]

    x = np.linspace(-np.pi, np.pi, 7).T
    # y = [yi(xi) for xi in x]
    plt.plot([np.sin(x_) for x_ in np.linspace(-np.pi, np.pi, 70).T])
    # plt.show()
    x = x.reshape(-1, 1)
    ls = 0.5
    K = generate_k(x, ls)
    mean = np.zeros(x.shape)
    Y = np.random.multivariate_normal(mean.flatten(), K, 1)

    for y in Y:
        plt.scatter(x, y)
    plt.show()


def yi(xi):
    """Return yi, for a given xi"""
    sig = np.sqrt(0.5)                     # Deviation
    epsilon = sig * np.random.randn() + 0  # Noise
    return np.sin(xi) + epsilon


if __name__ == "__main__":
    main()