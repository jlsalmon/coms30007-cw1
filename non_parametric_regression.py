import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


def generate_k(x, ls):
    """ Generate the covariance matrix K """
    return np.exp(-cdist(x, x)**2/(ls**2))


def main():
    fig = plt.figure(figsize=(14, 8))
    x = np.linspace(-2, 2, 201)
    x = x.reshape(-1, 1)
    length_scales = [0.1, 0.5, 1, 2.5, 5, 7.5]
    mean = np.zeros(x.shape)

    for i, ls in enumerate(length_scales):
        K = generate_k(x, ls)
        Y = np.random.multivariate_normal(mean.flatten(), K, 5)
        ax = fig.add_subplot(2, 3, i + 1)
        for y in Y:
            ax.plot(x, y)
        ax.set_title("Length scale: {} ".format(ls))
    plt.show()


if __name__ == '__main__':
    main()
