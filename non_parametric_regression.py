import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def main():
    # Q13 ---------------------------------------------------------------------
    plt.figure(figsize=(14, 14))

    x = np.linspace(-2, 2, 201)
    length_scales = [0.1, 0.5, 2.5]

    # Prior mean
    mean = np.zeros_like(x)

    # Plot sample functions from the prior for different length scales
    for i, l in enumerate(length_scales):
        plt.subplot(3, 3, 3 * i + 1)
        plt.title("l = {}".format(l))

        # Prior covariance for this length scale
        K = k(x, x, l)
        samples = np.random.multivariate_normal(mean, K, 5)

        for sample in samples:
            plt.plot(x, sample)

    # Q14 ---------------------------------------------------------------------

    for i, l in enumerate(length_scales):
        plt.subplot(3, 3, 3 * i + 2)

        # Generate and plot some noisy observations of a true sine wave
        x = np.linspace(-2, 2, 50)
        y = f(x)
        plt.plot(x, y, ':', label='true function')
        y = y + (0.3 * np.random.randn(len(y)))
        plt.plot(x, y, '.', label='noisy data')

        # Compute the predictive posterior distribution
        beta = 3.0
        x_n_plus_one = np.linspace(-3, 3, 101)

        # Bishop p307/308, lecture 6 slide 66
        C_N = k(x, x, l) + (1 / beta) * np.eye(len(x))
        k_ = k(x, x_n_plus_one, l)
        k_T = k_.T
        c = k(x_n_plus_one, x_n_plus_one, l)  # + 1.0 / beta * np.eye(len(x_n_plus_one))

        posterior_mean = k_T.dot((np.linalg.inv(C_N))).dot(y)
        posterior_covariance = c - k_T.dot((np.linalg.inv(C_N))).dot(k_)

        # Plot the predictive mean and predictive variance of the posterior
        # from the data
        plt.plot(x_n_plus_one, posterior_mean, label='predictive mean')
        plt.fill_between(x_n_plus_one,
                         posterior_mean + np.sqrt(np.diag(posterior_covariance)),
                         posterior_mean - np.sqrt(np.diag(posterior_covariance)),
                         alpha=0.2)
        plt.legend()
        plt.title('$\\beta={},l = {}$'.format(beta, l))

        # Plot samples from the posterior
        plt.subplot(3, 3, 3 * i + 3)
        plt.plot(x, y, '.', label='noisy data')
        samples = np.random.multivariate_normal(posterior_mean, posterior_covariance, 3)
        for sample in samples:
            plt.plot(x_n_plus_one, sample)
            plt.title("l = {}".format(l))

    plt.tight_layout()
    plt.savefig('non_parametric_regression.png', bbox_inches='tight')
    plt.show()


def k(xi, xj, l):
    """ Generate the covariance matrix K """
    xi = xi.reshape(-1, 1)
    xj = xj.reshape(-1, 1)
    return np.exp(-cdist(xi, xj) ** 2 / (l ** 2))


def f(x):
    return np.sin(2 * x) + 0.2 * np.sin(x) + 0.1 * x


if __name__ == '__main__':
    main()
