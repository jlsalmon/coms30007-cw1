import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# -----------------------------------------------------------------------------
# These are the parameters we wish to infer
TRUE_W = [-1.3, 0.5]
# -----------------------------------------------------------------------------


def main():
    # Noise variance
    alpha = 2.0
    beta = (1 / 0.2) ** 2

    # Generate some data using the real parameters
    x = np.linspace(-2, 2, num=20); np.random.shuffle(x)
    # Generate the true y-values (plus some noise)
    y = f(x, TRUE_W[0], TRUE_W[1]) + np.random.normal(scale=0.3, size=x.shape)

    # These are the x and y axis values in W-space
    w0 = np.linspace(-2, 2, 200)
    w1 = np.linspace(-2, 2, 200)

    # Define a D-dimensional starting mean vector. Assume that the mean
    # is 0, 0 to begin with
    w_mean = np.array([0, 0])

    # Define a DxD starting covariance matrix
    w_covariance = np.linalg.inv(alpha * np.eye(2))

    # Fit the data (calculate the new mean and covariance based on the given
    # data). In the initial case we use no data, so this is basically a noop.
    w_mean, w_covariance = calculate_posterior(x[0:0], y[0:0], w_mean, w_covariance, beta)

    # Plot the prior
    plt.figure(figsize=(10, 12))
    plt.subplot(4, 3, 2)
    plot_prior(w0, w1, w_mean, w_covariance)

    # Make some initial predictions of W and plot them as functions
    predictions = predict(x, w_mean, w_covariance, 6)
    plt.subplot(4, 3, 3)
    plot_predictions(x, predictions)

    # Start with 0 data points, progressively adding more
    for i, n in enumerate([1, 2, 20]):

        # Visualise likelihood
        plt.subplot(4, 3, 3*i + 4)
        # TODO: I think this should be the conditional gaussian p(x_n | mu)
        plot_likelihood(w0, w1, x[0:n], y[0:n], beta)

        # Fit again with some real data
        w_mean, w_covariance = calculate_posterior(x[0:n], y[0:n], w_mean, w_covariance, beta)

        # Plot the posterior
        plt.subplot(4, 3, 3*i + 5)
        plot_posterior(w0, w1, w_mean, w_covariance)

        # Make some more predictions and plot them
        predictions = predict(x, w_mean, w_covariance, 6)
        plt.subplot(4, 3, 3*i + 6)
        plot_predictions(x, predictions)

        # Plot the actual data points for reference
        plt.scatter(x[0:n], y[0:n], s=50, zorder=10)

    plt.tight_layout()
    plt.show()


def f(x, w0, w1):
    return w0 * x + w1
    # return w0 + w1 * x


def plot_prior(w0, w1, mean, covariance):
    return plot_posterior(w0, w1, mean, covariance)


def plot_likelihood(w0, w1, x, y, beta):
    n = w0.shape[0]
    dist = np.zeros((n, n))

    x = x[-1]
    y = y[-1]

    # TODO: can this be rewritten in terms of numpy/scipy functions?
    for xt in range(0, n - 1):
        for yt in range(0, n - 1):
            mean = f(x, w0[xt], -w1[yt])

            norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * (1 / beta))
            exponent = -0.5 * (y - mean) * beta * (y - mean)

            dist[xt, yt] = (norm_const * np.exp(exponent))

    dist2 = np.random.mul

    plt.title('likelihood')
    plt.gca().set_aspect('equal')
    plt.xlabel('$\mathregular{w_0}$')
    plt.ylabel('$\mathregular{w_1}$')

    # plt.imshow(dist, extent=[-2, 2, -2, 2])
    plt.contourf(w0, w1, dist)
    # plt.contourf(w0, w1, multivariate_normal.pdf(dist))

    plt.plot(TRUE_W[0], TRUE_W[1], 'r+')


def calculate_posterior(x, y, mean, covariance, beta):
    # This is the "design matrix" as defined on p142
    phi_x = phi(x)
    # Precision is the inverse of the covariance
    precision = np.linalg.inv(covariance)

    new_precision = precision + beta * phi_x.T.dot(phi_x)
    new_mean = np.linalg.solve(
        new_precision,
        precision.dot(mean) + beta * phi_x.T.dot(y)
    )
    new_covariance = np.linalg.inv(new_precision)
    return new_mean, new_covariance


def plot_posterior(w0, w1, mean, covariance):
    # M, N = w0.shape[0], w1.shape[0]
    # dist = np.zeros((M, M))
    # S_N_inv = alpha * np.eye(2, 2) + beta * PHI.T * PHI
    # S_N = inv(S_N_inv)
    # m_N = beta * S_N * PHI.T * data
    # for xt in range(0, M - 1):
    #     for yt in range(0, M - 1):
    #         x_v = np.array([w0[xt], w1[yt]])
    #
    #         norm_const = (1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(S_N))))
    #
    #         tmp3 = np.array([[x_v[0] - m_N[0, 0], x_v[1] - m_N[1, 0]]])
    #         exponent = -0.5 * tmp3 * np.linalg.inv(S_N) * tmp3.T
    #
    #         w_dist[xt, yt] = norm_const * np.exp(exponent)
    #
    # plt.imshow(w_dist, extent=[0, 100, 0, 100])
    # plt.contourf(w_dist)

    plt.title('prior/posterior')
    plt.gca().set_aspect('equal')
    plt.xlabel('$\mathregular{w_0}$')
    plt.ylabel('$\mathregular{w_1}$')

    plt.contourf(w0, w1, multivariate_normal.pdf(
        np.dstack(np.meshgrid(w0, w1)), mean=mean, cov=covariance))
    plt.plot(TRUE_W[0], TRUE_W[1], 'r+')


def predict(x, mean, covariance, num_samples):
    phi_x = phi(x)
    w_sample = np.random.multivariate_normal(
        mean, covariance, size=num_samples)
    y = phi_x.dot(w_sample.T)
    return y


def plot_predictions(x, y):
    plt.title('data space')
    plt.gca().set_aspect('equal')
    plt.xlabel('$\mathregular{x}$')
    plt.ylabel('$\mathregular{y}$')

    plt.plot(x, y, "-r")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


def phi(x):
    return np.column_stack((phi_0(x), phi_1(x)))


def phi_0(x):
    return x


def phi_1(x):
    return np.ones(np.size(x))


if __name__ == '__main__':
    main()
