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
    mean = np.array([0, 0])

    # Define a DxD starting covariance matrix
    covariance = np.linalg.inv(alpha * np.eye(2))

    # Plot the prior (which is effectively just the first posterior)
    prior = calculate_prior(w0, w1, mean, covariance)
    plot_pdf(w0, w1, prior, 2)

    # Make some initial predictions of W and plot them as functions
    predictions = predict(x, mean, covariance, 6)
    plot_predictions(x, predictions, 3)

    # Start with 1 data point, progressively adding more
    for i, n in enumerate([1, 2, 20]):

        # Calculate the likelihood of the newest data point
        likelihood = calculate_likelihood(w0, w1, x[0:n][-1], y[0:n][-1], beta)

        # Plot the likelihood
        plot_pdf(w0, w1, likelihood, 3 * i + 4)

        # Calculate the posterior, given the new data
        # mean, covariance, posterior = calculate_posterior(
        #     w0, w1, x[0:n], y[0:n], mean, covariance, beta, likelihood, prior)
        posterior = calculate_posterior(
                w0, w1, x[0:n], y[0:n], mean, covariance, beta, likelihood, prior)

        # Plot the posterior distribution
        plot_pdf(w0, w1, posterior, 3*i + 5)

        # Make some more predictions and plot them
        predictions = predict(x, mean, covariance, num_samples=6)
        plot_predictions(x, predictions, 3*i + 6)

        # Plot the actual data points for reference
        plt.scatter(x[0:n], y[0:n], s=50, zorder=10)

    plt.tight_layout()
    plt.show()


def f(x, w0, w1):
    """Mapping function f: X -> Y"""
    return w0 * x + w1


def phi(x):
    """Calculate the "design matrix" of x as defined on p142"""
    return np.column_stack((x, np.ones(np.size(x))))


def calculate_prior(w0, w1, mean, covariance):
    """Return the prior pdf over w0 and w1 with given mean and covariance"""
    return multivariate_normal.pdf(
        np.dstack(np.meshgrid(w0, w1)),
        mean=mean,
        cov=covariance
    )


def calculate_likelihood(w0, w1, x, y, beta):
    """Return the likelihood pdf of the given point x, y"""
    n = w0.shape[0]
    pdf = np.zeros((n, n))

    # TODO: can this be rewritten in terms of numpy/scipy functions?
    for i in range(0, n - 1):
        for j in range(0, n - 1):

            mean = f(x, w0[i], w1[j])

            norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * (1 / beta))
            exponent = -0.5 * (y - mean) * beta * (y - mean)

            pdf[j, i] = (norm_const * np.exp(exponent))

    return pdf


def calculate_posterior(w0, w1, x, y, mean, covariance, beta, likelihood, prior):
    """Update the mean and covariance matrices based on the given data"""
    phi_x = phi(x)
    # Precision is the inverse of the covariance
    precision = np.linalg.inv(covariance)

    # p31?
    new_precision = precision + beta * phi_x.T.dot(phi_x)
    new_mean = np.linalg.solve(
        new_precision,
        precision.dot(mean) + beta * phi_x.T.dot(y)
    )
    new_covariance = np.linalg.inv(new_precision)

    pdf = multivariate_normal.pdf(
        np.dstack(np.meshgrid(w0, w1)),
        mean=new_mean,
        cov=new_covariance
    )

    # M, N = w0.shape[0], w1.shape[0]
    # pdf = np.zeros((M, M))
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
    #         pdf[xt, yt] = norm_const * np.exp(exponent)

    return likelihood * prior

    # return new_mean, new_covariance, pdf


def predict(x, mean, covariance, num_samples):
    """Predict a number of y-values given some x-values"""
    w_sample = np.random.multivariate_normal(
        mean, covariance, size=num_samples)
    y = phi(x).dot(w_sample.T)
    return y


def plot_pdf(w0, w1, pdf, i):
    """Make a contour plot over w-space parameterised by the given mean and
    covariance matrices"""
    plt.subplot(4, 3, i)
    # plt.title('prior/posterior')
    plt.gca().set_aspect('equal')
    plt.xlabel('$\mathregular{w_0}$')
    plt.ylabel('$\mathregular{w_1}$')
    plt.contourf(w0, w1, pdf)
    plt.plot(TRUE_W[0], TRUE_W[1], 'r+')


def plot_predictions(x, y, i):
    """Plot the given x and y values as linear functions"""
    plt.subplot(4, 3, i)
    plt.title('data space')
    plt.gca().set_aspect('equal')
    plt.xlabel('$\mathregular{x}$')
    plt.ylabel('$\mathregular{y}$')
    plt.plot(x, y, "-r")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


if __name__ == '__main__':
    plt.figure(figsize=(10, 12))
    main()
