import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import inv

w = [-1.3, .5]


def plot_prior(N):
    """ Plot the prior gaussian distribution """
    k = [[1, 0],
         [0, 1]]
    mean = np.array([0,0])
    w_prior = multivariate_normal(mean.flatten(), k)
    x = np.linspace(-2, 2, 201)
    x1p, x2p = np.meshgrid(x, x)
    x1p_flat = x1p.flatten()
    x2p_flat = x2p.flatten()
    pos = np.vstack((x1p_flat, x2p_flat))
    pos = pos.T
    Z = w_prior.pdf(pos)
    Z = Z.reshape(N, N)
    plt.figure(figsize=(4, 4))
    plt.plot(w[0], w[1], 'w+')
    plt.ylabel('w1')
    plt.xlabel('w0')
    plt.imshow(Z, cmap='jet', extent=(-2, 2, -2, 2))
    plt.title('Prior')
    plt.show()
    return Z, pos


def get_random_x():
    """Return a random number in the range of -1 to 1 """
    return round(np.random.uniform(-1, 1), 2)


def get_xy_value(mean):
    """ Get a random x value, and generate its y """
    xi = get_random_x()
    # sig = np.sqrt(0.3)                        # Deviation
    # epsilon = sig * np.random.randn() + mean[0]  # Noise
    epsilon = 0
    # Todo: Why is this a minus instead of yi = w[0] * xi + w[1]
    yi = w[0] * xi - w[1] + epsilon
    return xi, yi


def get_ys(xy):
    """Retrieve y values fro xy pairs"""
    ys = []
    for x_y in xy:
        ys.append(x_y[1])
    Y = np.array([ys])
    return Y.T


def get_x_matrix(XY):
    """Generate a matrix of the x values and 1's """
    x_vals = np.array([[x[0] for x in XY]])
    blank = np.array([np.ones(len(XY))])
    return np.concatenate((x_vals.T, blank.T), axis=1)


def get_posterior(X, Y, cur_cov):
    """ Calculate the posterior covariance, mean and w values """
    posterior_cov = inv((1 / 0.3) * np.dot(X.T, X) + inv(cur_cov))
    posterior_mean = (1 / 0.3) * np.dot(posterior_cov, np.dot(X.T, Y))
    posterior_w = multivariate_normal(posterior_mean.flatten(), posterior_cov)
    return posterior_cov, posterior_mean, posterior_w


def plot_posterior(Z_posterior, n):
    """Plot the posterior"""
    plt.figure(figsize=(4, 4))
    plt.imshow(Z_posterior, cmap='jet', extent=(-2, 2, -2, 2))
    plt.plot(w[0], w[1], 'w+')
    plt.ylabel('w1')
    plt.xlabel('w0')
    plt.title('Posterior at iteration {}'.format(n+1))
    plt.show()


def plot_samples(ws, xy, n):
    """Plot sample functions and the data used to generate them"""
    for i, (w0, w1) in enumerate(ws):
        plt.plot(np.linspace(-2, 2, 201), w0 * np.linspace(-2, 2, 201) + w1)
        print(w0, w1)
    plt.scatter([x_y[0] for x_y in xy], [x_y[1] for x_y in xy], s=100, marker="o")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sample functions at iteration {}'.format(n+1))
    plt.show()


def main():
    # Setup variables
    prior_mean = np.array([0, 0])
    prior_cov = [[1, 0],
                 [0, 1]]
    num_data = 201
    xy = []
    # Plot prior distribution
    post_z, pos = plot_prior(num_data)
    for i in range(20):
        xy.append(get_xy_value(prior_mean))
        Y = get_ys(xy)
        X = get_x_matrix(xy)
        post_cov, post_mean, post_w = get_posterior(X, Y, prior_cov)
        post_z = post_w.pdf(pos).reshape(num_data, num_data)
        if i == 0 or i == 2 or i == 19:
            plot_posterior(post_z, i)
            ws = np.random.multivariate_normal(post_mean.flatten(), post_cov, 5)
            plot_samples(ws, xy, i)


if __name__ == '__main__':
    main()