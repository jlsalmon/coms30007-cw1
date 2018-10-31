import itertools as it

import numpy as np
import matplotlib.pyplot as plt


def main():
    D = generate_D()
    mean1 = np.zeros(1)
    mean2 = np.zeros(2)
    mean3 = np.zeros(3)
    covariance1 = (10 ** 3) * np.eye(1)
    covariance2 = (10 ** 3) * np.eye(2)
    covariance3 = (10 ** 3) * np.eye(3)

    num_samples = 10

    data = np.zeros([4, 512])

    for i in range(512):
        data[0][i] = monte_carlo(m0, D[i], np.zeros(num_samples))
        data[1][i] = monte_carlo(m1, D[i], sample_prior(mean1, covariance1, num_samples))
        data[2][i] = monte_carlo(m2, D[i], sample_prior(mean2, covariance2, num_samples))
        data[3][i] = monte_carlo(m3, D[i], sample_prior(mean3, covariance3, num_samples))

    index = order_data_sets(np.sum(data, axis=0))

    plt.plot(data[3, index], 'g', lw=0.5, label="H3")
    plt.plot(data[2, index], 'r', lw=0.5, label='H2')
    plt.plot(data[1, index], 'b', lw=0.5, label='H1')
    plt.plot(data[0, index], 'm--', lw=0.5, label='H0')
    plt.legend()
    plt.show()

    print(np.sum(data[0]))
    print(np.sum(data[1]))
    print(np.sum(data[2]))
    print(np.sum(data[3]))

    plt.plot(data[3, index], 'g', lw=0.5, label="H3")
    plt.plot(data[2, index], 'r', lw=0.5, label='H2')
    plt.plot(data[1, index], 'b', lw=0.5, label='H1')
    plt.plot(data[0, index], 'm--', lw=0.5, label='H0')
    plt.xlim(0, 60)
    plt.legend()
    plt.show()


def sample_prior(mu, sigma, num_samples):
    return np.random.multivariate_normal(mu, sigma, num_samples)


def monte_carlo(model, D, samples):
    evidence = []

    for s in samples:
        evidence.append(model(D, s))

    return np.sum(evidence) / len(samples)


def order_data_sets(data):
    # Create distance matrix
    size = data.shape[0]
    distance = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            distance[i, j] = data[i] - data[j]
            if i == j:
                distance[i, j] = np.inf

    L = []
    D = list(range(data.shape[0]))

    # Chose start of data set L as argmin
    LL = data.argmin()

    D.remove(LL)
    L.append(LL)

    while len(D) != 0:
        N = []

        # Find set of points in D with L as nearest neighbour
        for k in range(len(D)):
            # Get the nearest neighbour to D[k]
            n = distance[D[k], D].argmin()
            if D[n] == LL:
                N.append(D[n])
        if not N:
            # Choose nearest neighbour in D to L
            LL = D[distance[LL, D].argmin()]
        else:
            # Choose furthest point from L in N
            LL = N[distance[LL, N].argmin()]
        D.remove(LL)
        L.append(LL)
    return L


def generate_D():
    product = list(it.product([-1, 1], repeat=9))
    D = []
    for d in product:
        D.append(np.reshape(np.asarray(d), (3, 3)))
    return D


def m0(d, _):
    return 1.0 / 512


def m1(d, theta):
    p = 1.0
    for i in range(3):
        for j in range(3):
            e = np.exp(-d[i, j] * theta[0] * (i - 1))
            p = p * 1 / (1 + e)
    return p


def m2(d, theta):
    p = 1.0
    for i in range(3):
        for j in range(3):
            e = np.exp(-d[i, j] * (theta[0] * (i - 1) + theta[1] * (j - 1)))
            p = p * 1 / (1 + e)
    return p


def m3(d, theta):
    p = 1.0
    for i in range(3):
        for j in range(3):
            e = np.exp(-d[i, j] * (theta[0] * (i - 1) + theta[1] * (j - 1) + theta[2]))
            p = p * 1 / (1 + e)
    return p


if __name__ == '__main__':
    main()
