import itertools as it

import numpy as np
import matplotlib.pyplot as plt


def main():
    D = generate_D()
    mean1 = np.zeros(1)
    mean2 = np.zeros(2)
    mean3 = np.zeros(3)



    #mean1 = np.full((1), 20)
    #mean2 = np.full((2), 20)
    #mean3 = np.full((3), 20)
    #mean1 = np.random.rand(1, 0)
    #mean2 = np.random.rand(2, 0)
    #mean3 = np.random.rand(3, 0)


    covariance1 = (10 ** 3) * np.eye(1)
    covariance2 = (10 ** 3) * np.eye(2)
    covariance3 = (10 ** 3) * np.eye(3)

    #covariance1 = np.random.rand(1, 1)
    #covariance2 = np.random.rand(2, 2)
    #covariance3 = np.random.rand(3, 3)


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

    max0 = np.argmax(data[0])
    min0 = np.argmin(data[0])

    max1 = np.argmax(data[1])
    min1 = np.argmin(data[1])

    max2 = np.argmax(data[2])
    min2 = np.argmin(data[2])

    max3 = np.argmax(data[3])
    min3 = np.argmin(data[3])

    plotboard(D[min0])
    plotboard(D[max0])
    plotboard(D[min1])
    plotboard(D[max1])
    plotboard(D[min2])
    plotboard(D[max2])
    plotboard(D[min3])
    plotboard(D[max3])


def plotboard(data):
    # create a 8" x 8" board
    fig = plt.figure(figsize=[3, 3])
    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(4):
        ax.plot([x, x], [0, 3], 'k')
    for y in range(4):
        ax.plot([0, 3], [y, y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0, 0, 0, 0])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()
    for i in range(3):
        for j in range(3):
            if data[i][j] == -1:
                ax.plot(i + 0.5, j + 0.5, 'o', markersize=30, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                              markeredgewidth=2)
            if data[i][j] == 1:
                ax.plot(i + 0.5, j + 0.5, 'x', markersize=30, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                        markeredgewidth=2)

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
