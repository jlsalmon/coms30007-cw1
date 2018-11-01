import itertools as it

import numpy as np
import matplotlib.pyplot as plt


def main():
    D = create_data()

    mu1, mu2, mu3 = np.zeros(1), np.zeros(2), np.zeros(3)
    c1, c2, c3 = (10 ** 3) * np.eye(1), (10 ** 3) * np.eye(2), (10 ** 3) * np.eye(3)
    num_samples = 100

    results, index = run_experiment(D, num_samples, mu1, mu2, mu3, c1, c2, c3)

    # Q26
    print(np.sum(results[0]))
    print(np.sum(results[1]))
    print(np.sum(results[2]))
    print(np.sum(results[3]))

    # Q27
    plot_results(results, index, 'q27')

    # Q28
    plt.figure(figsize=(10, 5))
    for i in range(4):
        max = np.argmax(results[i])
        ax = plt.subplot(2, 4, (2 * i) + 1)
        plot_board(D[max], ax, "Max for $\mathcal{M}_%s$" % i)
        min = np.argmin(results[i])
        ax = plt.subplot(2, 4, (2 * i) + 2)
        plot_board(D[min], ax, "Min for $\mathcal{M}_%s$" % i)
    plt.tight_layout()
    plt.savefig('report/q28.png')
    plt.show()

    # Q29
    # Using a non-diagonal covariance matrix for the prior
    mu1, mu2, mu3 = np.zeros(1), np.zeros(2), np.zeros(3)
    c1, c2, c3 = np.random.rand(1, 1), np.random.rand(2, 2), np.random.rand(3, 3)
    results, index = run_experiment(D, num_samples, mu1, mu2, mu3, c1, c2, c3)
    plot_results(results, index, 'q29_a')

    # Using a non-zero mean
    mu1, mu2, mu3 = np.full(1, 5), np.full(2, 5), np.full(3, 5)
    c1, c2, c3 = (10 ** 3) * np.eye(1), (10 ** 3) * np.eye(2), (10 ** 3) * np.eye(3)
    results, index = run_experiment(D, num_samples, mu1, mu2, mu3, c1, c2, c3)
    plot_results(results, index, 'q29_b')


def run_experiment(D, num_samples, mu1, mu2, mu3, c1, c2, c3):
    results = np.zeros([4, 512])

    for i in range(len(D)):
        results[0][i] = monte_carlo(m0, D[i], np.zeros(num_samples))
        results[1][i] = monte_carlo(m1, D[i], sample_prior(mu1, c1, num_samples))
        results[2][i] = monte_carlo(m2, D[i], sample_prior(mu2, c2, num_samples))
        results[3][i] = monte_carlo(m3, D[i], sample_prior(mu3, c3, num_samples))

    index = order(np.sum(results, axis=0))
    return results, index


def sample_prior(mu, sigma, num_samples):
    return np.random.multivariate_normal(mu, sigma, num_samples)


def monte_carlo(model, D, samples):
    evidence = []
    for s in samples:
        evidence.append(model(D, s))
    return np.sum(evidence) / len(samples)


def create_data():
    data = []
    for d in list(it.product([-1, 1], repeat=9)):
        data.append(np.reshape(np.asarray(d), (3, 3)))
    return data


def m0(d, _):
    return 1.0 / 512


def m1(d, theta):
    p = 1.0
    for x1 in range(0, 3):
        for x2 in range(0, 3):
            y = d[x1, x2]
            p *= 1 / (1 + np.exp(-y * theta[0] * x1))
    return p


def m2(d, theta):
    p = 1.0
    for x1 in range(0, 3):
        for x2 in range(0, 3):
            y = d[x1, x2]
            p *= 1 / (1 + np.exp(-y * (theta[0] * x1 + (theta[1] * x2))))
    return p


def m3(d, theta):
    p = 1.0
    for x1 in range(0, 3):
        for x2 in range(0, 3):
            y = d[x1, x2]
            p *= 1 / (1 + np.exp(-y * (theta[0] * x1 + (theta[1] * x2) + theta[2])))
    return p


def order(d):
    dist = np.zeros([len(d), len(d)])
    l = []
    d = list(range(len(d)))
    ll = d.argmin()
    d.remove(ll)
    l.append(ll)

    for i in range(len(d)):
        for j in range(len(d)):
            dist[i, j] = d[i] - d[j]
            if i == j:
                dist[i, j] = np.inf

    while len(d) != 0:
        n = []
        for k in range(len(d)):
            n = dist[d[k], d].argmin()
            if d[n] == ll:
                n.append(d[n])
        if n:
            ll = n[dist[ll, n].argmin()]
        else:
            ll = d[dist[ll, d].argmin()]

        l.append(ll)
        d.remove(ll)

    return l


def plot_results(results, index, q):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results[0, index], 'm--', lw=0.5, label='$\mathcal{M}_0$')
    plt.plot(results[1, index], 'b', lw=0.5, label='$\mathcal{M}_1$')
    plt.plot(results[2, index], 'r', lw=1, label='$\mathcal{M}_2$')
    plt.plot(results[3, index], 'g', lw=2, label="$\mathcal{M}_3$")
    plt.legend()
    plt.title('All data sets $D$')
    plt.subplot(1, 2, 2)
    plt.xlim(0, 60)
    plt.plot(results[0, index], 'm--', lw=0.5, label='$\mathcal{M}_0$')
    plt.plot(results[1, index], 'b', lw=0.5, label='$\mathcal{M}_1$')
    plt.plot(results[2, index], 'r', lw=1, label='$\mathcal{M}_2$')
    plt.plot(results[3, index], 'g', lw=2, label="$\mathcal{M}_3$")
    plot_max(results[0], index, 0)
    plot_max(results[1], index, 1)
    plot_max(results[2], index, 2)
    plot_max(results[3], index, 3)
    plt.legend()
    plt.title('Subset of data sets $D$')
    plt.tight_layout()
    plt.savefig('report/%s.png' % q)
    plt.show()


def plot_max(results, index, M):
    max = np.argmax(results)
    maxval = results[max]
    plt.annotate('Max for ${M_%s}$' % M,
                 xy=(index.index(max), maxval), textcoords='data')
    plt.plot(index.index(max), maxval, 'x', c='black')


def plot_board(data, ax, title):
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.gca().set_aspect('equal')
    plt.title(title)

    for i in range(3):
        for j in range(3):
            if data[i][j] == -1:
                ax.plot(i + 0, j + 0, 'o',
                        markersize=30, markeredgecolor='r',
                        markerfacecolor='w', markeredgewidth=5)
            if data[i][j] == 1:
                ax.plot(i + 0, j + 0, 'x',
                        markersize=30, markeredgecolor='b',
                        markerfacecolor='w', markeredgewidth=5)


if __name__ == '__main__':
    main()
