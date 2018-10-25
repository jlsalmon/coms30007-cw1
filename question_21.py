import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

args = ()


def f(x, *args):
    return f_lin(f_non_lin(x))


def f_non_lin(xi):
    return np.array([xi * np.sin(xi), xi * np.cos(xi)])


def f_lin(x):
    A = np.random.randn(10, 2)
    return A.dot(x)


def dfx(x, *args):
    return np.gradient(x)


def plot_y(Y):
    plt.plot(Y)
    plt.show()


def main():
    x = np.linspace(0, 4 * np.pi, 100)
    x_star = opt.fmin_cg(f, np.asarray((0, 0)), fprime=dfx, args=args, disp=False)

    plt.plot(x_star)
    plt.show()


if __name__ == '__main__':
    main()
