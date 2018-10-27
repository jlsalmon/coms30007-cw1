import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from numpy.linalg import inv

N = 100
D = 10
W = np.array([-1.3, 0.5])
sigma = 0.2
mu = np.zeros(2)

args = ()


def f(x, *args):
    """Return the value of the objective at x"""
    # See summary document 7.2 and also Bishop p574
    C = W.dot(W.T) + ((sigma ** 2) * np.eye(len(x)))

    term_1 = (N * D / 2) * np.log(2 * np.pi)
    term_2 = (N/2) * (np.log(np.linalg.det(C)))
    term_3 = 0

    for i in range(0, N):
        term_3 += (x[i] - mu).T * np.linalg.inv(C) * (x[i] - mu)

    return -term_1 - term_2 - (1/2) * term_3


def dfx(x, *args):
    """Return the gradient of the objective at x"""
    # See the appendix in the assignment
    C = W.dot(W.T) + ((sigma ** 2) * np.eye(len(x)))

    # TODO: this part is not correct
    dC_by_dW = np.diag(C) * np.diag(W)  # np.kron(C, W)  #np.diff(C) / np.diff(W)

    term_2 = np.trace(inv(C).dot(dC_by_dW))

    # TODO: if Y has shape 10x100, how do we multiply it with dC/dW which
    # presumably has shape 2x2??
    Y = f_lin(f_non_lin(x))
    term_3 = np.trace(Y.dot(Y.T).dot(-inv(C).dot(dC_by_dW.dot(inv(C)))))

    return term_2 + term_3


def f_non_lin(xi):
    return np.array([xi * np.sin(xi), xi * np.cos(xi)])


def f_lin(x):
    A = np.random.randn(10, 2)
    return x.T.dot(A.T)
    # return A.dot(x)


def main():
    x = np.linspace(0, 4 * np.pi, 100)

    # Plot Y for reference
    Y = f_lin(f_non_lin(x))
    plt.plot(Y, Y)

    x_star = opt.fmin_cg(f, np.zeros_like(x), fprime=dfx, args=args, disp=False)
    # plt.plot(x_star)

    plt.show()


if __name__ == '__main__':
    main()
