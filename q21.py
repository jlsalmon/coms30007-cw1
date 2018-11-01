import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from numpy.linalg import inv


def main():
    N = 100
    D = 10

    # The input data that we will try to recover
    x = np.linspace(0, 4 * np.pi, N)
    # Map the data non-linearly to 2D space
    x_dash = f_non_lin(x)
    # Map the data linearly to 10D space using a random parameter matrix
    A = np.random.randn(D, 2)
    Y = f_lin(x_dash, A)

    # The mean of the marginalised likelihood that we want to maximise
    mu = np.zeros(D)
    # The covariance matrix of the marginalised likelihood
    C = A.dot(A.T) + ((0.2 ** 2) * np.eye(len(A)))

    # We are optimising the parameters, i.e. W, so we pick a random starting
    # point for the gradient descent
    W0 = np.random.randn(D, 2)

    # Run the optimisation
    W_prime = opt.fmin_cg(L, W0,
                          fprime=dLdW,
                          args=(Y, N, D, mu, C))

    # Reshape the result back to the right shape
    W_prime = np.array(W_prime).reshape(D, 2)

    # Map back to 2D space using the learned parameters
    z1 = f_lin_reverse(Y, W_prime)

    # Draw a random 2D mapping (Q22)
    z2 = f_lin_reverse(Y, np.random.randn(D, 2))

    plt.plot(x_dash[0], x_dash[1])
    plt.gca().set_aspect('equal')
    plt.savefig('report/q21_a.png')
    plt.show()

    plt.gca().set_aspect('equal')
    plt.plot(z1[0], z1[1])
    plt.savefig('report/q21_b.png')
    plt.show()

    plt.gca().set_aspect('equal')
    plt.plot(z2[0], z2[1])
    plt.savefig('report/q22.png')
    plt.show()


def L(W, *args):
    """Return the value of the objective at W"""
    # the objective function is a scalar function as it is a probability
    # measure, it is p(y|w) and you know what y is and your optimiser will
    # specify what w is. i.e. L(w) \in \mathbb{R}

    Y, N, D, mu, C = args
    W = W.reshape(D, 2)

    # See summary document 7.2 and also Bishop p574
    term_1 = (N * D / 2) * np.log(2 * np.pi)
    term_2 = (N / 2) * (np.log(np.linalg.det(C)))
    term_3 = np.trace(Y.dot(inv(C)).dot(Y.T))

    return -term_1 - term_2 - (1 / 2) * term_3


def dLdW(W, *args):
    """Return the gradient of the objective at W"""
    # The dfx function should return a derivative of each of the scalars in w,
    # one for each w_{ij}. So you have the objective function L which is a
    # scalar and you take the derivative of a scalar with a scalar you get a
    # scalar. Now you have 10x2 scalars that you want the derivative with
    # respect to. For the opimiser to work you have to return a vector from the
    # derivative, you can do this by simply collecting all the W in a vector by
    # returning W.flatten(). Now in the objective function the optimiser will
    # pass W as a vector, so in order to calculate this you have to reshape it
    # into the form it needs to be to compute L, you can do this by
    # W = W.reshape(10,2).

    Y, N, D, mu, C = args
    W = W.reshape(D, 2)

    result = np.zeros((D, 2))

    for j in range(0, W.shape[0]):
        for i in range(0, W.shape[1]):
            # think about these matrices in terms of the output dimensionality,
            # you take the derivative. you take the derivative of WW^T which we
            # have stated is 10x10 by a scalar, now the final derivative of a
            # matrix with a scalar is the same size as the matrix. So the output
            # should be 10x10. So for these products to work out you will need
            # [10x2][2x10] + [10x2][2x10]
            J_ij = np.zeros((2, D))
            J_ji = np.zeros((D, 2))
            J_ij[i, j] = 1
            J_ji[j, i] = 1

            dCdW = W.dot(J_ij) + J_ji.dot(W.T)

            term_2 = np.trace(inv(C).dot(dCdW))
            term_3 = np.trace(Y.T.dot(Y) * -inv(C).dot(dCdW * (inv(C))))

            result[j, i] = term_2 + term_3

    return result.flatten()


def f_non_lin(xi):
    return np.array([xi * np.sin(xi), xi * np.cos(xi)])


def f_lin(x, A):
    return x.T.dot(A.T)
    # return A.dot(x)


def f_lin_reverse(Y, W_prime):
    return W_prime.T.dot(Y.T)


if __name__ == '__main__':
    main()
