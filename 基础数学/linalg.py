########################################################################

# Describe what problem can be solved by the code defined here.
# Inform what functions and classes are included in this file.

# Author : Guo Jun-Lin
# E-mail : guojl19@tsinghua.edu.cn
# Date   : 2019 / 11 / 17

########################################################################

import numpy as np

# set up some parameters
# c = 10

########################################################################


def conj_T(mat2D):
    return np.conj(mat2D).T


def symmetrixIO(mat2D):
    return np.sum(mat2D != mat2D.T) == 0


def skew_symmetrixIO(mat2D):
    return np.sum(-mat2D != mat2D.T) == 0


def hermitianIO(mat2D):
    return np.sum(mat2D != np.conj(mat2D).T) == 0


def minor(mat2D, row=1, col=1):
    mat2D = np.delete(mat2D, row, axis=0)
    mat2D = np.delete(mat2D, col, axis=1)
    return mat2D


def cofactor(mat2D, row, col):
    val = np.linalg.det(minor(mat2D, row, col))
    return ((-1) ** (row + col)) * val


def determinant(mat):
    val = 0
    for i in range(mat.shape[0]):
        val += mat[i, 0] * cofactor(mat, i, 0)

    # for j in range(mat.shape[1]):
    #     val += mat[0, j] * cofactor(mat, 0, j)

    return val


def adjoint(mat):
    adj = np.zeros(mat.shape)
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            adj[row, col] = cofactor(mat, row, col)
    return adj.T


def inverse(mat):
    return adjoint(mat) / determinant(mat)


def orthogonalIO(mat, decimal=4):
    # A^T = A^(-1)
    cond1 = np.sum(np.round(mat.T, decimal) !=
                   np.round(inverse(mat), decimal)) == 0
    # A^T·A = A·A^T
    cond2 = np.sum(np.round(np.dot(mat.T, mat), decimal) !=
                   np.round(np.dot(mat, mat.T), decimal)) == 0
    # |A| = +-1
    cond3 = np.round(np.abs(determinant(mat)), decimal) == 1
    return np.sum([cond1, cond2, cond3]) == 3


def singularIO(mat):
    val, vec = np.linalg.eig(mat)
    return np.sum(val == 0) > 0


def variance(data):
    """ Return the variance of the features in dataset X """
    mean = np.ones(data.shape) * np.mean(data, axis=0)
    n_samples = data.shape[0]
    return (1 / n_samples) * np.diag((data - mean).T.dot(data - mean))


def standard_deviation(data):
    return np.sqrt(variance(data))


def coveriance(std, mean):
    # The result of this function is the same as "np.cov(std.T)".

    ans = np.dot((std - mean).T, (std - mean))
    return ans / (std.shape[0] - 1)


########################################################################

if __name__ == '__main__':
    # run the code here
    A = np.array([[2, -2, 1],
                  [1, 2, 2],
                  [2, 1, -2]])
    A = (1 / 3) * A
    print(orthogonalIO(A, decimal=3))

    B = np.array([[-0.23939017, 0.58743526, -0.77305379],
                  [0.81921268, -0.30515101, -0.48556508],
                  [-0.52113619, -0.74953498, -0.40818426]])

    print(orthogonalIO(B, decimal=3))

    C = np.array([[1, 1],
                  [1, -1]])
    C = (1 / np.sqrt(2)) * C
    print(orthogonalIO(C, decimal=3))
