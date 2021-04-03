########################################################################

# Describe what problem can be solved by the code defined here.
# Inform what functions and classes are included in this file.

# Author : Kuo Chun-Lin
# E-mail : guojl19@tsinghua.org.cn
# Date   : 2020.12.04 - 2020.12.06

########################################################################

import copy
import numpy as np
from scipy.stats import multivariate_normal as normal

# Matplotlib is imported for visualizing results.
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

########################################################################


def covariance(x, z=None):
    if z is None:
        z = copy.deepcopy(x)
    prod = np.dot((x - np.mean(x, axis=0)).T, (z - np.mean(z, axis=0)))
    return np.asarray(prod / (len(x) - 1), dtype=np.float)


def multivariate_gaussian(x, means, covars):
    ns, nf = x.shape
    det = np.linalg.det(covars)
    inv = np.linalg.pinv(covars)
    coef = 1 / (np.power(2 * np.pi, nf / 2) * np.sqrt(det))

    likelihoods = np.zeros(ns)
    for i in range(ns):
        likelihoods[i] = np.exp(-0.5 * (
            x[i] - means).T.dot(inv).dot(x[i] - means))

    # The reason why we apply "for" loop to compute likelihood iteratively
    # is because matrix means_ltiplication spends too means_ch space relative to time.
    return coef * likelihoods


########################################################################


class GaussianMixtureModel(object):
    def __init__(self, n_components, max_iter=300, tol=1e-4, scipy=True):
        self.nc = n_components
        self.max_iter = max_iter
        self.tol = tol

        # Probability Density Function (pdf). scipy can be faster.
        self.pdf = normal.pdf if scipy else multivariate_gaussian

        self.ns = 0    # num_samples
        self.nf = 0    # num_features
        self.history = []

    def __init_params(self, x):
        self.ns, self.nf = x.shape

        # Assume that the probabilities of each cluster exists are equal.
        self.weights_ = np.ones(self.nc) / self.nc
        self.Q_x_z = np.ones([self.ns, self.nc]) / self.nc

        # Clearify the shape of both "mean" and "covariance" matrix.
        # Params of all Gaussian PDF are saved together here.
        self.means_ = np.zeros([self.nc, self.nf])
        self.covars_ = np.zeros([self.nc, self.nf, self.nf])

        # Randomly assign the value of some samples to "means_".
        self.means_ = x[np.random.choice(range(len(x)), self.nc)]
        # Randomly initialize the diagonal values of covars_.
        for k in range(self.nc):
            self.covars_[k] = covariance(x)
            # np.fill_diagonal(self.covars_[k], np.random.rand(self.nf))

    def __Gaussian_likelihood(self, x):
        Gaussian = np.zeros([self.ns, self.nc])
        for k in range(self.nc):
            Gaussian[:, k] = self.pdf(x, self.means_[k], self.covars_[k])

        return Gaussian

    def __E_step(self, x):    # Expectation.
        # Calc the weighted probabilities of "x" belonging to the diff components.
        q_z = self.weights_ * self.__Gaussian_likelihood(x)
        # Update the probability : Q(x, z) = α_k * N(x_k|μ,Σ) / \sum{α * N(x|μ,Σ)}
        self.Q_x_z = q_z / np.sum(q_z, axis=1).reshape(-1, 1)

    def __M_step(self, x):    # Maximization.
        for k in range(self.nc):
            # Get the probabilities from a cluster one by one.
            Q_x_zk = self.Q_x_z[:, k].reshape(-1, 1)
            # Update "mean" of THAT cluster using the weighted probabilities.
            self.means_[k] = np.sum(Q_x_zk * x, axis=0) / np.sum(Q_x_zk)
            # Update "cov" of THAT cluster using the weighted probabilities.
            self.covars_[k] = (x - self.means_[k]).T.dot((
                x - self.means_[k]) * Q_x_zk) / np.sum(Q_x_zk)

        # Update the original equal weights using the weighted probabilities.
        self.weights_ = np.sum(self.Q_x_z, axis=0) / self.ns

    def fit(self, x):
        history = []
        self.__init_params(x)

        for it in range(self.max_iter):
            self.__E_step(x)    # Update: Q_x_z
            self.__M_step(x)    # Update: means_, covars_, weights_

            # Record the value to check if it meets the convergence criteria.
            history.append(np.max(self.Q_x_z, axis=1))

            if len(history) > 2:
                diff = np.linalg.norm(history[-1] - history[-2])
                if diff <= self.tol:
                    print("Totally {} iters".format(it + 1))
                    break

            if it + 1 == self.max_iter:
                print("Totally {} iters".format(it + 1))

    def predict(self, x):
        # Run E-step again in case the last M-step has updated sth new.
        self.__E_step(x)
        # Assign samples to a cluster that has highest probability.
        return np.argmax(self.Q_x_z, axis=1)

    def score(self, x):
        q_z = self.weights_ * self.__Gaussian_likelihood(x)
        log_likelihood = np.log(np.sum(q_z, axis=1))
        return np.mean(log_likelihood)


########################################################################
# Helper functions.

def illustration():
    x = np.linspace(-3, 2, 500)
    def y(x): return x**3 - x**2 - 4 * x
    z1 = np.linspace(-2.9, -1.1, 500)
    def f1(x): return -20 * (x + 2)**2 - 7
    z2 = np.linspace(-2.6, -0.8, 500)
    def f2(x): return -20 * (x + 1.7)**2 - 2.3

    fig = plt.figure(figsize=(15, 5))
    plt.plot(x, y(x), label=r'$\mathcal{LL}(\bar{\theta})$')
    plt.plot(z1, f1(z1), label='lower bound func 1')
    plt.plot(z2, f2(z2), label='lower bound func 2')
    plt.scatter(-2.5, -11.8, s=100, c='r')
    plt.text(-2.6, -11, "1", fontsize=15)

    plt.plot([-2, -2], [-24, 0], linestyle='dashed', color='r')
    plt.scatter([-2, -2], [-7, -3.8], s=100, c='r')
    plt.text(-1.95, -9.5, "2", fontsize=15)
    plt.text(-1.95, -2, "3", fontsize=15)

    plt.plot([-1.7, -1.7], [-22, 2], linestyle='dashed', color='r')
    plt.scatter([-1.7, -1.7], [-2.3, -0.8], s=100, c='r')
    plt.text(-1.65, 0.5, "5", fontsize=15)
    plt.text(-1.65, -4.8, "4", fontsize=15)

    plt.scatter(-0.9, 2, s=200, c='r', marker='v')
    plt.text(-1.15, -1, "optimal", fontsize=20)

    plt.axis([-3, 2, -25, 5])
    plt.legend(prop={'size': 20})
    plt.grid(True)
    plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # "gca" stands for "get current axis".

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 5):
        ax.add_patch(Ellipse(position, nsig * width,
                             nsig * height, angle,
                             **kwargs))


def visualization(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    gmm.fit(X)

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=gmm.predict(X),
                   edgecolors='black', linewidth=0.5,
                   s=5, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=5, zorder=2,
                   edgecolors='black', linewidth=0.5)

    ax.axis('equal')
    # Setup transparency.
    # w_factor = 0.4 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, fill=False, edgecolor='r', lw=2)

    plt.show()


def data_gen(n1, n2, n3):
    X_1 = np.random.multivariate_normal([0.5, 0.5], [[1, 0], [0, 3]], n1)
    X_2 = np.random.multivariate_normal([5.5, 2.5], [[2, 0], [0, 2]], n2)
    X_3 = np.random.multivariate_normal([1, 7], [[6, 0], [0, 2]], n3)
    return X_1, X_2, X_3


########################################################################


if __name__ == '__main__':
    # run the code here
    import time

    x1, x2, x3 = data_gen(800, 900, 700)
    x = np.vstack([x1, x2, x3])
    y = np.hstack([np.zeros(len(x1)),
                   np.ones(len(x2)),
                   np.ones(len(x3)) + 1])

    t1 = time.time()
    gmm = GaussianMixtureModel(n_components=3, max_iter=50,
                               tol=1e-4, scipy=True)
    gmm.fit(x)
    print("My GMM (sec): {}".format(round(time.time() - t1, 2)))

    pred = gmm.predict(x)
    print("My GMM acc: {}%".format(round(np.sum(pred == y) / len(y), 2)))
    print(gmm.score(x))

    # ------------------------------------------------------------------

    # from sklearn.mixture import GaussianMixture

    # t2 = time.time()
    # skgmm = GaussianMixture(n_components=3, max_iter=100, tol=0.001)
    # skgmm.fit(x)
    # print("sk GMM func (sec): {}".format(round(time.time() - t2, 2)))

    # skpred = skgmm.predict(x)
    # print("sk GMM acc: {}%".format(round(np.sum(skpred == y) / len(y), 2)))
    # # acc: 0.964583
    # print(skgmm.score(x, y))

    # ------------------------------------------------------------------

    visualization(gmm, x)
