import sklearn as sk
import scipy as sp
from functools import partial
import numpy as np


def rounder(samples, thresholds=[.5, 1.5, 2.5, 3.5]):
    """
    :param samples: the samples to quantize
    :param thresholds: Thresholds to quantize in
    :return: quantize samples.
    for example: samples = [1.2, 1.6, 2.1 3.2], thresholds = [1.3, 2.05, 3.1]
    output = [1, 2, 3, 4]
    """
    return(np.digitize(samples, sorted(thresholds)))

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = rounder(X, coef)
        return -1 * sk.metrics.cohen_kappa_score(y, X_p)

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X):
        return rounder(X, self.coef_['x'])

    def coefficients(self):
        return self.coef_['x']

def smooth_rounder(x, r=15):
    """
    This function is a simulatur of a smooth step function.
    This function incrases sharply from 0 to 1 around 0.5, from 1 to 2 around 1.5 and so on.

    :param r: determines how sharp the step function is
    :return: triple of f(x), f`(x), f``(x) where f is the smooth-step-function itself.
    """
    # used = np.where(x < 0, 0 , x)
    c = np.exp(0.5 * r)
    fractional, integral = np.modf(x)
    erx = np.exp(r * fractional)
    return (integral + erx / (c + erx),
            c * r * erx / np.square(c + erx))
