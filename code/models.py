import lightgbm as lgbm
import numpy as np
import sklearn as sk
import xgboost as xg
from functools import partial
from rounding import rounder, OptimizedRounder, smooth_rounder

def hopeful_obj(preds, train_data, alpha=0.1, r=15):
    y_hat = preds
    f_y_hat, df_y_hat = smooth_rounder(y_hat, r)
    y = train_data.get_label()
    N = len(y)

    vals, occ = np.unique(y, return_counts=True)
    h = dict(zip(vals, occ))
    h = np.array([h[0], h[1], h[2], h[3], h[4]])

    F = np.sum(np.square(f_y_hat - y))
    G = np.sum(np.square(f_y_hat - 0) * h[0] +
               np.square(f_y_hat - 1) * h[1] +
               np.square(f_y_hat - 2) * h[2] +
               np.square(f_y_hat - 3) * h[3] +
               np.square(f_y_hat - 4) * h[4])

    c = h[1] + 2 * h[2] + 3 * h[3] + 4 * h[4]

    G_der = (N * f_y_hat - c) * df_y_hat
    nom = (f_y_hat - y) * df_y_hat * G - F * G_der
    grad = 2 * N * nom / (np.square(G))

    grad, hess = np.sign(grad) * np.power(np.abs(grad), alpha), np.array([1] * N)

    return grad, hess

def kappa_loss(preds, train_data):
    return "kappa", sk.metrics.cohen_kappa_score(train_data.get_label(), rounder(preds), weights="quadratic")

XG_REGRESS_DEFAULT_PARAMS = {
        "eta": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "lambda": 1,
        "gamma": 1,
        "silent": 1,
        # "verbosity": 0
    }
def regress_cv(X, Y, alpha=0.08, r=15, max_depth=7s, subsample=0.6, gamma=1, eta=0.05, num_iters=1500):
    data = xg.DMatrix(data=X, label=Y)

    sub_obj = partial(hopeful_obj, alpha=alpha, r=r)
    params = XG_REGRESS_DEFAULT_PARAMS
    params["max_depth"] = max_depth
    params["subsample"] = subsample
    params["gamma"] = gamma
    params["eta"] = eta
    return xg.cv(params, data,
                 num_boost_round=num_iters,
                 nfold=4,
                 obj=sub_obj,
                 feval=kappa_loss,
                 maximize=True)
