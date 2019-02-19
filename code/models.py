import lightgbm as lgbm
import numpy as np
import sklearn as sk
import xgboost as xg

loss = "l2-rounded"

def quantizer(samples, thresholds=[.5, 1.5, 2.5, 3.5]):
    """
    :param samples: the samples to quantize
    :param thresholds: Thresholds to quantize in
    :return: quantize samples.
    for example: samples = [1.2, 1.6, 2.1 3.2], thresholds = [1.3, 2.05, 3.1]
    output = [1, 2, 3, 4]
    """
    assert(len(thresholds) == 4)
    return(np.digitize(samples, [0, thresholds[0], 1, thresholds[1],
                                 2, thresholds[2], 3, thresholds[3], 4])//2)


def crazy_rounder(x, r=10):
    """
    This function is a simulatur of a smooth step function.
    This function incrases sharply from 0 to 1 around 0.5, from 1 to 2 around 1.5 and so on.

    :return: triple of f(x), f`(x), f``(x) where f is the smooth-step-function itself.
    """
    # used = np.where(x < 0, 0 , x)
    c = np.exp(0.5 * r)
    fractional, integral = np.modf(x)
    erx = np.exp(r * fractional)
    return (integral + erx / (c + erx),
            c * r * erx / np.square(c + erx),
            c * r * r * erx * (c - erx) / np.power(c + erx, 3))


def ranked_classes_obj(preds, train_data):
    y_hat = np.where(preds < 0, 0, preds)
    y = train_data.get_label()
    f_y_hat, df_y_hat, ddf_y_hat = crazy_rounder(y_hat)
    # print("y_hat", y_hat)
    grad = 2 * (f_y_hat - y) * df_y_hat
    hessian = 2 * (ddf_y_hat * (f_y_hat - y) + np.square(df_y_hat))
    # print("grad", grad)
    # print("hessian", hessian)
    return grad, hessian

def last_hope_obj(preds, train_data):
    y_hat = preds
    y = train_data.get_label()
    N = len(y)

    vals, occ = np.unique(y, return_counts=True)
    h = dict(zip(vals, occ))
    h = np.array([h[0], h[1], h[2], h[3], h[4]])

    F = np.sum(np.square(y_hat - y))
    G = np.sum(np.square(y_hat - 0) * h[0] +
               np.square(y_hat - 1) * h[1] +
               np.square(y_hat - 2) * h[2] +
               np.square(y_hat - 3) * h[3] +
               np.square(y_hat - 4) * h[4])

    c = h[1] + 2 * h[2] + 3 * h[3] + 4 * h[4]

    G_der = N * y_hat - c
    nom = (y_hat - y) * G - F * G_der
    grad = 2 * N * nom / (np.square(G))

    hess = (G + (y_hat - y) * G_der - 2 * (y_hat - y) * G_der - N * F) * G - 2 * G_der * nom
    hess = 2 * N * hess / np.power(G, 3)

    grad, hess = np.sign(grad) * np.power(np.abs(grad), 0.1), np.array([1] * len(grad))
    # print("y", y_hat)
    # print("grad", grad, "\nhess", hess)
    # print("hess sum =", np.sum(hess))
    # print("grad sum =", np.sum(grad))
    return grad, hess

def kappa_loss(preds, train_data):
    return "kappa", sk.metrics.cohen_kappa_score(train_data.get_label(), quantizer(preds), weights="quadratic")#, True

def l2_rounded_loss(preds, train_data):
    return "l2-rounded", sk.metrics.mean_squared_error(preds, train_data.get_label())

def xg_regress(X, Y):
    data = xg.DMatrix(data=X, label=Y)
    params = {
        "eta" : 0.1,
        "max_depth": 7,
        "min_child_weight": 1,
        # "subsample": 0.8,
        "verbosity": 0,
        "lambda": 1,
        "gamma": 1
    }
    return xg.cv(params, data,
                 num_boost_round=500,
                 nfold=3,
                 obj=last_hope_obj,
                 feval=kappa_loss,
                 maximize=True)


CAT_FEATS = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
             'Color3', 'MaturitySize', 'Vaccinated', 'Dewormed',
             'Sterilized', 'Health', 'Quantity', 'State']
# def xg_regress(X, Y, cat_feats=CAT_FEATS):
#     """
#     This function learns lgbm using a rgressor. Should go through a rounder.
#     :return: The cv of the algorithm
#     """
#     data = lgbm.Dataset(X, Y, categorical_feature=cat_feats)
#     params = {
#         "num_leaves" : 128,
#         "bagging_fraction" : 0.8,
#         "bagging_freq" : 2,
#         "feature_fraction" : 0.9,
#         "early_stopping_round" : 10,
#         "learning_rate" : 0.1,
#         # "metric" : "l2",
#         "num_iterations" : 500
#     }
#
#     return lgbm.cv(params, data, categorical_feature=cat_feats,
#                    #metrics="l2",
#                    feval=kappa_loss, fobj=last_hope_obj,
#                    nfold=2)