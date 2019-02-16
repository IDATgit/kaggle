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


def kappa_loss(preds, train_data):
    return "kappa", sk.metrics.cohen_kappa_score(train_data.get_label(), quantizer(preds), weights="quadratic")

def l2_rounded_loss(preds, train_data):
    return "l2-rounded", sk.metrics.mean_squared_error(preds, train_data.get_label())

def xg_regress(X, Y):
    data = xg.DMatrix(data=X, label=Y)
    params = {
        "eta" : 0.05,
        "max_depth" : 6,
        "min_child_weight" : 1,
        "subsample" : 0.8
    }
    return xg.cv(params, data,
          num_boost_round=100000,
          nfold=3,
          obj=ranked_classes_obj, feval=kappa_loss, maximize=True)