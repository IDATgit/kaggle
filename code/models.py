import lightgbm as lgbm
import numpy as np
from quadratic_kappa import quadratic_weighted_kappa

def quantizer(samples, thresholds):
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


def kappa_objective(preds, train_data):
    return "kappa", quadratic_weighted_kappa(train_data.get_label(), quantizer(preds, [.5, 1.5, 2.5, 3.5])), True


CAT_FEATS = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
             'Color3', 'MaturitySize', 'Vaccinated', 'Dewormed',
             'Sterilized', 'Health', 'Quantity', 'State']
def lgbm_regress(X, Y, cat_feats=CAT_FEATS):
    """
    This function learns lgbm using a rgressor. Should go through a rounder.
    :return: The cv of the algorithm
    """
    data = lgbm.Dataset(X, Y, categorical_feature=cat_feats)
    params = {
        "num_leaves" : 128,
        "bagging_fraction" : 0.8,
        "bagging_freq" : 2,
        "feature_fraction" : 0.9,
        #"early_stopping_round" : 10,
        "learning_rate" : 0.01,
        "metric" : "l2",
        "num_iterations" : 5000
    }
    return lgbm.cv(params, data, categorical_feature=cat_feats, feval=kappa_objective, nfold=10)

def lgbm_classify(X, Y, cat_feats=CAT_FEATS):
    """
    This function learns lgbm using a rgressor. Should go through a rounder.
    :return: The cv of the algorithm
    """
    data = lgbm.Dataset(X, Y, categorical_feature=cat_feats)
    params = {
        "objective" : "multiclass",
        "num_class" : 5,
        "num_leaves" : 128,
        "bagging_fraction" : 0.8,
        "bagging_freq" : 2,
        "feature_fraction" : 0.9,
        #"early_stopping_round" : 10,
        "learning_rate" : 0.01,
        "metric" : "l2",
        "num_iterations" : 1000
    }
    return lgbm.cv(params, data, categorical_feature=cat_feats, feval=kappa_objective, nfold=10)
