import numpy as np
import sklearn as sk
import xgboost as xg
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from models import hopeful_obj, kappa_loss, XG_REGRESS_DEFAULT_PARAMS, OptimizedRounder
from functools import partial


def optimized_kappa_loss(preds, train_data):

    return "kappa", sk.metrics.cohen_kappa_score(train_data.get_label(), quantizer(preds), weights="quadratic")#, True



full_data = pd.read_csv("../input/train/train.csv")


Y = full_data.AdoptionSpeed
X = full_data.drop(columns=["AdoptionSpeed", "RescuerID", "PetID", "VideoAmt"])
print(X.columns)
X = X.drop(columns=["Name", "Description"])

fold = 5
inds = np.arange(len(X))
np.random.shuffle(inds)
train_inds = inds[:int((fold - 1) * len(inds) / fold)]
test_inds = inds[int((fold - 1) * len(inds) / fold):]

X_train = X.iloc[train_inds]
Y_train = Y.iloc[train_inds]

X_test = X.iloc[test_inds]
Y_test = Y.iloc[test_inds]

train_data_dogs = xg.DMatrix(data=X_train[X_train["Type"] == 1], label=Y_train[X_train["Type"] == 1])
train_data_cats = xg.DMatrix(data=X_train[X_train["Type"] == 2], label=Y_train[X_train["Type"] == 2])

test_data_dogs = xg.DMatrix(data=X_test[X_test["Type"] == 1], label=Y_test[X_test["Type"] == 1])
test_data_cats = xg.DMatrix(data=X_test[X_test["Type"] == 2], label=Y_test[X_test["Type"] == 2])

train_data = xg.DMatrix(data=X_train, label=Y_train)
test_data = xg.DMatrix(data=X_test, label=Y_test)


XG_REGRESS_PARAMS = {
        "eta": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "lambda": 1,
        "gamma": 1,
        "verbosity": 0,
        "eval_metric": "rmse"
    }
sub_obj = partial(hopeful_obj, alpha=0.07)#, r=15, max_depth=7, subsample=0.8, gamma=1)
booster = xg.train(XG_REGRESS_PARAMS,
                   train_data,
                   num_boost_round=2000,
                   obj=sub_obj,
                   feval=kappa_loss,
                   maximize=True)

optr = OptimizedRounder()

predictions = booster.predict(test_data)
predictions_train = booster.predict(train_data)
optr.fit(predictions_train, train_data.get_label())
predictions_rounded = optr.predict(predictions)
print("optimized rounding loss is ", kappa_loss(predictions_rounded, test_data))
print("regular rounding loss is ", kappa_loss(predictions, test_data))


#
# booster_dogs = xg.train(XG_REGRESS_PARAMS,
#                    train_data_dogs,
#                    num_boost_round=1000,
#                    obj=hopeful_obj,
#                    feval=kappa_loss,
#                    maximize=True)
#
# booster_cats = xg.train(XG_REGRESS_PARAMS,
#                    train_data_cats,
#                    num_boost_round=1000,
#                    obj=hopeful_obj,
#                    feval=kappa_loss,
#                    maximize=True)

