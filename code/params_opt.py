import pandas as pd
from models import regress_cv
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

full_data = pd.read_csv("../input/train/train.csv")


Y = full_data.AdoptionSpeed
X = full_data.drop(columns=["AdoptionSpeed", "RescuerID", "PetID", "VideoAmt"])
X_base = X.drop(columns=["Name", "Description"])


def hyper_params_grade(alpha, r, max_depth, subsample, gamma):
    max_depth = int(max_depth)
    learn_res = regress_cv(X_base, Y, alpha, r, max_depth, subsample, gamma)
    return np.max((learn_res["test-kappa-mean"] - learn_res["test-kappa-std"]))

pbounds={
    "alpha": (0, 0.1),
    "r": (10, 30),
    "max_depth": (0, 15),
    "subsample": (0.5, 1),
    "gamma": (0, 5)
}

optimizer = BayesianOptimization(f=hyper_params_grade, pbounds=pbounds, random_state=154)
optimizer.probe(
    params={"alpha": 0.1, "r": 15, "max_depth": 7, "subsample": 0.8, "gamma": 1},
    lazy=True,
)

logger = JSONLogger(path="./hyper-parameters-optimization-logs.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
optimizer.maximize(init_points=5, n_iter=20)