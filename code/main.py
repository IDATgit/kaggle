import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
import models
from quadratic_kappa import quadratic_weighted_kappa

full_data = pd.read_csv("../input/train/train.csv")


Y = full_data.AdoptionSpeed
X = full_data.drop(columns=["AdoptionSpeed", "RescuerID", "PetID", "VideoAmt"])
print(X.columns)
X_base = X.drop(columns=["Name", "Description"])

res = models.xg_regress(X_base, Y)

print(res)
plt.plot(res['test-kappa-mean'], label='test-kappa-mean')
plt.plot(res['train-kappa-mean'], label='train-kappa-mean')
plt.show()

#
# fig, axes = plt.subplots(2)
#
# axes[0].plot(range(len(model_history["kappa-mean"])), model_history["kappa-mean"], label="kappa-mean")
# # axes[0].plot(range(len(model_history["l2-mean"])), model_history["l2-mean"], label="l2-mean")
# axes[1].plot(range(len(model_history["kappa-stdv"])), model_history["kappa-stdv"], label="kappa-stdv")
# # axes[1].plot(range(len(model_history["l2-stdv"])), model_history["l2-stdv"], label="l2-stdv")
#
# axes[0].legend()
# axes[1].legend()
#
# print("The best cv of l2 was received after %d iterations. the result was %f" %
#       (np.argmin(model_history["l2-mean"]), np.min(model_history["l2-mean"])))
# print("The best cv of kappa was received after %d iterations. the result was %f" %
#       (np.argmax(model_history["kappa-mean"]), np.max(model_history["kappa-mean"])))
#
# plt.show()
