import pandas as pd
import matplotlib.pyplot as plt
import models

full_data = pd.read_csv("../input/train/train.csv")


Y = full_data.AdoptionSpeed
X = full_data.drop(columns=["AdoptionSpeed", "RescuerID", "PetID", "VideoAmt"])
print(X.columns)
X_base = X.drop(columns=["Name", "Description"])

res = models.regress_cv(X_base, Y)

print(res)
plt.plot(res['test-kappa-mean'], label='test-kappa-mean')
plt.plot(res['train-kappa-mean'], label='train-kappa-mean')
plt.legend()
plt.show()