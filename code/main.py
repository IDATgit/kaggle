import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.listdir("../input"))
print(os.listdir("../input/train"))
full_data = pd.read_csv("../input/train/train.csv")