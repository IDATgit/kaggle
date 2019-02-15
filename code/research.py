import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def breed():
    full_data = pd.read_csv("../input/train/train.csv")
    dogs = full_data.loc[full_data['Type'] == 1]
    cats = full_data.loc[full_data['Type'] == 2]
    plt.close('all')
    plt.figure()
    plt.title('dogs breed1 total = ' + str())
    plt.hist(dogs['Breed1'], bins=range(308))
    plt.figure()
    plt.title('Cats breed1')
    plt.hist(cats['Breed1'], bins=range(308))
    plt.figure()
    plt.title('dogs breed2')
    plt.hist(dogs['Breed2'], bins=range(308))
    plt.figure()
    plt.title('Cats breed2')
    plt.hist(cats['Breed2'], bins=range(308))
    plt.show()






if __name__ == "__main__":
    breed()