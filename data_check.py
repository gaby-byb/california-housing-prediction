from fetch_data import load_housing_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

housing = load_housing_data()

print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist(bins=50, figsize=(20,15))
plt.show()

#Create a test set*.
from sklearn.model_selection import train_test_split
#set the random generator seed to 42
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#ensures set is representitive of the data - creates 5 categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
                               
housing["income_cat"].hist()

