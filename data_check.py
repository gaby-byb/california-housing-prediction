from fetch_data import load_housing_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import train_test_split


housing = load_housing_data()

print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist(bins=50, figsize=(20,15))
plt.show()

#Create a test set*.
#set the random generator seed to 42
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#ensures set is representitive of the data - creates 5 categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
                               
housing["income_cat"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

#split data while keeping proportions of income_cat
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#shows the proportions of each category inside the test set
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

#drop income_cat to clean the data
#correlation matrix, feature engineering, and model training only see the real features
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#Plotting
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")

#More plotting with color and size 

# Looking for correlation
housing.plot(kind="scatter",
             x= "longitude",
             y="latitude",
             alpha=0.4, #transparency
             s=housing["population"]/100, #point size ~ population
             label="population", figsize=(10,7),
             c="median_house_value", #figure size
             cmap=plt.get_cmap("jet"), #color map (blue -> red spectrum)
             colorbar=True, #legend
             )
plt.legend()

#skip "ocean_proximity" because its a non-int number
corr_matrix = housing.corr(numeric_only=True)

corr_matrix["median_house_value"].sort_values(ascending=False)