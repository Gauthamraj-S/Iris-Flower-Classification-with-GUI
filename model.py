# Importing Packages
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Reading the csv data
iris=pd.read_csv('Iris.csv')

# Setting Feature & Target Variable
X = iris.iloc[:, :-1].values    #   X -> Feature Variables
y = iris.iloc[:, -1].values #   y ->  Target

# Splitting the data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Support Vector Machine
Model = SVC()
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)

# Dumping pickle file
pickle.dump(Model,open('model.pkl','wb'))
