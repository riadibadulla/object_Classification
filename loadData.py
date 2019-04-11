import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from Data import Data
import os

os.system('clear')

#import Data from Data

data = Data()
#data.cutFeatures()
print(data.df)
print("----------------------")
print(data.df.head())

data.visulalise()
data.plotCorrelation()
print(data.df)

X_train, X_test, y_train, y_test = train_test_split(data.df, data.df_labels, test_size=0.3, random_state=4)

def logRegression():
    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
    logreg.fit(X_train,y_train)
    print(logreg.score(X_test,y_test))

def neuralNets():
    mlp = MLPClassifier(hidden_layer_sizes=(50,), activation="logistic", verbose=True, max_iter=500, early_stopping=True, validation_fraction=0.3)
    mlp.fit(X_train,y_train)
    print(mlp.score(X_test,y_test))

logRegression()
neuralNets()
