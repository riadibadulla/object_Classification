import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from Data import Data

X_train = None
X_test = None
y_train = None
y_test = None

def clearScreen():
    import os
    os.system('clear')

def logRegression():
    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
    logreg.fit(X_train,y_train)
    print(logreg.score(X_test,y_test))

def neuralNets():
    mlp = MLPClassifier(hidden_layer_sizes=(500,), activation="relu", verbose=True, max_iter=500, early_stopping=True, validation_fraction=0.3)
    mlp.fit(X_train,y_train)
    print(mlp.score(X_test,y_test))

def main():
    global X_train, X_test, y_train, y_test
    clearScreen()
    data = Data()
    #data.cleanData()
    #data.applyPCA(2)
    #data.visualiseOn2D()
    #data.visulalise()
    data.cutFeatures()
    data.delteCorrelatedFeatures()
    data.plotHeatMap()
    print(data.df)
    #X_train, X_test, y_train, y_test = train_test_split(data.df, data.df_labels, test_size=0.3, random_state=4)
    #logRegression()
    # neuralNets()

main()
