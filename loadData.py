import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from Data import Data
#import Data from Data

data = Data()
#
#df_labels = df_labels.drop(df_labels.index[:275],1)
print(data.df)
print("----------------------")
print(data.df.head())

 
# import seaborn as sns
# corr = df.corr()
# sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(data.df, data.df_labels, test_size=0.3, random_state=4)

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
logreg.fit(X_train,y_train)
print(logreg.score(X_test,y_test))
#-----------------------NeuralNets-------------------------
mlp = MLPClassifier()
mlp.fit(X_train,y_train)
print(mlp.score(X_test,y_test))

data.visulalise()