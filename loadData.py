import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("binary/X.csv")
df = df.dropna()
df_labels = pd.read_csv("binary/y.csv")
print(df)
print("----------------------")
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(df, df_labels, test_size=0.33, random_state=42)

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
logreg.fit(X_train,y_train)
print(logreg.score(X_test,y_test))
