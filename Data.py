import pandas as pd
import matplotlib.pyplot as plt
class Data:
    df = None
    df_labels = None
    def __init__(self):
        self.df = pd.read_csv("binary/X.csv", header=None)
        self.df = self.df.dropna()
        self.df_labels = pd.read_csv("binary/y.csv", header=None)

    def cutFeatures(self):
        self.df = self.df.ix[:,:275]

    def visulalise(self):
        plt.plot(self.df.columns,self.df.loc[0,:].values)
        plt.show()