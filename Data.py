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
        self.df = self.df.ix[:,:255]

    def plotGraphs(self, title, classNumber):
        plt.title(title)
        for i in self.df.index:
            if self.df_labels.loc[i,0] == classNumber:
                plt.plot(self.df.columns,self.df.loc[i,:].values)
        plt.ylabel("Values")
        plt.xlabel("indices")
        plt.show()

    def visulalise(self):
        self.plotGraphs("Binary classification for book class",0)
        self.plotGraphs("Binary classification for plastic case class",1)


    def plotCorrelation(self):
        array = []
        corr = self.df.corr()
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if (corr[i][j] >= 0.90):
                    #print(i,"  ",j)
                    array.append(i)
        array = list(set(array))
        print(array)
        #self.df = self.df.drop(i,1)
        #print(corr)
        for i in array:
            self.df = self.df.drop(i,1)
        print("Number of features:", len(self.df.columns))