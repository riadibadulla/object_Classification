import pandas as pd  
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np

class Data:
    df = None
    df_labels = None
    def __init__(self):
        self.df = pd.read_csv("binary/X.csv", header=None)
        self.df_labels = pd.read_csv("binary/y.csv", header=None)

    def cutFeatures(self):
        print("PORCO DIO")
        print(self.df.loc[:,0])
        print(self.df.loc[:,256])
        print(self.df.loc[:,512])
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

    def plotHeatMap(self):
        import seaborn as sns
        corr = self.df.corr()
        sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True)
        plt.title("Correlation matrix")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.show()

        return corr

    def delteCorrelatedFeatures(self):
        array = []
        corr = self.df.corr()
        print(corr)
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if (corr[i][j] >= 0.90):
                    array.append(i)
        array = list(set(array))
        for i in array:
            self.df = self.df.drop(i,1)
        print("Features left:", self.df.columns)

    def cleanData(self):
        for i in self.df.index:
            for j in self.df.columns:
                if (self.df.isna().loc[i,j]==True):
                    self.df.drop(i,0)
                    self.df_labels.drop(i,0)
    
    def applyPCA(self,number):
        pca = decomposition.PCA(n_components=number)
        self.df = pd.DataFrame(pca.fit_transform(self.df))
        print(self.df)
    
    def visualiseOn2D(self):
        if (len(self.df.columns) != 2):
            self.applyPCA(2)

        redElements = []
        blueElements = []
        for index, row in self.df.iterrows():
            if self.df_labels.loc[index,0] == 0:
                redElements.append(row)
            else:
                blueElements.append(row)
        redElements = pd.DataFrame(redElements)
        blueElements = pd.DataFrame(blueElements)
        plt.scatter(redElements.loc[:,0],redElements.loc[:,1], color="red")
        plt.scatter(blueElements.loc[:,0],blueElements.loc[:,1],color="blue")
        plt.grid(b=True)
        plt.title("Feature against feature graph after PCA applied")
        plt.xlabel("Generated Feature 1")
        plt.ylabel("Generated Feature 2")
        plt.show()