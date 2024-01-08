import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
import pickle

class PrepoceesingData():
    def __init__(self,dataset=None, x_test=None, y_test=None,x_train=None, y_train=None):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
       
    def proses(self,dataset):
        self.dataset=pd.read_csv(dataset)
        print(self.dataset)
        

    
    def DataSelection(self):
        x = self.dataset[['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']] #features
        y = self.dataset['categori'] 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, test_size = 0.035, random_state=0)
        print(self.x_train, self.x_test, self.y_train, self.y_test)
        for dataset in [self.x_train, self.x_test]:
            dataset['pm25'].fillna(self.x_train['pm25'].mode()[0], inplace=True)

    
    def MetodeTree(self):
        sc = StandardScaler()
        self.x_test = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)
        clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
        clf_gini.fit(self.x_train, self.y_train)
        pickle.dump(clf_gini,open("modelTreePencemaran.pkl","wb"))
    
    def MetodeKnn(self):
        sc = StandardScaler()
        self.x_test = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)
        knn_6 = KNeighborsClassifier(n_neighbors=6)
        knn_6.fit(self.x_train, self.y_train)
        pickle.dump(knn_6,open("modelKnnPencemaran.pkl","wb"))

    def MetodeNaiveBayes(self):
        sc = StandardScaler()
        self.x_test = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)
        gnb = GaussianNB()
        gnb.fit(self.x_train, self.y_train)
        pickle.dump(gnb,open("modelNBPencemaran.pkl","wb"))
