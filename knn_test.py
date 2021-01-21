import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


from knn import KNN

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


dataset = pd.read_csv('data1.csv')

label = dataset.iloc[:,-1:].values.ravel()
feature = dataset.iloc[:,:-1]


X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)

# Inspect data

#print(X_train.shape)
#print(X_train[0])

#print(y_train.shape)
#print(y_train)

#plt.figure()
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
#plt.show()

k = 3
clf = KNN(k=k)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("custom KNN classification accuracy", accuracy(y_test, pred))