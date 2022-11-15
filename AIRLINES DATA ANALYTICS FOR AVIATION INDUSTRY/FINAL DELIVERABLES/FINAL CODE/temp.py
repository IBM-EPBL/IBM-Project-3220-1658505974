from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

dt = pd.read_csv(r"C:/Users/Anjana/Downloads/airports.csv")
dt = dt.dropna()

dt=dt.replace('NaN',0)
dt=dt.replace('OC',1)
dt=dt.replace('AF',2)
dt=dt.replace('AN',3)
dt=dt.replace('EU',4)
dt=dt.replace('AS',5)
dt=dt.replace('SA',6)

#feature and target arrays
train=dt['elevation_ft']
target=dt['elevation_ft']
train=np.array(train)
target=np.array(target)

X_train, X_test, y_train, y_test = train_test_split(train,target, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train.reshape(-1,1), y_train)

file = open("model.pkl","wb")

pickle.dump(knn,file)

file.close()
