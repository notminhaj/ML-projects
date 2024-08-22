# import sklearn
# from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
# import numpy as np

data = pd.read_csv("car.data")


data.loc[data["buying"] == "low", "buying"] = 0
data.loc[data["buying"] == "med", "buying"] = 1
data.loc[data["buying"] == "high", "buying"] = 2
data.loc[data["buying"] == "vhigh", "buying"] = 3

data.loc[data["maint"] == "low", "maint"] = 0
data.loc[data["maint"] == "med", "maint"] = 1
data.loc[data["maint"] == "high", "maint"] = 2
data.loc[data["maint"] == "vhigh", "maint"] = 3

data.loc[data["door"] == "5more", "door"] = 5
data.loc[data["door"] == "more", "door"] = 5

data.loc[data["persons"] == "more", "persons"] = 5

data.loc[data["lug_boot"] == "small", "lug_boot"] = 0
data.loc[data["lug_boot"] == "med", "lug_boot"] = 1
data.loc[data["lug_boot"] == "big", "lug_boot"] = 2

data.loc[data["safety"] == "low", "safety"] = 0
data.loc[data["safety"] == "med", "safety"] = 1
data.loc[data["safety"] == "high", "safety"] = 2

data.loc[data["class"] == "unacc", "class"] = 0
data.loc[data["class"] == "acc", "class"] = 1
data.loc[data["class"] == "good", "class"] = 2
data.loc[data["class"] == "vgood", "class"] = 3

buying = data["buying"].values.astype(int)
maint = data["maint"].values.astype(int)
door = data["door"].values.astype(int)
persons = data["persons"].values.astype(int)
lug_boot = data["lug_boot"].values.astype(int)
safety = data["safety"].values.astype(int)
cls = data["class"].values.astype(int)

'''
le = preprocessing.LabelEncoder()
buying2 = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
'''

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict([[0, 3, 4, 4, 2, 2]])
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ")
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)