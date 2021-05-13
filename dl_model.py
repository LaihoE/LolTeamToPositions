from numpy import loadtxt
import pandas as pd
import sklearn
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pickle
import numpy as np
pd.options.display.width = 0


"""df=pd.read_csv("final_df.csv")

print(df)
indexes=[]
chonker=[]
for x in range(len(df)):
    target=df.iloc[x][-1]
    row=df.iloc[x].tolist()

    inx=row.index(target)
    print(row, target,inx)
    indexes.append(inx)
    if inx==1:
        guess=np.array([1, 0, 0, 0, 0])
    if inx==2:
        guess=np.array([0,1,0,0,0])
    if inx==3:
        guess=np.array([0,0,1,0,0])
    if inx==4:
        guess=np.array([0,0,0,1,0])
    if inx==5:
        guess=np.array([0,0,0,0,1])
    print(guess)
    chonker.append(guess)

print(chonker)
chonker=np.asarray(chonker)
np.save('y.npz', chonker)

print("before")
df = df.drop(df.columns[0],axis=1)
print(df)
df=df.drop('true_bplayer0',axis=1)
df["inx"]=indexes
df.to_csv("modeldata.csv")
np.save('y.npz', chonker)"""

df=pd.read_csv("modeldata.csv",index_col=0)
y=np.load("y.npz.npy")

X = df.drop('inx', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import keras
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#X=X.drop(X.columns[0])
print(X)
model = Sequential()
model.add(Dense(10, activation='relu',input_shape=(None,10)))
model.add(Dense(9, activation='relu'))
model.add(Dense(8,activation='sigmoid'))
model.add(Dense(7, activation='relu'))

model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


X_train=np.asarray(X_train)
y_train=np.asarray(y_train)
X_test=np.asarray(X_test)
y_test=np.asarray(y_test)


model.fit(model.fit(X, y, epochs=100, batch_size=1))



"""cb_modelc = CatBoostClassifier(iterations=1000,
                             depth=6,
                             task_type="GPU",
                             eval_metric='Accuracy',
                             random_seed=42,
                             bagging_temperature=0.2,
                             od_type='Iter',
                             metric_period=50,
                             od_wait=10,
                             learning_rate=0.555
                             )
cb_modelc.fit(X_train, y_train,verbose=50)


with open("C:/Users/emill/PycharmProjects/CLIP/cb_model.model", "w+b") as f:
    pickle.dump(cb_modelc, f)"""d