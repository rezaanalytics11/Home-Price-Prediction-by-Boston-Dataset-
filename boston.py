import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
df=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\HousingData.csv')
x_train=df.RM
y_train=df.MEDV
x_train=np.array(x_train)
y_train=np.array(y_train)
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
print(x_train.shape)
print(y_train.shape)
N=x_train.shape[0]

w=np.random.rand(1,1)
Errors=[]
fig,(ax1,ax2)=plt.subplots(1,2)
learning_rate=0.05
epochs=4

for epoch in range(epochs):
 for i in range(N):
    y_pred=np.matmul(x_train[i],w)


    e=y_train[i]-y_pred

    print(e)

    w=w+e*learning_rate*x_train[i]

    Y_pred=np.matmul(x_train,w)
    ax1.clear()
    ax1.scatter(x_train,y_train,color='r')
    ax1.plot(x_train,Y_pred,linewidth=4)


    Error= np.mean(y_train-y_pred)
    Errors.append(Error)
    ax2.clear()
    ax2.plot(Errors)
    plt.pause(0.01)

plt.show()
print(w)

def predict(y_test,x_test,w):
    yy_predict=w*x_test
    print(yy_predict)
    print((y_test-yy_predict)/y_test)

