# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AMEESHA JEFFI J
RegisterNumber:  212223220007
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset

![dataset](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/3d8acaef-514f-420b-bd2d-2a64c3516e8b)

Head values

![head](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/90089f61-1837-41fc-8e5b-5034f5c115b8)

Tail Values

![tail](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/be681286-b1a1-40d2-8294-e38cf2eba400)

X and Y values

![xyvalues](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/af2109f3-3cf7-4ae7-a44e-045697e107d2)

Predication values of X and Y

![predict ](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/80d9e272-c340-43bb-86dd-7b8b6535d106)

MSE,MAE and RMSE

![values](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/7dba03ba-f8ef-4794-9dfa-73a653c75323)

Training Set

![train](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/3f58e8df-1b40-4809-a890-1008f6589344)

Testing Set

![test](https://github.com/ameeshajeffi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150773598/a1be3288-5ffb-44ce-a51c-ac23be78fe24)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
