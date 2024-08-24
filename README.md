# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KAAMESH M
RegisterNumber:  212223040080
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
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
df.head()

![Screenshot 2024-08-24 232324](https://github.com/user-attachments/assets/4b4b918f-eb04-40dc-a7d8-fcfe4c1af17e)

df.tail()

![Screenshot 2024-08-24 232434](https://github.com/user-attachments/assets/f8bc1007-e362-44ca-9f1e-ba96a08a6c8e)

Array value of X

![Screenshot 2024-08-24 232508](https://github.com/user-attachments/assets/3a81a55b-7e65-43a2-9207-1be77cec1320)

Array value of Y

![Screenshot 2024-08-24 232548](https://github.com/user-attachments/assets/c070f702-b2a9-4910-a7a2-7e6a2463392d)

Values of Y prediction

![Screenshot 2024-08-24 232625](https://github.com/user-attachments/assets/62a51c5a-1cac-43cb-8790-933d441e98ae)

Array values of Y test

![Screenshot 2024-08-24 232712](https://github.com/user-attachments/assets/5bf2d558-718f-40af-804d-4b927049324c)

Training Set Graph

![Screenshot 2024-08-24 232756](https://github.com/user-attachments/assets/8c0d665b-85b2-4845-bec5-a296fe4fe33c)

Test Set Graph

![Screenshot 2024-08-24 232831](https://github.com/user-attachments/assets/08c6c744-2eb8-443a-b0e5-db4c9cb54a48)

Values of MSE, MAE and RMSE

![Screenshot 2024-08-24 232901](https://github.com/user-attachments/assets/77087f57-3efa-4787-8e3a-e9d91a7b5fed)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
