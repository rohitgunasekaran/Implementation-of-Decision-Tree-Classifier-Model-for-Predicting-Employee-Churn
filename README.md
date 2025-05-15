# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import Libraries and Load Dataset.
2. Preprocess the Data.
3. Split the Dataset.
4. Train the Decision Tree Classifier.
5. Make Predictions and Evaluate the Model

## Program:
```
Developed by:ROHIT G
RegisterNumber:  212222240083

```
~~~
import pandas as pd
data=pd.read_csv("Employee.csv")
data
~~~
## Output:
![image](https://github.com/user-attachments/assets/6f475207-c462-45a8-878e-7195adf4ea84)

~~~
data["left"].value_counts()
~~~
## Output:
![image](https://github.com/user-attachments/assets/3b93c134-c3e7-4f90-8749-20e9faa21c90)

~~~
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
~~~
data.head()

## Output:
![image](https://github.com/user-attachments/assets/2b6e8188-cdc4-4d81-bc8c-cda6e4c1edcb)

~~~
data["salary"]=le.fit_transform(data["salary"])
data
~~~
## Output:
![image](https://github.com/user-attachments/assets/4030bd82-eef4-460f-8fb2-749127c196de)

~~~
x=data[["satisfaction_level","last_evaluation","number_project","time_spend_company"]]
x.head()
~~~
## Output:
![image](https://github.com/user-attachments/assets/0dabfd52-bcf5-45db-b36f-60cb19e28ef2)

~~~
y=data["left"]
~~~
~~~
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
~~~
~~~
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
~~~
~~~
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
~~~
## Output:
![image](https://github.com/user-attachments/assets/d766fc19-b108-472a-8d56-04976fe61c17)
~~~
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
~~~
## Output:
![image](https://github.com/user-attachments/assets/57c1c903-7ae4-45f5-9cf8-4b2c2f341d4a)

