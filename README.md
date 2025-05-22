# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the necessary libraries.
Read the dataset and separate the independent and dependent variables.
Split the dataset into training and testing.
Do preprocessing if needed, in this case vectorization is needed which is done using CountVectorizer()
Train the model using SVC() algorithm and .fit()
Predict the model on x_test.
Measure its accuracy

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIJAYASHREE B
RegisterNumber:  212223040238
*/
```

```        
    import pandas as pd
    data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
    data.info()
    
    x=data['v2'].values
    y=data['v1'].values
    x.shape
    y.shape
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer()
    x_train=cv.fit_transform(x_train)
    x_test=cv.transform(x_test)
    x_train
    
    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x_train,y_train)
    y_pred=svc.predict(x_test)
    y_pred
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,y_pred)
    acc
  ```

## Output:

![image](https://github.com/user-attachments/assets/c0bd4fb9-e799-4f14-b235-1d3700eeaf86)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
