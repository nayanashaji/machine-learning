'''
Implement a Naïve Bayes classifier to categorize text documents into topics using the 20 
Newsgroups dataset. Compare the performance of Multinomial Naïve Bayes with 
Bernoulli Naïve Bayes. 
Tasks: 
● Load and preprocess the 20 Newsgroups dataset. 
● Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers. 
● Evaluate and compare the performance of both models using metrics such as 
accuracy and F1-score. 
● Discuss the strengths and weaknesses of each Naïve Bayes variant for text 
classification.
'''

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

data=load_breast_cancer()

df=pd.DataFrame(data.data, columns=data.feature_names)

df["op"]=data.target

X=df[data.feature_names]
y=df["op"]

X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.2
)

binarizer=Binarizer()

X_train_b=binarizer.fit_transform(X_train)
X_test_b=binarizer.transform(X_test)

model=MultinomialNB()
model.fit(X_train_b,y_train)
pred=model.predict(X_test_b)

print("Multinomial Accuracy Score: ",accuracy_score(y_test,pred))

model=BernoulliNB()
model.fit(X_train_b,y_train)
pred=model.predict(X_test_b)

print("Bernoulli Accuracy Score: ",accuracy_score(y_test,pred))
