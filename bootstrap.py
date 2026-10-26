"""

Implement bootstrapping and cross-validation on the Iris dataset. Compare the model
performance metrics (e.g., accuracy, F1-score) obtained using these resampling methods.
Discuss the advantages and disadvantages of each method.
Tasks:
● Load and preprocess the Iris dataset.
● Implement bootstrapping to generate multiple samples and evaluate the model.
● Implement k-fold cross-validation and evaluate the model.
● Compare the performance metrics and discuss the pros and cons of each
resampling method.

"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score

data=load_iris()

df=pd.DataFrame(data.data,columns=data.feature_names)

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Bootstrap

accuracies=[]
f1_scores=[]

for i in range(10):
    X_boot,y_boot=resample(
        X_train,
        y_train,
        random_state=i
    )

    model=DecisionTreeClassifier()
    model.fit(X_boot,y_boot)
    pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,pred)
    f1score=f1_score(y_test,pred,average='weighted')
    accuracies.append(accuracy)
    f1_scores.append(f1score)

print("\nBootstrapping")
print("Accuracy:", sum(accuracies) / len(accuracies))
print("F1 Score:", sum(f1_scores) / len(f1_scores))

# K-Fold Cross Validation

model=DecisionTreeClassifier()
accuracy=cross_val_score(model,X,y,cv=5,scoring="accuracy")
f1score=cross_val_score(model,X,y,cv=5,scoring="f1_weighted")

print("\n5-Fold Cross Validation")
print("Accuracy:", accuracy.mean())
print("F1 Score:", f1score.mean())

# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.utils import resample
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, f1_score

# # Load Iris dataset
# data = load_iris()

# df = pd.DataFrame(data.data, columns=data.feature_names)
# df["target"] = data.target

# print(df.head())

# # Features and target
# X = df[data.feature_names]
# y = df["target"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ---------------- BOOTSTRAPPING ----------------

# accuracies = []
# f1_scores = []

# for i in range(10):

#     # Create bootstrap sample
#     X_boot, y_boot = resample(
#         X_train, y_train,
#         replace=True,
#         random_state=i
#     )

#     model = DecisionTreeClassifier(random_state=42)
#     model.fit(X_boot, y_boot)

#     pred = model.predict(X_test)

#     accuracies.append(accuracy_score(y_test, pred))
#     f1_scores.append(f1_score(y_test, pred, average="weighted"))

# print("\nBootstrapping")
# print("Accuracy:", sum(accuracies) / len(accuracies))
# print("F1 Score:", sum(f1_scores) / len(f1_scores))

# # ---------------- K-FOLD CROSS VALIDATION ----------------

# model = DecisionTreeClassifier(random_state=42)

# accuracy_scores = cross_val_score(
#     model, X, y, cv=5, scoring="accuracy"
# )

# f1_scores = cross_val_score(
#     model, X, y, cv=5, scoring="f1_weighted"
# )

# print("\n5-Fold Cross Validation")
# print("Accuracy:", accuracy_scores.mean())
# print("F1 Score:", f1_scores.mean())