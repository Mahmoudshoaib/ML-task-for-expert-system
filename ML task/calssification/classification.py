
"""Data contains
 * age - age in years
 * sex - (1 = male; 0 = female)
 * cp - chest pain type
 * trestbps - resting blood pressure (in mm Hg on admission to the hospital)
 * chol - serum cholestoral in mg/dl
 * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
 * restecg - resting electrocardiographic results
 * thalach - maximum heart rate achieved
 * exang - exercise induced angina (1 = yes; 0 = no)
 * oldpeak - ST depression induced by exercise relative to rest
 * slope - the slope of the peak exercise ST segment
 * ca - number of major vessels (0-3) colored by flourosopy
 * thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
 * target - have disease or not (1=yes, 0=no)"""
## Reading Dataset
import pandas as pd
dataSet = pd.read_csv("heart.csv")

## Data Exploration
print(dataSet.target.value_counts())
print("-------------------------")
countNoDisease = len(dataSet[dataSet.target == 0])
countHaveDisease = len(dataSet[dataSet.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(dataSet.target)) * 100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(dataSet.target)) * 100)))
print("-------------------------")
countFemale = len(dataSet[dataSet.sex == 0])
countMale = len(dataSet[dataSet.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(dataSet.sex)) * 100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(dataSet.sex)) * 100)))
print("-------------------------")

## divide the data into features and test
heart_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                  "ca", "thal"]
X = dataSet[heart_features]
heart_output = ["target"]
y = dataSet[heart_output]

## Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

## handlling the missing values
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
# returning columns
imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns

## Logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(imputed_X_train, y_train.values.ravel())
y_predict = logistic_model.predict(imputed_X_test)
logistic_error = mean_absolute_error(y_test, y_predict)
print(f"logistic_error: {logistic_error}")
print("-------------------------")

## KNN model

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# feature scaling
X_scale = StandardScaler()
imputed_X_train = X_scale.fit_transform(imputed_X_train)
imputed_X_test = X_scale.transform(imputed_X_test)

## matrix confusion for KNN

KNN_model = KNeighborsClassifier(n_neighbors=15, p=2, metric='euclidean')
KNN_model.fit(imputed_X_train, y_train.values.ravel())
y_predict_KNN = KNN_model.predict(imputed_X_test)
cm = confusion_matrix(y_test, y_predict_KNN)
print(cm)
print("-------------------------")

## f1 score and accuracy for logistic and KNN
print(f"the accuracy for KNN is {accuracy_score(y_test, y_predict_KNN)} ")

print(f"the accuracy for logistic is {accuracy_score(y_test,y_predict)} ")
