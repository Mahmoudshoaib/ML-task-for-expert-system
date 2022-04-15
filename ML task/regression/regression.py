import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv("houses.csv")
X = ds.iloc[:, 0:17].values
y = ds.iloc[:, 17].values
# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
LRM = LinearRegression()

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Linear Regression to the dataset
LRM = LinearRegression()

# Train Data
LRM.fit(X_train, y_train)

# Prediction.
linear_y_pred = LRM.predict(X_test)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, linear_y_pred)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, linear_y_pred)

from sklearn.metrics import median_absolute_error

median_absolute_error(y_test, linear_y_pred)
