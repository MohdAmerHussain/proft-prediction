import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv(r"F:\DATA SCIENCE\multiple linear regression\assignment\50_startups.csv")
dataset
dataset.describe()
import seaborn as sns
sns.boxplot(dataset["Administration"])
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4:5].values
X
y
X[0]
dataset["State"].unique()
X.shape
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
ct = ColumnTransformer([("oh", OneHotEncoder(),[3])],remainder = "passthrough")
X = ct.fit_transform(X)
X
import joblib
joblib.dump(ct,"column")
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LinearRegression
multilinear = LinearRegression()
multilinear.fit(X_train,y_train)
y_pred = multilinear.predict(X_test)
y_pred
y_test
from sklearn.metrics import r2_score
accuracy = r2_score(y_test,y_pred)
import pickle
pickle.dump(multilinear,open('profit.pkl','wb'))
