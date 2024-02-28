import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from autofeat import AutoFeatRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import warnings
warnings.filterwarnings("ignore")
from datasci import *


# import encoded dataset
df = pd.read_csv('CABG_5yr_preselect40.csv')
#df = pd.read_csv('CABG_preselect.csv')
print(df.head())


# select top 10 important features
#df = df[['OTHBLEED','BMI','AGE','RACE_NEW','DIABETES','DYSPNEA','SEX','SMOKE','HYPERMED','BLEEDIS','HXCHF']]

# select top 5 important features
df = df[['OTHBLEED',
         'BMI','AGE','RACE_NEW','DIABETES','DYSPNEA']]


# change data types to avoid errors
print(df.dtypes)
df=df.astype('int32')
df['BMI']=df['BMI'].astype('float32')
df['AGE']=df['AGE'].astype('float32')
print(df.dtypes)


# define X, y and split into train and test sets
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

test_size = 0.3
random_state = 55
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

#
model = AutoFeatRegressor()

#
X_train_feature_creation = model.fit_transform(X_train, y_train)
X_test_feature_creation = model.transform(X_test)
print(X_train_feature_creation.head())

print('Number of new features -', X_train_feature_creation.shape[1] - X_train.shape[1])
X_train_feature_creation.to_csv('CABG_autofeat_top5.csv', index=False)
#X_train_feature_creation.to_csv('CABG_autofeat_top10.csv', index=False)