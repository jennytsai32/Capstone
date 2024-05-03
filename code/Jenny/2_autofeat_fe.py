import pandas as pd
from autofeat import AutoFeatRegressor
import warnings
warnings.filterwarnings("ignore")
from datasci import *


# import encoded dataset
df = pd.read_csv('/processed_data/2018_2022/CABG_5yr_preselect40.csv')
print(df.head())


# select top 20 important features
#df = df[['OTHBLEED','PRHCT', 'RACE_NEW', 'OPTIME', 'BMI', 'PRBUN', 'PRPTT', 'PRBILI', 'PRCREAT', 'PRPLATE', 'PRWBC', 'PRSGOT', 'PRALKPH', 'AGE', 'PRSODM', 'PRALBUM', 'TOTHLOS', 'PRINR', 'PUFYEAR', 'DYSPNEA', 'SEX']]

# select top 10 important features
#df = df[['OTHBLEED', 'PRHCT', 'RACE_NEW', 'OPTIME', 'BMI', 'PRBUN','PRPTT', 'PRBILI', 'PRCREAT', 'PRPLATE', 'PRWBC']]

# select top 5 important features
df = df[['OTHBLEED', 'PRHCT', 'RACE_NEW', 'OPTIME', 'BMI', 'PRBUN']]


# change data types to avoid errors
print(df.dtypes)
df=df.astype('float32')
df['OTHBLEED']=df['OTHBLEED'].astype('int32')
df['RACE_NEW']=df['RACE_NEW'].astype('int32')
#df['PUFYEAR']=df['PUFYEAR'].astype('int32')
#df['DYSPNEA']=df['DYSPNEA'].astype('int32')
#df['SEX']=df['SEX'].astype('int32')
print(df.dtypes)


# define X, y and split into train and test sets
X_train = df.iloc[:, 1:]
y_train = df.iloc[:, 0]

#
model = AutoFeatRegressor()

#
X_train_feature_creation = model.fit_transform(X_train, y_train)

print('Number of new features -', X_train_feature_creation.shape[1] - X_train.shape[1])
df_new = pd.concat([df['OTHBLEED'], X_train_feature_creation], axis=1)
print(df_new.shape)

#df_new.to_csv('CABG_autofeat_top20.csv', index=False)
#df_new.to_csv('CABG_autofeat_top10.csv', index=False)
df_new.to_csv('CABG_autofeat_top5.csv', index=False)

