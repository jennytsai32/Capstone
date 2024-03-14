#%% Merge and convert datasets ***************
'''
### year 2018 - 2020 ###

import pandas as pd

df_20 = pd.read_spss('raw_data/CABG 2020 Original.sav')
df_19 = pd.read_spss('raw_data/CABG 2019 Original.sav')
df_18 = pd.read_spss('raw_data/CABG 2018.sav')

print(f'2020 size:{df_20.shape}')
print(f'2019 size:{df_19.shape}')
print(f'2018 size:{df_18.shape}')

print(f'Total sample size:{df_20.shape[0]+df_19.shape[0]+df_18.shape[0]}')
print(f'Total number of features: 274 - 276')



df_1 = pd.concat([df_18, df_19], axis=0)
df = pd.concat([df_1, df_20], axis=0)
print(df.head())
print(df.shape)

'''
#%%
### year 2018 - 2022 ###

import pandas as pd
import pyreadstat
from datasci import *

#df_18_20 = pd.read_csv('processed_data/CABG_2018_2020.csv')
df_18_21 = pd.read_csv('processed_data/CABG_2018_2021.csv')
df_21 = pd.read_spss('raw_data/CABG 2021 Original.sav')
df_22, meta = pyreadstat.read_sav("raw_data/CABG 2022 Original.sav", encoding="latin1")

print(f'2021 size:{df_21.shape}')
print(f'2022 size:{df_22.shape}')

#%%
# capitalize all variables in 2022 data and rename BLEEDIS
df_22.columns = df_22.columns.str.upper()
df_22 = df_22.rename(columns={'BLEEDDIS':'BLEEDIS'})

print(df_22['BLEEDIS'])
print(df_22.head())

#%%
# recode ETHNICITY_HISPANIC and OTHBLEED
m = datasci(df_18_21)
m.size()

oldVal = ['N', 'U', 'Y']
newVal = ['No', 'Unknown', 'Yes']
m.recode('ETHNICITY_HISPANIC',oldVal=oldVal, newVal=newVal,inplace=True)
print(df_18_21['ETHNICITY_HISPANIC'].value_counts())

oldVal = ['No Complication', 'Transfusions/Intraop/Postop']
newVal = ['No Complication', 'Blood Transfusion']
m.recode('OTHBLEED',oldVal=oldVal, newVal=newVal,inplace=True)
print(df_18_21['OTHBLEED'].value_counts())

#%%
# compare two files
#new_col, rmv_col, dif_val = file_compare(df_18_20, df_21)
new_col, rmv_col, dif_val = file_compare(df_18_21, df_22)


# check if flawed columns are in preselect features
df_preselect = pd.read_csv('preselect_features.csv')
preselct_features = df_preselect.Name.tolist()
preselct_features.append('OTHBLEED')

note = [col for col in rmv_col if col in preselct_features]
check = [col for col in dif_val if col in preselct_features]
print("Preselect cols removed:",note)
print("Preselect cols with different values:",check)

# check same variable with different values 
for col in check:
    print(col)
    print('df1:', sorted(df_18_21[col].unique().tolist()))
    print('df2:', sorted(df_22[col].unique().tolist()))

#%%
# merge the clean files
df_18_22 = pd.concat([df_18_21, df_22], axis=0)
print(df_18_22.head())
print(df_18_22.shape) # (8587,294)

df_18_22.to_csv('processed_data/2018_2022/CABG_2018_2022.csv', index=False)



#%%
'''
import sys
import os

path = os.path.abspath(os.path.join(os.curdir, os.pardir))
sys.path.append(path)
print(sys.path )

'''

# Data preprocessing ****************************
#*****************************************************
#%%

import pandas as pd
import numpy as np
from datasci import *


df = pd.read_csv('processed_data/2018_2022/CABG_2018_2022.csv')

print(df.head())
print(df.shape) # (8587, 294)


m = datasci(df)
m.size()

#%%
# recode -99 as nan
for col in df.columns:
    df[col] = df[col].replace([-99,'-99'], np.nan)

#df['DPRPT']
#%%
# remove columns with all nans
m.remove_all_nan_columns()

#%% check missing values
missing = m.missingReport()
print(missing) # 150 columns with missing values


#%% dropping variables with over 50% missing and cut down to 129 features
missing_50up = missing[missing['Percent of Nans'] > 50]
df.drop(missing_50up.index, axis=1, inplace=True)
df.shape #(85877, 129)

#%%
df.to_csv('processed_data/2018_2022/CABG_5yr_129.csv', index=False)



#%% Baseline data with 129 features ******************
#++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
from datasci import *


df = pd.read_csv('processed_data/2018_2022/CABG_5yr_129.csv')

print(df.head())
print(df.shape) #(8587, 129)

#%%
from datasci import *
glossary('NOTHBLEED')
glossary('DOTHBLEED')
glossary('PRHCT')

#%%
m = datasci(df)
m.missingReport() # 48 columns with missing less than 50%


#%% fix DOTHBLEED: replace nan with -1
df['DOTHBLEED'].fillna(-1, inplace=True)
print(df['DOTHBLEED'].isnull().sum())

#%% fix AGE and convert it to numeric
df['AGE'].replace('90+','90', inplace=True)
df['AGE'] = df['AGE'].astype(float)
type(df['AGE'][0])

#%%
std_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
std_cols.remove('PUFYEAR')
std_cols

#%% imputation
m.impute_all()

#%% calculate BMI using weight and height after imputation
df['BMI']= (df['WEIGHT']/df['HEIGHT']**2) *703

#%%


#%%
# standardize
std_cols.append('BMI')
m.standardize(col_list=std_cols)


#%%
df = df.drop(['CASEID','NOTHBLEED','DOTHBLEED'], axis=1)
df.head()

#%%
df.to_csv('processed_data/2018_2022/CABG_5yr_baseline_std.csv',index=False)
#df.to_csv('processed_data/2018_2022/CABG_5yr_baseline.csv',index=False)

#%% correlation with target

import pandas as pd
import numpy as np
from datasci import *


df = pd.read_csv('processed_data/2018_2022/CABG_5yr_baseline.csv')
df_preselect = pd.read_csv('preselect_features.csv')

preselect_features = df_preselect.Name.tolist()
preselect_features.append('BMI')

corr_lst = [df[col].corr(df['OTHBLEED']) for col in df.columns]
preselect = [(col in preselect_features) for col in df.columns]
output = pd.DataFrame({'Name':df.columns,
                       'r': corr_lst,
                       'preselect':preselect})
output = output.sort_values(by=['r'], ascending=False)
print(output.head())
output.to_csv('processed_data/2018_2022/CABG_5yr_baseline_review.csv',index=False)

#**** Preselct 43 features ****
#%%
import pandas as pd
from datasci import *

df = pd.read_csv('processed_data/2018_2022/CABG_5yr_baseline_std.csv')

df_preselect = pd.read_csv('preselect_features.csv')
print(df.shape) # (8587, 129)
print(df_preselect.shape) # (43, 2)

#%%
preselect_features = df_preselect.Name.tolist()
preselect_features.insert(0,'OTHBLEED') # add target
preselect_features.append('BMI') # add back to preselect
preselect_features.remove('PRPT') # remove PRPT because it's all null
df_43 = df[preselect_features]

print(df_43.head())
print(df_43.shape) # (8587, 44)


#%% cut down to 39 features after dropping highly correlated features
df_40 = df_43.drop(['HEIGHT','WEIGHT','ETHNICITY_HISPANIC'], axis=1)
df_40.shape

#%%
df_40.to_csv('processed_data/2018_2022/CABG_5yr_preselect40.csv',index=False)

#%%
#**** Preselct 20 features ****
preselect_20 = ['OTHBLEED','SEX','RACE_NEW','BMI','INOUT','AGE',
            'ANESTHES','DIABETES','SMOKE','DYSPNEA','FNSTATUS2',
            'HXCOPD','ASCITES','HXCHF','HYPERMED','DIALYSIS',
            'DISCANCR','STEROID','WTLOSS','BLEEDIS','TRANSFUS']

df_20 = df[preselect_20]
df_20.shape #(8587, 21)

#%%
df_20.to_csv('processed_data/2018_2022/CABG_5yr_preselect20.csv',index=False)


# %% feature selection using random forest
m = datasci(df_20)
target = 'OTHBLEED'
features = df_20.columns[1:]
m.featureSelection(features, target)

# %%
import pandas as pd
from datasci import *

df_40 = pd.read_csv('processed_data/2018_2022/CABG_5yr_preselect40.csv')
m = datasci(df_40)
target = 'OTHBLEED'
features = df_40.columns[1:]
m.featureSelection(features, target)


# %% quick check

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('processed_data/2018_2022/CABG_5yr_baseline.csv')
#df = pd.read_csv('processed_data/2018_2022/CABG_autofeat_top5.csv')


X = df.drop('OTHBLEED', axis=1)
y = df['OTHBLEED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1-score:", f1)

# %%
