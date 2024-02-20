#%% Merge and convert datasets ***************
'''
import pandas as pd

df_20 = pd.read_spss('raw_data/CABG 2020 Original.sav')
df_19 = pd.read_spss('raw_data/CABG 2019 Original.sav')
df_18 = pd.read_spss('raw_data/CABG 2018.sav')

print(f'2020 size:{df_20.shape}')
print(f'2019 size:{df_19.shape}')
print(f'2018 size:{df_18.shape}')

print(f'Total sample size:{df_20.shape[0]+df_19.shape[0]+df_18.shape[0]}')
print(f'Total number of features: 274 - 276')


#%%
df_1 = pd.concat([df_18, df_19], axis=0)
df = pd.concat([df_1, df_20], axis=0)
print(df.head())
print(df.shape)


#%%
import pandas as pd
import pyreadstat

df_18_20 = pd.read_csv('processed_data/CABG_2018_2020.csv')
df_21 = pd.read_spss('raw_data/CABG 2021 Original.sav')
df_22, meta = pyreadstat.read_sav("raw_data/CABG 2022 Original.sav", encoding="latin1")

print(f'2020 size:{df_21.shape}')
print(f'2022 size:{df_22.shape}')

#%%
df_1 = pd.concat([df_18_20, df_21], axis=0)
df = pd.concat([df_1, df_22], axis=0)
print(df.head())
print(df.shape)

#%%
df.to_csv('processed_data/CABG_18_22.csv', index=False)


#%%
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
from datasci import datasci


df = pd.read_csv('processed_data/CABG_18_22.csv')

print(df.head())
print(df.shape) # (8587, 303)


#%% 
m = datasci(df)
m.size()

#%%
# recode -99 as nan

for col in df.columns:
    df[col].replace(-99, np.nan, inplace=True)

#%%
m.remove_all_nan_columns()

#%% 
missing = m.missingReport()
print(missing) # 165 features with missing values


#%%
missing_50up = missing[missing['Percent of Nans']>50]

#%% Cut down to 129 features after dropping variables with over 50% missing
df.drop(missing_50up.index, axis=1, inplace=True)
df.shape (8587, 129)

#%%
df.to_csv('CABG_129.csv', index=False)


#%% 129 features *****************
import pandas as pd
import numpy as np
from datasci import datasci

df = pd.read_csv('processed_data/CABG_129.csv')
df_preselect = pd.read_csv('preselect_features.csv')

print(df.head())
print(df.shape)

#%%
preselct_features = df_preselect.Name.tolist()
preselct_features.remove('PRPT')
df_preselect = df[preselct_features]

#%% 
m = datasci(df_preselect)
m.missingReport()

#%% Preselect 43 features ******************
#++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
from datasci import datasci

df = pd.read_csv('processed_data/CABG_2018_2020.csv')
df_preselect = pd.read_csv('preselect_features.csv')

print(df.head())
print(df.shape) #(4953, 276)
print(df_preselect.shape) # (43, 2)

#%%
preselct_features = df_preselect.Name.tolist()
df_43 = df[preselct_features]
df_43['OTHBLEED'] = df['OTHBLEED']

cols = df_43.columns.tolist()
order = cols[-1:] + cols[:-1]
df_43 = df_43[order]

print(df_43.head())
print(df_43.shape)

#%%
df_43.to_csv('processed_data/CABG_preselect_original.csv',index=False)

#%%
df_43.dtypes.to_csv('processed_data/dtype.csv')
#%%
# recode -99 as nan
for col in df_43.columns:
    df_43[col].replace(-99, np.nan, inplace=True)

df_43['PRPT']

#%%
m = datasci(df_43)
m.missingReport()

#%%
m.remove_all_nan_columns()
m.missingReport()


#%%
#check = df_43[df_43.dtypes.sort_values().index]

#check.to_csv('test.csv',index=False)
#%%
df_43['AGE'].replace('90+','90', inplace=True)
df_43['AGE'] = df_43['AGE'].astype(float)
type(df_43['AGE'][0])

#%%
numeric_cols = df_43.select_dtypes(include=['int64', 'float64']).columns

#%% Imputation ***************************
m.impute_all()

#%% calculate BMI using weight and height after imputation
df_43['BMI']= (df_43['WEIGHT']/df_43['HEIGHT']**2) *703

df_43['BMI']


#%%
df_43['PUFYEAR'] = df_43['PUFYEAR'].astype('int32')
type(df_43['PUFYEAR'][0])
#%%
# standardize
m.standardize()


#%%
df_43.drop(['HEIGHT','WEIGHT'],axis=1)


#%%
df_43.head()
#%%
df_43.to_csv('processed_data/CABG_preselect.csv',index=False)




#%% Whole dataset (119 features, 2840 observations)
import pandas as pd
import numpy as np
from datasci import datasci

df = pd.read_csv('CABG_119_2840.csv')

print(df.head())
print(df.shape)


#%% 
df.isnull().sum()

#%%
df2 = df[df.dtypes.sort_values().index]

#%%
idx = df2.columns.get_loc('WNDCLAS')

#%%
cat_features = df2.columns[idx:].to_list()


#%% 20 pre-select features + target + year
features = ['OTHBLEED','SEX','RACE_NEW','BMI','INOUT','AGE',
            'ANESTHES','DIABETES','SMOKE','DYSPNEA','FNSTATUS2',
            'HXCOPD','ASCITES','HXCHF','HYPERMED','DIALYSIS',
            'DISCANCR','STEROID','WTLOSS','BLEEDIS','TRANSFUS','PUFYEAR']

df_20 = df[features]

#%%
df_20.to_csv('CABG_2018_2020_preselect.csv', index=False)


# RECODING (20 FEATURES ONLY) **************************
# %%
import pandas as pd
import numpy as np
from datasci import datasci

df_20 = pd.read_csv('CABG_2018_2020_preselect.csv')

print(df_20.head())
m = datasci(df_20)
m.size()


# %% check no missing values
print(df_20.isnull().sum())

# %% target variable EDA
print(m.eda('OTHBLEED'))

#%% categorical features
cat_features = df_20.columns[1:].to_list()
cat_features.remove('BMI')
cat_features.remove('AGE')
print(cat_features)


#%% (1) recode categorical features
m.recode(col_list=cat_features,inplace=True)


#%% (2) recode target varible
m.recode(col_list=['OTHBLEED'],inplace=True)

#%%

# %% (3) recode continuous variable

# Standarization: AGE & BMI
df_20['AGE'].replace('90+','90', inplace=True)
df_20['AGE'] = df_20['AGE'].astype(float)
type(df_20['AGE'][0])


#%%
df_20['AGE_NEW'] = (df_20['AGE'] - df_20['AGE'].mean())/(df_20['AGE'].std())
print(df_20['AGE_NEW'])

# %%
df_20['BMI_NEW'] = (df_20['BMI'] - df_20['BMI'].mean())/(df_20['BMI'].std())
print(df_20['BMI_NEW'])

# %%
df_20.to_csv('CABG_20_recoded.csv')


# %% feature selection using random forest
#features = cat_features.append('BMI_NEW')
#features = features.append('AGE_NEW')
target = 'OTHBLEED'
m.featureSelection(cat_features, target)

# %%

