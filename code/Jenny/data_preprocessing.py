#%% Merge and convert datasets ***************
'''
import pandas as pd

df_20 = pd.read_spss('data/CABG 2020 Original.sav')
df_19 = pd.read_spss('data/CABG 2019 Original.sav')
df_18 = pd.read_spss('data/CABG 2018.sav')

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
df.to_csv('CABG_2018_2020.csv', index=False)
'''

# Data preprocessing ****************************
#*****************************************************
#%%
import pandas as pd
import numpy as np
from datasci import datasci

df = pd.read_csv('CABG_2018_2020.csv')

print(df.head())
print(df.shape)


#%% 
m = datasci(df)
m.size()

#%% Missing values analysis: NULL & -99 **************
# (1) Missing as NULL
missing = m.missingReport()
print(missing) # 104 features with missing values
# missing.to_csv('missing_report.csv',index=True )


#%%
missing_check = missing[missing['Percent of Nans']<50]
missing_check.index

# DISCHDEST: discharge destination, post-surgery variable
# READMISSION1: Any readminssion, post-surgery variable


#%% Cut down to 173 features after dropping variables with missing values (NULL)
df = df.drop(missing.index, axis=1)
df.shape


df.to_csv('CABG_173.csv',index=False)

#%%

df.columns.to_csv()

#%% 
# (2) Missing as -99 ********************

# How to treat -99: no response??

for col in df.columns:
    df[col].replace(-99, np.nan, inplace=True)

#%%
m = datasci(df)
missing99 = m.missingReport()
missing99
#missing99.to_csv('missing99_report.csv',index=True)


#%%
missing_50up = missing99[missing99['Percent of Nans']>50]
missing_50up.index

#%% Cut down to 125 features after dropping -99 (variables with over 50% missing)
df = df.drop(missing_50up.index, axis=1)
df.shape

#%% Cut down to 120 features by dropping irrelevant features
df = df.drop(['CASEID','PRNCPTX','CPT','WORKRVU','DOTHBLEED'], axis=1)


#%%
#%%
#df.to_csv('CABG_120.csv', index=False)


#%% 120 features
import pandas as pd
import numpy as np
from datasci import datasci

df = pd.read_csv('CABG_120.csv')

print(df.head())
print(df.shape)


#%% 
m = datasci(df)
m.missingReport()


#%% Imputation ***************************

#%% not skewed
m.eda('WEIGHT') # 0.18% missing

#%% not skewed
m.eda('HEIGHT') # 0.97% missing

#%%
m.imputation(['WEIGHT','HEIGHT'],impute='mean')

#%% calculate BMI using weight and height after imputation
df['BMI']= (df['WEIGHT']/df['HEIGHT']**2) *703

#%%
df = df.drop(['DOTHBLEED'], axis=1) # 45% missing

#%% 119 features, 2840 rows, no missing
df = df.dropna()
df.shape

#%%
df.to_csv('CABG_119_2840.csv', index=False)


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

#%%
m = datasci(df)

#%%
m.recode(col_list=cat_features,inplace=True)

#%%
df.to_csv('CABG_119_2840_recoded.csv',index=False)
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
