#%% Merge datasets, no need to re-run ***************
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
df.to_csv('CABG_2018_2020.csv')

# Run codes starting here ****************************
#*****************************************************
#%%
import pandas as pd
import numpy as np
from datasci import datasci

#%%
df = pd.read_csv('CABG_2018_2020.csv')

print(df.head())
print(df.shape)


#%% checking if -99 is read as null
print(df['PODIAG_OTHER'].isnull().sum())

#%% target variable: not imbalanced
print(df['OTHBLEED'].value_counts())

#%%
# df['BMI']= (df['WEIGHT']/df['HEIGHT']**2) *703

# df.to_csv('CABG_2018_2020.csv')
#%%
features = ['OTHBLEED','SEX','RACE_NEW','BMI','INOUT','AGE',
            'ANESTHES','DIABETES','SMOKE','DYSPNEA','FNSTATUS2',
            'HXCOPD','ASCITES','HXCHF','HYPERMED','DIALYSIS',
            'DISCANCR','STEROID','WTLOSS','BLEEDIS','TRANSFUS','PUFYEAR']

df_20 = df[features]
df_20.head()

#%%
#df_20.to_csv('CABG_2018_2020_preselect.csv')
# %%

m = datasci(df_20)
m.size()

# %%
#print(m.missingReport())
# %%
print(df_20.isnull().sum())

# %%
print(m.eda('OTHBLEED'))


#%%
binary_features = df_20.columns[11:21].to_list()

#%%
binary_features.append('SMOKE')
print(binary_features)

#%%
old = ['Yes','No']
new = [1,0]

for f in binary_features:
    m.recode(column = str(f), oldVal=old, newVal=new, inplace=True)

# %%
    
print(df_20['HXCOPD'].value_counts())
print(df_20['DISCANCR'].value_counts())


#%%
df_20.to_csv('check.csv')
# %%
#features = df_20.iloc[:, 1:].columns
features = binary_features
target = 'OTHBLEED'
m.featureSelection(features, target)

# %%
