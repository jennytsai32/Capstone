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

#%%
df = pd.read_csv('CABG_2018_2020.csv')

print(df.head())
print(df.shape)


#%% checking if -99 is read as null
print(df['PODIAG_OTHER'].isnull().sum())

#%%
features = ['OTHBLEED','SEX','RACE_NEW','HEIGHT','WEIGHT','INOUT','AGE',
            'ANESTHES','DIABETES','SMOKE','DYSPNEA','FNSTATUS2',
            'HXCOPD','ASCITES','HXCHF','HYPERMED','DIALYSIS',
            'DISCANCR','STEROID','WTLOSS','BLEEDIS','TRANSFUS','YEAR']


#%%
print(df['OTHBLEED'].value_counts())

# %%
