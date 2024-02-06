#%%
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
df_20['YEAR'] = 2020
df_19['YEAR'] = 2019
df_18['YEAR'] = 2018

#%%     
features = ['OTHBLEED','SEX','RACE_NEW','HEIGHT','WEIGHT','INOUT','AGE',
            'ANESTHES','DIABETES','SMOKE','DYSPNEA','FNSTATUS2',
            'HXCOPD','ASCITES','HXCHF','HYPERMED','DIALYSIS',
            'DISCANCR','STEROID','WTLOSS','BLEEDIS','YEAR']

df_20_s = df_20[features]
df_19_s = df_19[features]
df_18_s = df_18[features]

#%%
print(df_20['OTHBLEED'].value_counts())
# %%
