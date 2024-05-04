# import packages

import pandas as pd
import numpy as np
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# import the baseline models class
# get py file folder path
sys.path.append(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\component')
# os.chdir(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code')
# print('Current working directory:', os.getcwd())

# import utilities and class
from datasci import *
from class_decision_tree import DecisionTree
from class_svm import SVM
from class_logistic import LogReg
from class_gradient_boosting import GradientBoosting
from class_xgboost import XGB
from class_gaussian_nb import NaiveBayesGaussianNB
from class_random_forest import RandomForest
from class_knn import KNN
from utils_models import *

# adjust output window size

#np.set_printoptions(linewidth=desired_width)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

# ===================================================
# PART 1. Data pre-processing         ||
# ===================================================
#%% Merge and convert datasets ***************
#%% Sample size table

data_path = r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\main_code'

df_15 = pd.read_spss(data_path + r'\raw_data/CABG 2015 Original.sav')
df_16 = pd.read_spss(data_path + r'\raw_data/CABG 2016 Original.sav')
df_17 = pd.read_spss(data_path + r'\raw_data/CABG 2017 Original.sav')
df_18 = pd.read_spss(data_path + r'\raw_data/CABG 2018.sav')
df_19 = pd.read_spss(data_path + r'\raw_data/CABG 2019 Original.sav')
df_20 = pd.read_spss(data_path + r'\raw_data/CABG 2020 Original.sav')
df_21 = pd.read_spss(data_path + r'\raw_data/CABG 2021 Original.sav')
df_22, meta = pyreadstat.read_sav(data_path + r"\raw_data/CABG 2022 Original.sav", encoding="latin1")

summary = pd.DataFrame(
    {'Year':[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
     '# of rows':[df_15.shape[0],df_16.shape[0],df_17.shape[0],df_18.shape[0],df_19.shape[0],df_20.shape[0],df_21.shape[0],df_22.shape[0]],
     '# of cols':[df_15.shape[1],df_16.shape[1],df_17.shape[1],df_18.shape[1],df_19.shape[1],df_20.shape[1],df_21.shape[1],df_22.shape[1]]
     })

print(summary)
#summary.to_csv('dataset_summary.csv',index=False)

### year 2018 - 2020 ###

df_20 = pd.read_spss(data_path + r'\raw_data/CABG 2020 Original.sav')
df_19 = pd.read_spss(data_path + r'\raw_data/CABG 2019 Original.sav')
df_18 = pd.read_spss(data_path + r'\raw_data/CABG 2018.sav')

print(f'2020 size:{df_20.shape}')
print(f'2019 size:{df_19.shape}')
print(f'2018 size:{df_18.shape}')

df_1 = pd.concat([df_18, df_19], axis=0)
df = pd.concat([df_1, df_20], axis=0)
print(df.head())
print(df.shape)

#%%
### adding year 2021 - 2022 ###

#df_18_20 = pd.read_csv('processed_data/CABG_2018_2020.csv')
df_18_21 = pd.read_csv(data_path + r'\processed_data/CABG_2018_2021.csv')
df_21 = pd.read_spss(data_path + r'\raw_data/CABG 2021 Original.sav')
df_22, meta = pyreadstat.read_sav(data_path + r"\raw_data/CABG 2022 Original.sav", encoding="latin1")

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
### adding year 2015 - 2017 ###

df_18_22 = pd.read_csv(data_path+r'\processed_data/2018_2022/CABG_2018_2022.csv')
df_15 = pd.read_spss(data_path+r'\raw_data/CABG 2015 Original.sav')
df_16 = pd.read_spss(data_path+r'\raw_data/CABG 2016 Original.sav')
df_17 = pd.read_spss(data_path+r'\raw_data/CABG 2017 Original.sav')
df_18 = pd.read_spss(data_path+r'\raw_data/CABG 2018.sav')
#df_22, meta = pyreadstat.read_sav("raw_data/CABG 2022 Original.sav", encoding="latin1")

print(f'2015 size:{df_15.shape}')
print(f'2016 size:{df_16.shape}')
print(f'2017 size:{df_17.shape}')

#%%
#new_col, rmv_col, dif_val = file_compare(df_18, df_15)
#new_col, rmv_col, dif_val = file_compare(df_18, df_16)
new_col, rmv_col, dif_val = file_compare(df_18, df_17)

#%%
# check if flawed columns are in preselect features
df_preselect = pd.read_csv('output/preselect_features.csv')
preselct_features = df_preselect.Name.tolist()
preselct_features.append('OTHBLEED')
check = [col for col in dif_val if col in preselct_features]

#%%
# check same variable with different values
for col in check:
    print(col)
    print('df1:', sorted(df_18[col].unique().tolist()))
    print('df2:', sorted(df_17[col].unique().tolist()))

#%%
# merge the clean files
df_a = pd.concat([df_15, df_16], axis=0)
df_b = pd.concat([df_a, df_17], axis=0)

#%%
# recode ETHNICITY_HISPANIC and OTHBLEED
m = datasci(df_b)
m.size()

oldVal = ['N', 'U', 'Y']
newVal = ['No', 'Unknown', 'Yes']
m.recode('ETHNICITY_HISPANIC',oldVal=oldVal, newVal=newVal,inplace=True)
print(df_b['ETHNICITY_HISPANIC'].value_counts())

oldVal = ['No Complication', 'Transfusions/Intraop/Postop']
newVal = ['No Complication', 'Blood Transfusion']
m.recode('OTHBLEED',oldVal=oldVal, newVal=newVal,inplace=True)
print(df_b['OTHBLEED'].value_counts())

#%%
df_15_22 = pd.concat([df_b, df_18_22], axis=0)
df_15_22.to_csv(data_path+r'\processed_data/2015_2022/CABG_2015_2022.csv', index=False)


# Data preprocessing ****************************
#*****************************************************

df = pd.read_csv(data_path+r'\processed_data/2015_2022/CABG_2015_2022.csv')

print(df.head())
print(df.shape) # (13534, 296)


m = datasci(df)
m.size()

#%%
# recode -99 as nan
for col in df.columns:
    df[col] = df[col].replace([-99,'-99'], np.nan)

#%%
# remove columns with all nans
m.remove_all_nan_columns()

#%% check missing values
missing = m.missingReport()
print(missing) # 168 columns with missing values


#%% dropping variables with over 50% missing and cut down to 128 features
missing_50up = missing[missing['Percent of Nans'] > 50]
df.drop(missing_50up.index, axis=1, inplace=True)

#%%
df.to_csv(data_path+r'\processed_data/2015_2022/CABG_8yr_128.csv', index=False)

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
# standardize
std_cols.append('BMI')
m.standardize(col_list=std_cols)

#%%
df.to_csv(data_path+r'\processed_data/2015_2022/CABG_8yr_baseline_std.csv',index=False)
#df.to_csv('processed_data/2015_2022/CABG_8yr_baseline.csv',index=False)


#**** Preselct 41 features ****
#%%
df_preselect = pd.read_csv(data_path+r'\output/preselect_features.csv')
preselect_features = df_preselect.Name.tolist()
print(preselect_features)

#%%
preselect_features.insert(0,'OTHBLEED') # add target
df_41 = df[preselect_features]

print(df_41.head())
print(df_41.shape) # (13534, 42)


#%%
df_41.to_csv(data_path+r'\processed_data/2015_2022/CABG_8yr_preselect41.csv',index=False)
#df_41.to_csv('processed_data/2015_2022/CABG_8yr_preselect41_no_std.csv',index=False)

#%%
#**** Preselct 20 features ****
preselect_20 = ['OTHBLEED','SEX','RACE_NEW','BMI','INOUT','AGE',
            'ANESTHES','DIABETES','SMOKE','DYSPNEA','FNSTATUS2',
            'HXCOPD','ASCITES','HXCHF','HYPERMED','DIALYSIS',
            'DISCANCR','STEROID','WTLOSS','BLEEDIS','TRANSFUS']

df_20 = df[preselect_20]

#%%
df_20.to_csv(data_path+r'\processed_data/2015_2022/CABG_8yr_preselect20.csv',index=False)

# %% EDA

df = pd.read_csv(data_path+r'\processed_data/2015_2022/CABG_8yr_128.csv')

print(df.head())
print(df.shape) #(13534, 128)

# fix DOTHBLEED: replace nan with -1
df['DOTHBLEED'].fillna(-1, inplace=True)
print(df['DOTHBLEED'].isnull().sum())

# fix AGE and convert it to numeric
df['AGE'].replace('90+','90', inplace=True)
df['AGE'] = df['AGE'].astype(float)
type(df['AGE'][0])

# add BMI
df['BMI']= (df['WEIGHT']/df['HEIGHT']**2) *703

#
m = datasci(df)

#%%
m.eda('SEX')

# %%
m.eda('AGE')
print(df.AGE.mean())
print(df.AGE.std())

# %%
m.eda('RACE_NEW')

# %% Target variable analysis
m.eda('OTHBLEED')

#%% target recoded into 3-class
df['OTHBLEED3'] = df['DOTHBLEED'].mask(df['DOTHBLEED']>0,'Postop', inplace=False)
df['OTHBLEED3'].mask(df['DOTHBLEED']==0,'Intra', inplace=True)
df['OTHBLEED3'].mask(df['DOTHBLEED']==-1,'No Transfusions', inplace=True)
print(df['OTHBLEED3'].value_counts())

# %%
m.eda('OTHBLEED3')

# ==============================================================================
# PART 2. AutoFeat               ||
# ==============================================================================
import pandas as pd
from autofeat import AutoFeatRegressor
import warnings
warnings.filterwarnings("ignore")
from datasci import *

# import encoded dataset
df = pd.read_csv(data_path+r'\processed_data/2018_2022/CABG_5yr_preselect40.csv')
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

# =================================
# PART 3. TPOT                   ||
# =================================
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv(data_path+r'\processed_data/2018_2022/CABG_5yr_preselect40.csv')
print(df.shape)



Y = np.array(df['OTHBLEED'])
X = np.array(df.loc[:, df.columns != 'OTHBLEED'])

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, scoring='f1_weighted')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# ====================================
# PART4. DataSynthesizer            ||
# ====================================

pip install DataSynthesizer
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network

df = pd.read_csv(data_path+'\processed_data/2015_2022/CABG_8yr_preselect41.csv', header=0)

print(df.head())
print(df.shape)

# Backup the original dataset
df_backup = df.copy()

# Specify categorical attributes
categorical_attributes = {'OTHBLEED':True,
                          "PUFYEAR": True,
                          "SEX": True,
                          "RACE_NEW": True,
                          "INOUT": True,
                          "ANESTHES": True,
                          "DIABETES": True,
                          "SMOKE": True,
                          "DYSPNEA": True,
                          "FNSTATUS2": True,
                          "VENTILAT": True,
                          "HXCOPD": True,
                          "ASCITES": True,
                          "HXCHF": True,
                          "HYPERMED": True,
                          "RENAFAIL": True,
                          "DIALYSIS": True,
                          "DISCANCR": True,
                          "WNDINF": True,
                          "STEROID": True,
                          "WTLOSS": True,
                          "BLEEDIS": True,
                          "TRANSFUS": True,
                          "EMERGNCY": True,
                          "ASACLAS": True
                          }

# Define privacy settings
epsilon = 0.1
degree_of_bayesian_network = 2
num_tuples_to_generate = 13534

# Initialize DataDescriber with category threshold
describer = DataDescriber(category_threshold=20)
print('Initialization completed.')

# Describe the dataset to create a Bayesian network
describer.describe_dataset_in_correlated_attribute_mode(dataset_file='CABG_8yr_preselect41.csv',
                                                        epsilon=epsilon,
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes
                                                        )

print('Describer work completed.')

# Save dataset description to a JSON file
description_file = 'CABG_description.json'
describer.save_dataset_description_to_file(description_file)

print('Saved data discription.')

# Display the Bayesian network
display_bayesian_network(describer.bayesian_network)

print('Display Baysian network completed.')

# Generate the dataset
generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)

print('Data generation completed.')

# Save synthetic data to a CSV file
synthetic_data_file = 'CABG_synthetic_Bayesian_8yr.csv'
generator.save_synthetic_data(synthetic_data_file)

print('Synthetic data saved.')

# Check synthetic data file
df_synthetic = pd.read_csv(synthetic_data_file)
print(df_synthetic.head())

# =====================================
# PART5. RealtabFormer               ||
# =====================================
#%%
from realtabformer import REaLTabFormer

df = pd.read_csv(data_path+r'\processed_data/2015_2022/CABG_8yr_preselect41.csv', header=0)

print(df.head())
print(df.shape)

#%%
# Non-relational or parent table.
rtf_model = REaLTabFormer(
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

#%%
# Fit the model on the dataset.
# Additional parameters can be
# passed to the `.fit` method.
rtf_model.fit(df)

#%%
# Save the model to the current directory.
# A new directory `rtf_model/` will be created.
# In it, a directory with the model's
# experiment id `idXXXX` will also be created
# where the artefacts of the model will be stored.
rtf_model.save(data_path+r"\rtf_model/")

# Generate synthetic data with the same
# number of observations as the real dataset.
samples = rtf_model.sample(n_samples=len(df))

samples.to_csv(data_path+r'\realtabformer_data.csv', index=False)

#%%
# Load the saved model. The directory to the
# experiment must be provided.
#rtf_model2 = REaLTabFormer.load_from_dir(path="output/rtf_model/id000017122473330058563584")

# ==============================================================================
# PART6. Classical models and results (before feature selection)               ||
# ==============================================================================

# =====================================
# iteration 1 - 2018-2020
# =====================================

# df1. baseline 2018-2020, 126 features
df_baseline_2018_2022 = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/Jenny/processed_data/2018_2022/CABG_5yr_baseline.csv')
df_baseline_2018_2020 = df_baseline_2018_2022[(df_baseline_2018_2022['PUFYEAR']==2018) | (df_baseline_2018_2022['PUFYEAR']==2019) | (df_baseline_2018_2022['PUFYEAR']==2020)]

# =====================================
# iteration 2 - 2018-2022
# =====================================
# df2. 2018-2022, 126 features
# USE THIS df_baseline_2018_2022

# =====================================
# iteration 3 - 41 features
# =====================================

# df3. 5year, 41 selected features （added OTHERCPT1）
df_5yr_preselect41 = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_5yr_preselect41.csv')
df_5yr_preselect41_no_year = df_5yr_preselect41.drop(['PUFYEAR'], axis=1)

# =====================================
# iteration 4 - PCA
# =====================================
# PCA
from class_pca import PCA_for_Feature_Selection

df = df_5yr_preselect41_no_year

# import PCA class
pca_module = PCA_for_Feature_Selection(df, 'OTHBLEED')

# how many features suggested to reduce
pca_module.PCA_Reduced_Feature()

pca_module.Explained_Variance_Ratio()
pca_module.Reduced_Feature_Space_Plot()
pca_module.Reduced_Feature_Space_Heatmap()

# get new df to feed into the models
df_new = pca_module.PCA_New_df()
df_new.to_csv(r'C:\Users\wb576802\OneDrive - WBG\Documents\df_PCA_39feature.csv',index=False)

# add the target back
print(df_new.shape)
# adding original target value back to df_new
df_new['OTHBLEED'] = df['OTHBLEED'].values

# =====================================
# iteration 5 - AutoFeat             ||
# =====================================
df_autofeat20 = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_5yr_preselect20.csv')
df_autofeat10 = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_autofeat_top10.csv')
df_autofeat5 = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_autofeat_top5.csv')

# =====================================
# iteration 6 - TPOT                 ||
# =====================================
from class_tpot import TPOT

random_state = 100
test_size = .25
target = 'OTHBLEED'
k_folds = 10

# initiate the model
# 0.25 on test size, random_state = 100, generations=5, population_size=20, verbosity=2
df = df_5yr_preselect41_no_year
model_tpot = TPOT(df, target, test_size, random_state, 5, 20, 2)
model_tpot.scores

# ==========================================================
# iteration 7 -    Synthetic data - Bayesian networks   ||
# ==========================================================

# df_syn_GANs = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_synthetic_GANs.csv')
# ## convert target column back to 0 and 1
# df_syn_GANs.loc[abs(df_syn_GANs['OTHBLEED']) >= 0.5] = 1
# df_syn_GANs.loc[df_syn_GANs['OTHBLEED'] < 0.5] = 0

df_syn_bay = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_synthetic_Bayesian.csv')
df_syn_bay = df_syn_bay.drop(['PUFYEAR'], axis=1)

# ==============================
# iteration 8 -  2015-2022     ||
# ==============================
df_8yr_preselect41 = pd.read_csv(r'https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2015_2022/CABG_8yr_preselect41.csv')
df_8yr_feature40 = df_8yr_preselect41.drop(['PUFYEAR'], axis=1)

# ==========================================================
# iteration 9 -    Synthetic data - realtabformer   ||
# ==========================================================

df_syn_realtabformer = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2015_2022/realtabformer_data.csv')


########################################################################
########################################################################
##======================================================================
# MODEL ITERATION PART - starting below

# IMPORTANT - assign each of the datasets above to "df" below and run the entire model iteration code part, this process will need to be repeated many times.
##======================================================================
########################################################################
########################################################################


# set cross-cutting variables
df = df_8yr_preselect41    # replace all datasets above and run the iterations
random_state = 100
test_size = .25
target = 'OTHBLEED'
k_folds = 10

#===============================================
# Model 1: call Decision Tree model class
#===============================================

# initiate the model
# 0.25 on test size; random_state = 100; criterion = 'gini', max_depth = 3, min_samples_leaf = 5
model_dt_gini = DecisionTree(df, target, test_size, random_state,'gini', 3, 5)

# run Decision Tree model and display results
## prediction
Model_Predict(target, model_dt_gini.model_name, model_dt_gini.model, model_dt_gini.X_test)

## model report
y_pred_dt_gini = Model_Predict(target, model_dt_gini.model_name, model_dt_gini.model, model_dt_gini.X_test)
Model_Report(target, model_dt_gini.model_name, model_dt_gini.y_test, y_pred_dt_gini)

## model accuracy
Model_Accuracy(target, model_dt_gini.model_name, random_state, model_dt_gini.model, model_dt_gini.y_test, y_pred_dt_gini)
accuracy_dt_gini = Model_Accuracy(target, model_dt_gini.model_name, random_state, model_dt_gini.model, model_dt_gini.y_test, y_pred_dt_gini)

## model mean accuracy (k-fold)
Model_Mean_Accuracy(target, model_dt_gini.model_name, k_folds, random_state, model_dt_gini.model, model_dt_gini.X, model_dt_gini.y)
mean_accuracy_dt_gini = Model_Mean_Accuracy(target, model_dt_gini.model_name, k_folds, random_state, model_dt_gini.model, model_dt_gini.X, model_dt_gini.y)

## model RMSE
Model_RMSE(target, model_dt_gini.model_name, model_dt_gini.y_test, y_pred_dt_gini)
rmse_dt_gini = Model_RMSE(target, model_dt_gini.model_name, model_dt_gini.y_test, y_pred_dt_gini)

## model f1 score
Model_F1(target, model_dt_gini.model_name, model_dt_gini.y_test, y_pred_dt_gini)
f1_dt_gini = Model_F1(target, model_dt_gini.model_name, model_dt_gini.y_test, y_pred_dt_gini)

## confusion matrix
Model_Confusion_Matrix(target, model_dt_gini.model_name, model_dt_gini.y_test, y_pred_dt_gini)

## plot confusion matrix
Plot_Confusion_Matrix(target, model_dt_gini.model_name, model_dt_gini.y_test, y_pred_dt_gini, model_dt_gini.class_names)

## plot decision tree
Plot_Decision_Tree(target, model_dt_gini.model_name, model_dt_gini.model, model_dt_gini.feature_names)

## ROC AUC
Model_ROC_AUC_Score(target, model_dt_gini.model_name, model_dt_gini.model, model_dt_gini.X_test, model_dt_gini.y_test)
auc_dt_gini = Model_ROC_AUC_Score(target, model_dt_gini.model_name, model_dt_gini.model, model_dt_gini.X_test, model_dt_gini.y_test)

## plot ROC AUC
Plot_ROC_AUC(target, model_dt_gini.model_name, model_dt_gini.model, model_dt_gini.X_test, model_dt_gini.y_test)

## results table
Model_Results_Table(model_dt_gini.model_name, model_dt_gini.parameters, target, test_size, accuracy_dt_gini, mean_accuracy_dt_gini, k_folds, rmse_dt_gini, f1_dt_gini, auc_dt_gini)
results_dt_gini = Model_Results_Table(model_dt_gini.model_name, model_dt_gini.parameters, target, test_size, accuracy_dt_gini, mean_accuracy_dt_gini, k_folds, rmse_dt_gini, f1_dt_gini, auc_dt_gini)

#===============================================
# Model 2: call Decision Tree model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100; k_folds = 5; criterion = 'entropy', max_depth = 3, min_samples_leaf = 5
model_dt_entropy = DecisionTree(df, target, test_size, random_state,'entropy', 3, 5)

# calculate needed metrics
y_pred_dt_entropy = Model_Predict(target, model_dt_entropy.model_name, model_dt_entropy.model, model_dt_entropy.X_test)
accuracy_dt_entropy = Model_Accuracy(target, model_dt_entropy.model_name, random_state, model_dt_entropy.model, model_dt_entropy.y_test, y_pred_dt_entropy)
mean_accuracy_dt_entropy = Model_Mean_Accuracy(target, model_dt_entropy.model_name, k_folds, random_state, model_dt_entropy.model, model_dt_entropy.X, model_dt_entropy.y)
rmse_dt_entropy = Model_RMSE(target, model_dt_entropy.model_name, model_dt_entropy.y_test, y_pred_dt_entropy)
f1_dt_entropy = Model_F1(target, model_dt_entropy.model_name, model_dt_entropy.y_test, y_pred_dt_entropy)
auc_dt_entropy = Model_ROC_AUC_Score(target, model_dt_entropy.model_name, model_dt_entropy.model, model_dt_entropy.X_test, model_dt_entropy.y_test)

# results table
results_dt_entropy = Model_Results_Table(model_dt_entropy.model_name, model_dt_entropy.parameters, target, test_size, accuracy_dt_entropy, mean_accuracy_dt_entropy, k_folds, rmse_dt_entropy, f1_dt_entropy, auc_dt_entropy)

#===============================================
# Model 3: call SVM model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100; kernel='linear', C=1.0, gamma='auto'
model_svm_linear = SVM(df,target,test_size,random_state,'linear',1.0,'auto')

# calculate needed metrics
y_pred_svm_linear = Model_Predict(target, model_svm_linear.model_name, model_svm_linear.model, model_svm_linear.X_test)
accuracy_svm_linear = Model_Accuracy(target, model_svm_linear.model_name, random_state, model_svm_linear.model, model_svm_linear.y_test, y_pred_svm_linear)
mean_accuracy_svm_linear = Model_Mean_Accuracy(target, model_svm_linear.model_name, k_folds, random_state, model_svm_linear.model, model_svm_linear.X, model_svm_linear.y)
rmse_svm_linear = Model_RMSE(target, model_svm_linear.model_name, model_svm_linear.y_test, y_pred_svm_linear)
f1_svm_linear = Model_F1(target, model_svm_linear.model_name, model_svm_linear.y_test, y_pred_svm_linear)
auc_svm_linear = Model_ROC_AUC_Score(target, model_svm_linear.model_name, model_svm_linear.model, model_svm_linear.X_test, model_svm_linear.y_test)

# results table
results_svm_linear = Model_Results_Table(model_svm_linear.model_name, model_svm_linear.parameters, target, test_size, accuracy_svm_linear, mean_accuracy_svm_linear, k_folds, rmse_svm_linear, f1_svm_linear, auc_svm_linear)

#===============================================
# Model 4: call SVM model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100; kernel='rbf', C=1.0, gamma=0.2
model_svm_rbf = SVM(df,target,test_size,random_state,'rbf',1.0, .2)

# calculate needed metrics
y_pred_svm_rbf = Model_Predict(target, model_svm_rbf.model_name, model_svm_rbf.model, model_svm_rbf.X_test)
accuracy_svm_rbf = Model_Accuracy(target, model_svm_rbf.model_name, random_state, model_svm_rbf.model, model_svm_rbf.y_test, y_pred_svm_rbf)
mean_accuracy_svm_rbf = Model_Mean_Accuracy(target, model_svm_rbf.model_name, k_folds, random_state, model_svm_rbf.model, model_svm_rbf.X, model_svm_rbf.y)
rmse_svm_rbf = Model_RMSE(target, model_svm_rbf.model_name, model_svm_rbf.y_test, y_pred_svm_rbf)
f1_svm_rbf = Model_F1(target, model_svm_rbf.model_name, model_svm_rbf.y_test, y_pred_svm_rbf)
auc_svm_rbf = Model_ROC_AUC_Score(target, model_svm_rbf.model_name, model_svm_rbf.model, model_svm_rbf.X_test, model_svm_rbf.y_test)

# results table
results_svm_rbf = Model_Results_Table(model_svm_rbf.model_name, model_svm_rbf.parameters, target, test_size, accuracy_svm_rbf, mean_accuracy_svm_rbf, k_folds, rmse_svm_rbf, f1_svm_rbf, auc_svm_rbf)

#===============================================
# Model 5: call Logistic Regression model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100
model_log = LogReg(df,target,test_size, random_state)

# calculate needed metrics
y_pred_log = Model_Predict(target, model_log.model_name, model_log.model, model_log.X_test)
accuracy_log = Model_Accuracy(target, model_log.model_name, random_state, model_log.model, model_log.y_test, y_pred_log)
mean_accuracy_log = Model_Mean_Accuracy(target, model_log.model_name, k_folds, random_state, model_log.model, model_log.X, model_log.y)
rmse_log = Model_RMSE(target, model_log.model_name, model_log.y_test, y_pred_log)
f1_log = Model_F1(target, model_log.model_name, model_log.y_test, y_pred_log)
auc_log = Model_ROC_AUC_Score(target, model_log.model_name, model_log.model, model_log.X_test, model_log.y_test)

# results table
results_log = Model_Results_Table(model_log.model_name, model_log.parameters, target, test_size, accuracy_log, mean_accuracy_log, k_folds, rmse_log, f1_log, auc_log)

#===============================================
# Model 6: call GradientBoosting model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100; n_estimators=300, learning_rate=0.05
model_gb = GradientBoosting(df,target, test_size,random_state, 300, 0.05)

# calculate needed metrics
y_pred_gb = Model_Predict(target, model_gb.model_name, model_gb.model, model_gb.X_test)
accuracy_gb = Model_Accuracy(target, model_gb.model_name, random_state, model_gb.model, model_gb.y_test, y_pred_gb)
mean_accuracy_gb = Model_Mean_Accuracy(target, model_gb.model_name, k_folds, random_state, model_gb.model, model_gb.X, model_gb.y)
rmse_gb = Model_RMSE(target, model_gb.model_name, model_gb.y_test, y_pred_gb)
f1_gb = Model_F1(target, model_gb.model_name, model_gb.y_test, y_pred_gb)
auc_gb = Model_ROC_AUC_Score(target, model_gb.model_name, model_gb.model, model_gb.X_test, model_gb.y_test)

# results table
results_gb = Model_Results_Table(model_gb.model_name, model_gb.parameters, target, test_size, accuracy_gb, mean_accuracy_gb, k_folds, rmse_gb, f1_gb, auc_gb)

#===============================================
# Model 7: call XGBoost model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100; n_estimators=100, eta=0.3
model_xgb = XGB(df,target, test_size, random_state, 100, 0.3)

# calculate needed metrics
y_pred_xgb = Model_Predict(target, model_xgb.model_name, model_xgb.model, model_xgb.X_test)
accuracy_xgb = Model_Accuracy(target, model_xgb.model_name, random_state, model_xgb.model, model_xgb.y_test, y_pred_xgb)
mean_accuracy_xgb = Model_Mean_Accuracy(target, model_xgb.model_name, k_folds, random_state, model_xgb.model, model_xgb.X, model_xgb.y)
rmse_xgb = Model_RMSE(target, model_xgb.model_name, model_xgb.y_test, y_pred_xgb)
f1_xgb = Model_F1(target, model_xgb.model_name, model_xgb.y_test, y_pred_xgb)
auc_xgb = Model_ROC_AUC_Score(target, model_xgb.model_name, model_xgb.model, model_xgb.X_test, model_xgb.y_test)

# results table
results_xgb = Model_Results_Table(model_xgb.model_name, model_xgb.parameters, target, test_size, accuracy_xgb, mean_accuracy_xgb, k_folds, rmse_xgb, f1_xgb, auc_xgb)

#===============================================
# Model 8: call GaussianNB model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100;
model_nb = NaiveBayesGaussianNB(df, target, test_size, random_state)

# calculate needed metrics
y_pred_nb = Model_Predict(target, model_nb.model_name, model_nb.model, model_nb.X_test)
accuracy_nb = Model_Accuracy(target, model_nb.model_name, random_state, model_nb.model, model_nb.y_test, y_pred_nb)
mean_accuracy_nb = Model_Mean_Accuracy(target, model_nb.model_name, k_folds, random_state, model_nb.model, model_nb.X, model_nb.y)
rmse_nb = Model_RMSE(target, model_nb.model_name, model_nb.y_test, y_pred_nb)
f1_nb = Model_F1(target, model_nb.model_name, model_nb.y_test, y_pred_nb)
auc_nb = Model_ROC_AUC_Score(target, model_nb.model_name, model_nb.model, model_nb.X_test, model_nb.y_test)

# results table
results_nb = Model_Results_Table(model_nb.model_name, model_nb.parameters, target, test_size, accuracy_nb, mean_accuracy_nb, k_folds, rmse_nb, f1_nb, auc_nb)

#===============================================
# Model 9: call KNN model class
#===============================================
# initiate the model
# 0.25 on test size; random_state = 100, n_neighbors = 3
model_knn = KNN(df,target, test_size, random_state, 3)

# calculate needed metrics
y_pred_knn = Model_Predict(target, model_knn.model_name, model_knn.model, model_knn.X_test)
accuracy_knn = Model_Accuracy(target, model_knn.model_name, random_state, model_knn.model, model_knn.y_test, y_pred_knn)
mean_accuracy_knn = Model_Mean_Accuracy(target, model_knn.model_name, k_folds, random_state, model_knn.model, model_knn.X, model_knn.y)
rmse_knn = Model_RMSE(target, model_knn.model_name, model_knn.y_test, y_pred_knn)
f1_knn = Model_F1(target, model_knn.model_name, model_knn.y_test, y_pred_knn)
auc_knn = Model_ROC_AUC_Score(target, model_knn.model_name, model_knn.model, model_knn.X_test, model_knn.y_test)

# results table
results_knn = Model_Results_Table(model_knn.model_name, model_knn.parameters, target, test_size, accuracy_knn, mean_accuracy_knn, k_folds, rmse_knn, f1_knn, auc_knn)

#===============================================
# Model 10: call RandomForest model class
#===============================================
# initiate the model
# 0.25 on test size, random_state = 100, n_estimators=100, feature_importances = 20
model_rf = RandomForest(df,target, test_size, random_state, 100, 20)

# plot feature importances
Plot_Feature_Importances(model_rf.f_importances)

# calculate needed metrics
y_pred_rf = Model_Predict(target, model_rf.model_name, model_rf.model_k_features, model_rf.newX_test)
accuracy_rf = Model_Accuracy(target, model_rf.model_name, random_state, model_rf.model_k_features, model_rf.y_test, y_pred_rf)
mean_accuracy_rf = Model_Mean_Accuracy(target, model_rf.model_name, k_folds, random_state, model_rf.model_k_features, model_rf.X, model_rf.y)
rmse_rf = Model_RMSE(target, model_rf.model_name, model_rf.y_test, y_pred_rf)
f1_rf = Model_F1(target, model_rf.model_name, model_rf.y_test, y_pred_rf)
auc_rf = Model_ROC_AUC_Score(target, model_rf.model_name, model_rf.model_k_features, model_rf.newX_test, model_rf.y_test)

# results table
results_rf = Model_Results_Table(model_rf.model_name, model_rf.parameters, target, test_size, accuracy_rf, mean_accuracy_rf, k_folds, rmse_rf, f1_rf, auc_rf)

# ===================================================
# Display all the model results to compare
# ===================================================
combined_results_table = pd.concat([results_dt_gini,
                                    results_dt_entropy,
                                    results_svm_linear,
                                    results_svm_rbf,
                                    results_nb,
                                    results_log,
                                    results_gb,
                                    results_xgb,
                                    results_knn,
                                    results_rf
                                    ])
combined_results_table

# ===================================================
# Plot combined ROC plots
# ===================================================
# get all the ROC variables
lst = [Plot_ROC_AUC(target, model_dt_gini.model_name, model_dt_gini.model, model_dt_gini.X_test, model_dt_gini.y_test),
       Plot_ROC_AUC(target, model_dt_entropy.model_name, model_dt_entropy.model, model_dt_entropy.X_test, model_dt_entropy.y_test),
       Plot_ROC_AUC(target, model_svm_linear.model_name, model_svm_linear.model, model_svm_linear.X_test, model_svm_linear.y_test),
       Plot_ROC_AUC(target, model_svm_rbf.model_name, model_svm_rbf.model, model_svm_rbf.X_test, model_svm_rbf.y_test),
       Plot_ROC_AUC(target, model_nb.model_name, model_nb.model, model_nb.X_test, model_nb.y_test),
       Plot_ROC_AUC(target, model_log.model_name, model_log.model, model_log.X_test, model_log.y_test),
       Plot_ROC_AUC(target, model_gb.model_name, model_gb.model, model_gb.X_test, model_gb.y_test),
       Plot_ROC_AUC(target, model_xgb.model_name, model_xgb.model, model_xgb.X_test, model_xgb.y_test),
       Plot_ROC_AUC(target, model_rf.model_name, model_rf.model_k_features, model_rf.newX_test, model_rf.y_test),
       Plot_ROC_AUC(target, model_knn.model_name, model_knn.model, model_knn.X_test, model_knn.y_test),
       ]

# display all ROC plots
Plot_ROC_Combined(lst)

########################################################################
########################################################################
##======================================================================
# MODEL ITERATION PART - ended above
##======================================================================
########################################################################
########################################################################


# ============================================
# PART 7: Post-Training Analysis
# ============================================

## step1. classificaiton report to check class balance
Model_Report(target, model_gb.model_name, model_gb.y_test, y_pred_gb)

## step 2. VIF
features = df_5yr_preselect41.drop(['OTHBLEED'], axis=1)
Calc_Plot_VIF(features, 41)
df_vif = Calc_Plot_VIF(features, 41)

## step 3. corr coef
Calc_Top_Corr(features, 40)
Plot_Heatmap_Top_Corr(features, .5, 'Top Correlation Feature Pairs - Threshold = 0.5')

## step 4. feature importance
### step 4.1. f_importances
# from model_rf
Plot_Feature_Importances(model_rf.f_importances[-40:])

model_rf.f_importances.plot(y='Features', x='Importance', kind='barh', figsize=(16, 9), fontsize=6)
plt.xlabel('Feature Importances')
plt.title('Feature Importances (126) - Random Forest')
plt.tight_layout()
plt.show()

model_rf.f_importances[-40:].plot(y='Features', x='Importance', kind='barh', figsize=(16, 9), fontsize=6)
plt.xlabel('Feature Importances')
plt.title('Feature Importances TOP 40 - Random Forest')
plt.tight_layout()
plt.show()

# ==============================================
# post-training analysis --- feature importance
# ==============================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

df = df_5yr_preselect41_no_year
random_state = 100
test_size = .25
target = 'OTHBLEED'
k_folds = 10
n_estimators = 300
learning_rate = 0.05

X = df.drop([target], axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# class_names = df[target].unique()
# feature_names = df.drop([target], axis=1).columns.to_list()

# creating the classifier object
model_gb_shap = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state,
                                        learning_rate=learning_rate)
model_gb_shap.fit(X_train, y_train)

### step 4.2. f_importances
#1) from model_gb - iteration 3
# get feature importances
importances = model_gb.model.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, df.iloc[:, 1:].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=True, inplace=True)

# plot
# f_importances.plot(y='Features', x='Importance', kind='barh', figsize=(16, 9), fontsize=6)
# plt.xlabel('Feature Importances')
# plt.title('Feature Importances (126) - Gradient Boosting')
# plt.tight_layout()
# plt.show()

f_importances[-20:].plot(y='Features', x='Importance', kind='barh', figsize=(16, 9), fontsize=6)
plt.xlabel('Feature Importances')
plt.title('Feature Importances TOP 20 - Gradient Boosting')
plt.show()

### SHAP
import shap
explainer = shap.Explainer(model_gb_shap)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Bar plot
shap.plots.bar(shap_values)

# Beeswarm plot
shap.plots.beeswarm(shap_values, max_display=20)

# Dependence plot - 1 feature
shap.plots.scatter(shap_values[:, 'OPTIME'])
shap.plots.scatter(shap_values[:, "OPTIME"], color=shap_values)

# Dependence plot - 2 features
shap.plots.scatter(shap_values[:, 'feature1'],
                   color=shap_values[:, 'feature2'])

shap.summary_plot(shap_values, X_test)
shap.decision_plot(explainer.expected_value[0], shap_values[0], X_test.columns)


#2) from model_gb - iteration 7 synthetic data generation
# get feature importances

importances = model_gb.model.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, df.iloc[:, 1:].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=True, inplace=True)

# plot
# f_importances.plot(y='Features', x='Importance', kind='barh', figsize=(16, 9), fontsize=6)
# plt.xlabel('Feature Importances')
# plt.title('Feature Importances (126) - Gradient Boosting')
# plt.tight_layout()
# plt.show()

f_importances[-20:].plot(y='Features', x='Importance', kind='barh', figsize=(16, 9), fontsize=6)
plt.xlabel('Feature Importances')
plt.title('Feature Importances TOP 20 - Gradient Boosting')
plt.show()

### SHAP on synthesizer bayesian
df_syn_bay = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_synthetic_Bayesian.csv')
df_syn_bay = df_syn_bay.drop(['PUFYEAR'], axis=1)

random_state = 100
test_size = .25
target = 'OTHBLEED'
n_estimators = 300
learning_rate = 0.05

# build the model

X = df.drop([target], axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# class_names = df[target].unique()
# feature_names = df.drop([target], axis=1).columns.to_list()

# creating the classifier object
model_gb_syn_bay_shap = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state,
                                        learning_rate=learning_rate)
model_gb_syn_bay_shap.fit(X_train, y_train)
#
# # calculate needed metrics
# y_pred_gb = Model_Predict(target, model_gb.model_name, model_gb.model, model_gb.X_test)
# accuracy_gb = Model_Accuracy(target, model_gb.model_name, random_state, model_gb.model, model_gb.y_test, y_pred_gb)
# mean_accuracy_gb = Model_Mean_Accuracy(target, model_gb.model_name, k_folds, random_state, model_gb.model, model_gb.X, model_gb.y)
# rmse_gb = Model_RMSE(target, model_gb.model_name, model_gb.y_test, y_pred_gb)
# f1_gb = Model_F1(target, model_gb.model_name, model_gb.y_test, y_pred_gb)
# auc_gb = Model_ROC_AUC_Score(target, model_gb.model_name, model_gb.model, model_gb.X_test, model_gb.y_test)
# # results table
# results_gb = Model_Results_Table(model_gb.model_name, model_gb.parameters, target, test_size, accuracy_gb, mean_accuracy_gb, k_folds, rmse_gb, f1_gb, auc_gb)

import shap
explainer_syn_bay = shap.Explainer(model_gb_syn_bay_shap)
shap_values_syn_bay = explainer_syn_bay(X_test)
shap.summary_plot(shap_values_syn_bay, X_test, plot_type='bar')

# Beeswarm plot
shap.plots.beeswarm(shap_values_syn_bay, max_display=20)

# Dependence plot - 1 feature
shap.plots.scatter(shap_values_syn_bay[:, 'OPTIME'])

# Dependence plot - 2 features
shap.plots.scatter(shap_values_syn_bay[:, 'feature1'],
                   color=shap_values_syn_bay[:, 'feature2'])

# ==============================================
# PART 8 FNNs
# ==============================================
# references: tensorflow documentation

"""## Import tensorflow and libraries"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


"""# Helper Functions"""

# convert dataframe to tf.data
def dataframe_to_dataset(dataframe, target):
  '''
  convert pandas dataframe to tf.data
  '''
  dataframe = dataframe.copy()
  if multi_class==False:
    labels = dataframe.pop(target)
  else:
    out = dataframe.pop(target).values
    labels = np.concatenate(out).ravel().reshape(dataframe.shape[0], -1)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  ds = ds.shuffle(buffer_size=len(dataframe))
  return ds

# FNN functions
def build_model_5_layer(output_func, n):
  '''
  build 5-layer binary classification model
  '''
  encoded_features = feature_space.get_encoded_features()
  x = keras.layers.Dense(32, activation="relu")(encoded_features)
  x = keras.layers.Dense(50, activation="relu")(x)
  x = keras.layers.Dropout(0.2)(x)
  predictions = keras.layers.Dense(n, activation=output_func)(x)
  model = keras.Model(inputs=encoded_features, outputs=predictions)

  return model


def build_model_multi_layer(output_func, n):
  '''
  build 7-layer binary classification model with more neurons
  '''
  encoded_features = feature_space.get_encoded_features()
  x = keras.layers.Dense(32, activation="relu")(encoded_features)
  x = keras.layers.Dense(50, activation="relu")(x)
  x = keras.layers.Dense(100, activation="relu")(x)
  x = keras.layers.Dense(50, activation="relu")(x)
  x = keras.layers.Dropout(0.2)(x)
  predictions = keras.layers.Dense(n, activation=output_func)(x)
  model = keras.Model(inputs=encoded_features, outputs=predictions)

  return model


def compile_model(model, opt, loss_func):
  '''
  complie model using the opitimizer and loss function of choice
  '''

  model.compile(
      optimizer=opt,
      loss="binary_crossentropy",
      metrics=['accuracy',
              tf.keras.metrics.F1Score(average='macro', threshold=0.5),
              tf.keras.metrics.RootMeanSquaredError(),
              tf.keras.metrics.AUC()])
  return model

# create lists to save best scores
best_val_acc = []
best_val_f1 = []
best_val_rmse = []
best_val_auc = []

# save best scores from training history
def save_best_scores():
  best_val_acc.append(max(history.history['val_accuracy']))
  best_val_f1.append(max(history.history['val_f1_score']))
  best_val_rmse.append(max(history.history['val_root_mean_squared_error']))
  best_val_auc.append(max(history.history['val_auc'+num]))

# encode target for multi-class modeling (for softmax)
def encode_target(Y):
  encoder = LabelEncoder()
  encoder.fit(Y)
  encoded_Y = encoder.transform(Y)
  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_y = to_categorical(encoded_Y)

  return dummy_y

"""# Hyperparamers"""

dataset_name = '8_year_data'
#dataset_name = 'realtabformer'
#dataset_name = 'datasynthesizer'
multi_class = True
batch_size = 32
train_val_split_ratio = 0.2

if multi_class==False:
  target = 'OTHBLEED'
  output_func='sigmoid'
  n=1
  loss_func="binary_crossentropy"

else:
  target = 'OTHBLEED_new'
  output_func='softmax'
  n=2
  loss_func="categorical_crossentropy"

"""# Data Preprocessing"""

# import data
df = pd.read_csv('/processed_data/2015_2022/CABG_8yr_preselect41.csv')
#df = pd.read_csv('/processed_data/2015_2022/realtabformer_data.csv')
#df = pd.read_csv('/processed_data/2015_2022/CABG_synthetic_Bayesian_8yr.csv')
print(df.head())
print(df.shape)

if multi_class==False:
  # convert target dtype to float32 for computing f1 score
  df['OTHBLEED'] = df['OTHBLEED'].astype(np.float32)

else:
  dummy_y = encode_target(df['OTHBLEED'])
  df['OTHBLEED_new'] = pd.Series(list(dummy_y))
  df = df.drop('OTHBLEED',axis=1)

# split into training and test data
val_dataframe = df.sample(frac=train_val_split_ratio, random_state=1337)
train_dataframe = df.drop(val_dataframe.index)

print(f"Using {len(train_dataframe)} samples for training and {len(val_dataframe)} for validation")


# convert dataframe to tf.data
train_ds = dataframe_to_dataset(train_dataframe, target)
val_ds = dataframe_to_dataset(val_dataframe, target)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

# batch
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# set up feature space
feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers (25)
        "PUFYEAR": "integer_categorical",
        "SEX": "integer_categorical",
        "RACE_NEW": "integer_categorical",
        "INOUT": "integer_categorical",
        "ANESTHES": "integer_categorical",
        "DIABETES": "integer_categorical",
        "SMOKE": "integer_categorical",
        "DYSPNEA": "integer_categorical",
        "FNSTATUS2": "integer_categorical",
        "VENTILAT": "integer_categorical",
        "HXCOPD": "integer_categorical",
        "ASCITES": "integer_categorical",
        "HXCHF": "integer_categorical",
        "HYPERMED": "integer_categorical",
        "RENAFAIL": "integer_categorical",
        "DIALYSIS": "integer_categorical",
        "DISCANCR": "integer_categorical",
        "WNDINF": "integer_categorical",
        "STEROID": "integer_categorical",
        "WTLOSS": "integer_categorical",
        "BLEEDIS": "integer_categorical",
        "TRANSFUS": "integer_categorical",
        "EMERGNCY": "integer_categorical",
        "ASACLAS": "integer_categorical",
        "OTHERCPT1": "integer_categorical",


        # Numerical features (16)
        "AGE": "float",
        "BMI": "float",
        "PRSODM": "float",
        "PRBUN": "float",
        "PRCREAT": "float",
        "PRALBUM": "float",
        "PRBILI": "float",
        "PRSGOT": "float",
        "PRALKPH": "float",
        "PRWBC": "float",
        "PRHCT": "float",
        "PRPLATE": "float",
        "PRPTT": "float",
        "PRINR": "float",
        "OPTIME": "float",
        "TOTHLOS": "float"
    },

    # one-hot encode all categorical features and concat all features into a single vector (one vector per sample).
    output_mode="concat",
)

# adapt the features (transform features based on feature space configuration)
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

# check feature size and dtype
for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)

# create train and val dataset and prefetch
preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

"""# Training

## 1. FNN with 5 layers, Optimizer = SGD
"""

# build and compile model
model = build_model_5_layer(output_func,n)
model = compile_model(model, opt='sgd',loss_func=loss_func)

# model summary
model.summary()

# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# create a figure to show loss and performance over training time
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

"""## 2. FNN with 5 layers, Optimizer = Adam"""

# build and compile model
model = build_model_5_layer(output_func=output_func,n=n)
model = compile_model(model, opt='adam',loss_func=loss_func)

# model summary
model.summary()


# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

"""## 3. FNN with 10 layers, more neurons, Optimizer = SGD"""

# build and compile model
model = build_model_multi_layer(output_func, n)
model = compile_model(model, opt='sgd',loss_func=loss_func)

# model summary
model.summary()


# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

"""## 4. FNN with 10 layers, more neurons, Optimizer = Adam"""

# build and compile model
model = build_model_multi_layer(output_func,n)
model = compile_model(model, opt='adam',loss_func=loss_func)

# model summary
model.summary()


# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

# best scores summary from all models
summary = pd.DataFrame.from_dict(
    {'Model':['FNN-5layer-SGD-'+output_func,'FNN-5layer-Adam-'+output_func,'FNN-multilayer-SGD-'+output_func,'FNN-multilayer-Adam-'+output_func],
     'accuracy':best_val_acc,
     'f1_score':best_val_f1,
     'rMSE':best_val_rmse,
     'AUC':best_val_auc}
                       )
print(summary)

#summary.to_csv('/output/summary_'+output_func+'_'+dataset_name+'.csv')

# ==============================================
# PART 9 CNNs
# ==============================================
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# set-up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# hyper parameters
LR = 5e-2
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.5

# accuracy function
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

# CNN class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 3))  # output (n_examples, 16, 26, 26)
        self.convnorm1 = nn.BatchNorm2d(16)
       # self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 13, 13)
        self.conv2 = nn.Conv2d(16, 32, (1, 3))  # output (n_examples, 32, 11, 11)
        self.convnorm2 = nn.BatchNorm2d(32)
    #    self.pool2 = nn.AvgPool2d((2, 2))  # output (n_examples, 32, 5, 5)
        self.linear1 = nn.Linear(1184, 400)  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(400, 2)
        self.act = torch.relu

    def forward(self, x):
        x = self.convnorm1(self.act(self.conv1(x)))
        x = self.convnorm2(self.act(self.conv2(x)))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)

# data preprocessing

# IMPORTANT - use each of the 3 datasets below and repeat the code below
df = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2015_2022/realtabformer_data.csv')
# df = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2018_2022/CABG_5yr_preselect41.csv')
# df = pd.read_csv('https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/processed_data/2015_2022/CABG_synthetic_Bayesian_8yr.csv')

random_state = 100
target = 'OTHBLEED'

# splitting the data - traning (70%), validation (15%), test (15%)

# Divide the data into training (70%) and test (30%)
df_train, df_test = train_test_split(df,
                                     train_size=0.7,
                                     random_state=random_state,
                                     stratify=df[target])

# Divide the test data into validation (50%) and test (50%)
df_val, df_test = train_test_split(df_test,
                                   train_size=0.5,
                                   random_state=random_state,
                                   stratify=df_test[target])

# Reset the index
df_train, df_val, df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])
# Print the dimension of df_val
pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns'])
# Print the dimension of df_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

# Splitting the feature and target
X_train = df_train[np.setdiff1d(df_train.columns, [target])].values
X_val = df_val[np.setdiff1d(df_val.columns, [target])].values
X_test = df_test[np.setdiff1d(df_test.columns, [target])].values

y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

# check original shape
X_train.shape

# Converts to Tensors and reshapes to suitable image shape

# train dataset
X_train = torch.Tensor(X_train).view(X_train.shape[0], 1, -1, X_train.shape[1]).float().to(device)
X_train.requires_grad = True
y_train = torch.LongTensor(y_train).to(device)

# test dataset
X_test = torch.Tensor(X_test).view(X_test.shape[0], 1, -1, X_test.shape[1]).float().to(device)
y_test = torch.LongTensor(y_test).to(device)

# check shape again after reshaping
X_train.shape

# Training Prep
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training loop
from inspect import indentsize
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    for batch in range(len(X_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(X_train[inds])

        print(logits)
        print(y_train[inds])

        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train/BATCH_SIZE, acc(X_train, y_train), loss_test, acc(X_test, y_test)))