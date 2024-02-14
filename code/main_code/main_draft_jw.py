import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# os.chdir(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code')
# print('Current working directory:', os.getcwd())

sys.path.append(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\component')

# ===================================================
# STEP1. Data cleaning and pre-processing         ||
# ===================================================
from class_datasci import datasci

# read in original data
df = 

# check the data
size(df)
missingReport(df)

# missing data imputation
imputation(df)

# encoding
recode(df)

# perform EDA
eda(df)

# standardize data
standardize(df)


# ALTERNATIVE - import clean data
df = pd.read_csv(r'https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/CABG_20_recoded.csv',index_col=0)
df = df.drop(['PUFYEAR'], axis=1)

# ==============================================================================
# STEP2. Baseline models and results (before feature selection)               ||
# ==============================================================================

from class_baseline_classification_models import BaselineClassificationModels

# pull the models from BasicClassificationModels class
data = BaselineClassificationModels(df,'OTHBLEED',0.3,100, 5)

# run different models and compare results
data.Decision_Tree('gini',3,5)
data.Decision_Tree('entropy',3,5)
data.SVM('linear',1.0,0)
data.SVM('rbf',1.0,0.2)
data.LogReg()
data.XGBoost(100, 0.3)