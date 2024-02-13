import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Basic_Classification_Models import BasicClassificationModels

# import clean data
df=pd.read_csv(r'https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/CABG_20_recoded.csv',index_col=0)
df=df.drop(['PUFYEAR'], axis=1)

# pull the models from BasicClassificationModels class
data = BasicClassificationModels(df,'OTHBLEED',0.3,100, 5)

# run different models and compare results
data.Decision_Tree('gini',3,5)
data.Decision_Tree('entropy',3,5)
data.SVM('linear',1.0,0)
data.SVM('rbf',1.0,0.2)
data.LogReg()
data.XGBoost(100, 0.3)