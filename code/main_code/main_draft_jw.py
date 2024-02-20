import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

#np.set_printoptions(linewidth=desired_width)

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

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



# ==============================================================================
# STEP2. Baseline models and results (before feature selection)               ||
# ==============================================================================

# import the baseline models class
from class_baseline_decision_tree import DecisionTree
from class_baseline_svm import SVM
from class_baseline_logistic import LogReg
from class_baseline_xgboost import XGB

# ALTERNATIVE - import clean data
df = pd.read_csv(r'https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/CABG_20_recoded.csv',index_col=0)
df = df.drop(['PUFYEAR','AGE','BMI'], axis=1)

#===============================================
# Model 1: call Decision Tree model class
#===============================================

# initiate the model
model_decision_tree = DecisionTree(df,'OTHBLEED',0.3,100, 5,'gini', 3, 5)

# run Decision Tree model and display results
model_decision_tree.Predict()
model_decision_tree.Report()
model_decision_tree.Accuracy()
model_decision_tree.RMSE()
model_decision_tree.Confusion_Matrix()
model_decision_tree.Confusion_Matrix_Plot()
model_decision_tree.Decision_Tree_Plot()
model_decision_tree.ROC_AUC_Plot()
model_decision_tree.Display_Model_Results()

#===============================================
# Model 2: call Decision Tree model class
#===============================================
# initiate the model
model_decision_tree2 = DecisionTree(df,'OTHBLEED',0.3,100, 5,'entropy', 3, 5)
model_decision_tree2.Display_Model_Results()

#===============================================
# Model 3: call SVM model class
#===============================================
model_SVM = SVM(df,'OTHBLEED',0.3,100, 5,'linear',1.0,0)
model_SVM.Display_Model_Results()

#===============================================
# Model 4: call SVM model class
#===============================================
model_SVM2 = SVM(df,'OTHBLEED',0.3,100, 5,'rbf',1.0,0.2)
model_SVM2.Display_Model_Results()

#===============================================
# Model 5: call Logistic Regression model class
#===============================================
model_log = LogReg(df,'OTHBLEED',0.3,100, 5)
model_log.Display_Model_Results()

#===============================================
# Model 6: call XGBoost model class
#===============================================
model_xgb = XGB(df,'OTHBLEED',0.3,100, 5,100, 0.3)
model_xgb.Display_Model_Results()
model_xgb.Accuracy()



from utils_models import *

a= model_decision_tree.Display_Model_Results()
b=model_SVM.Display_Model_Results()
c=model_SVM2.Display_Model_Results()

Results_Table([a,b,c])


