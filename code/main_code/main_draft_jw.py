# import packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# import the baseline models class
# get py file folder path
sys.path.append(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\component')

from class_decision_tree import DecisionTree
from class_svm import SVM
from class_logistic import LogReg
from class_gradient_boosting import GradientBoosting
from class_xgboost import XGB
from class_gaussian_nb import NaiveBayesGaussianNB
from class_random_forest import RandomForest



df = pd.read_csv(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\main_code\processed_data\CABG_preselect.csv')
df_features = df.drop(['OTHBLEED'], axis=1)

from class_decision_tree import DecisionTree

# import utilities
from utils_models import *

# adjust output window size

#np.set_printoptions(linewidth=desired_width)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

# os.chdir(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code')
# print('Current working directory:', os.getcwd())

# ===================================================
# STEP1. Data cleaning and pre-processing         ||
# ===================================================
from class_datasci import datasci

# read in original data
# df =

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

# skipping Step 1 data preprocessing - import clean data here below
df = pd.read_csv(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\main_code\processed_data\CABG_preselect.csv')
# df = df.drop(['PUFYEAR'], axis=1)

#===============================================
# Model 1: call Decision Tree model class
#===============================================

# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5; criterion = 'gini', max_depth = 3, min_samples_leaf = 5
model_decision_tree = DecisionTree(df,'OTHBLEED',0.3,100, 5,'gini', 3, 5)

# run Decision Tree model and display results
model_decision_tree.Predict()
model_decision_tree.Report()
model_decision_tree.Accuracy()
model_decision_tree.RMSE()
model_decision_tree.Confusion_Matrix()
model_decision_tree.Confusion_Matrix_Plot()
model_decision_tree.Decision_Tree_Plot()
model_decision_tree.ROC_AUC_Score()
model_decision_tree.ROC_AUC_Plot()
print(model_decision_tree.Display_Model_Results())

#===============================================
# Model 2: call Decision Tree model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5; criterion = 'entropy', max_depth = 3, min_samples_leaf = 5
model_decision_tree2 = DecisionTree(df,'OTHBLEED',0.3,100, 5,'entropy', 3, 5)
print(model_decision_tree2.Display_Model_Results())
model_decision_tree2.ROC_AUC_Plot()

#===============================================
# Model 3: call SVM model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5; kernel='linear', C=1.0, gamma='auto'
model_svm = SVM(df,'OTHBLEED',0.3,100, 5,'linear',1.0,'auto')
print(model_svm.Display_Model_Results())
model_svm.ROC_AUC_Score()
model_svm.ROC_AUC_Plot()
model_svm.Confusion_Matrix()

#===============================================
# Model 4: call SVM model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5; kernel='rbf', C=1.0, gamma=0.2
model_svm2 = SVM(df,'OTHBLEED',0.3,100, 5,'rbf',1.0,0.2)
print(model_svm2.Display_Model_Results())
model_svm2.ROC_AUC_Score()
model_svm2.ROC_AUC_Plot()

#===============================================
# Model 5: call Logistic Regression model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5
model_log = LogReg(df,'OTHBLEED',0.3,100, 5)
print(model_log.Display_Model_Results())
model_log.ROC_AUC_Score()
model_log.ROC_AUC_Plot()

#===============================================
# Model 6: call GradientBoosting model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5; n_estimators=300, learning_rate=0.05
model_gb = GradientBoosting(df,'OTHBLEED',0.3,100, 5,300, 0.05)
print(model_gb.Display_Model_Results())
model_gb.ROC_AUC_Score()
model_gb.ROC_AUC_Plot()

#===============================================
# Model 7: call XGBoost model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5; n_estimators=100, eta=0.3
model_xgb = XGB(df,'OTHBLEED',0.3,100, 5,100, 0.3)
print(model_xgb.Display_Model_Results())
model_xgb.ROC_AUC_Score()
model_xgb.ROC_AUC_Plot()

#===============================================
# Model 8: call GaussianNB model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5
model_nb = NaiveBayesGaussianNB(df,'OTHBLEED',0.3,100, 5)
print(model_nb.Display_Model_Results())
model_nb.ROC_AUC_Score()
model_nb.ROC_AUC_Plot()

#===============================================
# Model 9: call RandomForest model class
#===============================================
# initiate the model
# 0.3 on test size; random_state = 100; k_folds = 5, n_estimators=100, feature_importances = 20
model_rf = RandomForest(df,'OTHBLEED',0.3,100, 5, 100, 40)
print(model_rf.Display_Model_Results())
model_rf.Random_Forest_Feature_Importances_Plot()
model_rf.ROC_AUC_Score()
model_rf.ROC_AUC_Plot()


# ===================================================
# Display all the model results to compare
# ===================================================

# first, build all the models
model_decision_tree = DecisionTree(df,'OTHBLEED',0.3,100, 5,'gini', 3, 5)
model_decision_tree2 = DecisionTree(df,'OTHBLEED',0.3,100, 5,'entropy', 3, 5)
model_svm = SVM(df,'OTHBLEED',0.3,100, 5,'linear',1.0,'auto')
model_svm2 = SVM(df,'OTHBLEED',0.3,100, 5,'rbf',1.0,0.2)
model_log = LogReg(df,'OTHBLEED',0.3,100, 5)
model_gb = GradientBoosting(df,'OTHBLEED',0.3,100, 5,300, 0.05)
model_xgb = XGB(df,'OTHBLEED',0.3,100, 5,100, 0.3)
model_nb = NaiveBayesGaussianNB(df,'OTHBLEED',0.3,100, 5)
model_rf = RandomForest(df,'OTHBLEED',0.3,100, 5, 100, 20)

# get all the result metrics
a=model_decision_tree.Display_Model_Results()
b=model_decision_tree2.Display_Model_Results()
c=model_svm.Display_Model_Results()
d=model_svm2.Display_Model_Results()
e=model_nb.Display_Model_Results()
f=model_log.Display_Model_Results()
g=model_gb.Display_Model_Results()
h=model_xgb.Display_Model_Results()
i=model_rf.Display_Model_Results()

print(Combined_Results_Table([a,b,c,d,e,f,g,h,i]))

# ===================================================
# Plot combined ROC plots
# ===================================================
# get all the ROC variables
lst = [model_decision_tree.ROC_AUC_Plot(),
       model_svm.ROC_AUC_Plot(),
       model_svm2.ROC_AUC_Plot(),
       model_nb.ROC_AUC_Plot(),
       model_log.ROC_AUC_Plot(),
       model_gb.ROC_AUC_Plot(),
       model_xgb.ROC_AUC_Plot(),
       model_nb.ROC_AUC_Plot(),
       model_rf.ROC_AUC_Plot(),
       ]

# display all ROC plots
Combined_ROC_Plot(lst)

# =====================================
# STEP3. Feature Selection           ||
# =====================================
# import sys
# # import the baseline models class
# # get py file folder path
# sys.path.append(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\component')
#
# import pandas as pd
from class_pca import PCA_for_Feature_Selection
#
# df = pd.read_csv(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\main_code\processed_data\CABG_preselect.csv')


# import PCA class
pca_module = PCA_for_Feature_Selection(df, 'OTHBLEED')

# how many features suggested to reduce
pca_module.PCA_Reduced_Feature()

pca_module.Explained_Variance_Ratio()
pca_module.Reduced_Feature_Space_Plot()
pca_module.Reduced_Feature_Space_Heatmap()

# get new df to feed into the models
df_new = pca_module.PCA_New_df()

# add the target back
print(df_new.shape)
df_new['OTHBLEED'] = df['OTHBLEED'].values

# =============================
# display new results after PAC
# =============================

# first, build all the models
model_decision_tree_new = DecisionTree(df_new,'OTHBLEED',0.3,100, 5,'gini', 3, 5)
model_decision_tree2_new = DecisionTree(df_new,'OTHBLEED',0.3,100, 5,'entropy', 3, 5)
model_svm_new = SVM(df_new,'OTHBLEED',0.3,100, 5,'linear',1.0,'auto')
model_svm2_new = SVM(df_new,'OTHBLEED',0.3,100, 5,'rbf',1.0,0.2)
model_log_new = LogReg(df_new,'OTHBLEED',0.3,100, 5)
model_gb_new = GradientBoosting(df_new,'OTHBLEED',0.3,100, 5,300, 0.05)
model_xgb_new = XGB(df_new,'OTHBLEED',0.3,100, 5,100, 0.3)
model_nb_new = NaiveBayesGaussianNB(df_new,'OTHBLEED',0.3,100, 5)
model_rf_new = RandomForest(df_new,'OTHBLEED',0.3,100, 5, 100, 20)

# get all the result metrics
a2=model_decision_tree_new.Display_Model_Results()
b2=model_decision_tree2_new.Display_Model_Results()
c2=model_svm_new.Display_Model_Results()
d2=model_svm2_new.Display_Model_Results()
e2=model_nb_new.Display_Model_Results()
f2=model_log_new.Display_Model_Results()
g2=model_gb_new.Display_Model_Results()
h2=model_xgb_new.Display_Model_Results()
i2=model_rf_new.Display_Model_Results()

# display results comparison
print(Combined_Results_Table([a2,b2,c2,d2,e2,f2,g2,h2,i2]))

