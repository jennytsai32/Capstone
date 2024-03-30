# import packages

import pandas as pd
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
# STEP1. Data cleaning and pre-processing         ||
# ===================================================

# ==============================================================================
# STEP2. Baseline models and results (before feature selection)               ||
# ==============================================================================

# skipping Step 1 data preprocessing - import clean data here below
# df=pd.read_csv(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\main_code\processed_data\2018_2020\CABG_preselect.csv')

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


########################################################################
########################################################################
##======================================================================
# MODEL ITERATION PART - starting below
##======================================================================
########################################################################
########################################################################


# set cross-cutting variables
df = df_5yr_preselect41_no_year
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

# Post-Training Analysis

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
### 4.1. f_importances
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

# ================================
# post-training analysis --- feature importance
# ================================
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

### 4.2. f_importances
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

### 4.2. SHAP
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

### 4.2. SHAP
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


### 4.3. LIME
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(model_gb.X_train.values, feature_names=model_gb.feature_names,
                                                  class_names=['No', 'Transfusions'], verbose=True, mode='regression')
# Choose the 5th instance and use it to predict the results
j = 150
exp = explainer.explain_instance(model_gb.X_test.values[j], model_gb.model.predict, num_features=20)

# Show the predictions
# exp.show_in_notebook(show_table=True)
exp.save_to_file(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\lime.html')

# =====================================
# STEP3. Feature Selection           ||
# =====================================
# import sys
# # import the baseline models class
# # get py file folder path
# sys.path.append(r'C:\Users\wb576802\Documents\non-work\GWU\Capstone\Github folders\Capstone\code\component')
#
# import pandas as pd

# 1. VIF check
features40 = df_5yr_preselect40.drop(['PUFYEAR'], axis=1)
Calc_Plot_VIF(features40, 40)
Calc_Top_Corr(features40, 20)
Plot_Heatmap_Top_Corr(features40, .5, 'Top Correlation Feature Pairs - Threshold = 0.5')

# 2. check feature importances from random forest

# 3. PCA
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

# =============================
# display new results after PAC
# =============================



# =====================================
# STEP4. AutoML - TPOT               ||
# =====================================
from class_tpot import TPOT
# initiate the model
# 0.25 on test size, random_state = 100, generations=5, population_size=20, verbosity=2
df = df_5yr_preselect40
model_tpot = TPOT(df, target, test_size, random_state, 5, 20, 2)
model_tpot.scores


