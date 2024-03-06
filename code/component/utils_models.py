
# =============================================================================
# This utilities.py includes needed functions to check model results
# =============================================================================

# 1. Model_Predict()
# 2. Model_Report()
# 3. Model_Accuracy()
# 4. Model_Mean_Accuracy()
# 5. Model_RMSE()
# 6. Model_F1()
# 7. Model_Confusion_Matrix()
# 8. Plot_Confusion_Matrix()
# 9. Plot_Decision_Tree()
# 10. Model_ROC_AUC_Score()
# 11. Plot_ROC_AUC()
# 12. Plot_Random_ForFeature_Importances
# 13. Model_Results_Table()
# 14. Plot_ROC_Combined()

# 15. Calc_Plot_VIF()
# 16. Calc_Top_Corr()
# 17. Plot_Heatmap_Top_Corr



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, mean_squared_error, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import tree

# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
# from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split

def Model_Predict(target, model_name, model, X_test):
    # display results
    y_pred = model.predict(X_test)
    print('-' * 80)
    print('Target: ', target)
    print('Model: ', model_name)
    print(f'Y-test prediction: {y_pred}')
    return y_pred

def Model_Report(target, model_name, y_test, y_pred):
    print('-' * 80)
    print('Target: ', target)
    print('Model: ', model_name)
    report = classification_report(y_test, y_pred)
    print('Report:\n', report)

def Model_Accuracy(target, model_name, random_state, model, y_test, y_pred):
    print('-' * 80)
    # accuracy score
    acc_score = accuracy_score(y_test, y_pred)*100
    print(f'Model Accuracy: {acc_score}')
    return acc_score

def Model_Mean_Accuracy(target, model_name, k_folds, random_state, model, X, y):
    print('-' * 80)
    # perform cross validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf)
    mean_accuracy = scores.mean() * 100
    print('Target: ', target)
    print('Model: ', model_name)
    print(f'Model Mean Accuracy (mean of {k_folds} folds): {mean_accuracy}')
    return mean_accuracy

def Model_RMSE(target, model_name, y_test, y_pred):
    print('-' * 80)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('Target: ', target)
    print('Model: ', model_name)
    print(f'RMSE: {rmse: .3f}')
    print('-' * 80)
    return rmse

def Model_F1(target, model_name, y_test, y_pred):
    print('-' * 80)
    print('Target: ', target)
    print('Model: ', model_name)
    # report = classification_report(y_test, y_pred)
    # f1 = report['macro avg']['f1-score']
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'F-1 score: {f1: .3f}')
    return f1

def Model_Confusion_Matrix(target, model_name, y_test, y_pred):
    print('-' * 80)
    print('Target: ', target)
    print('Model: ', model_name)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: ' + '\n', conf_matrix)
    return conf_matrix

def Plot_Confusion_Matrix(target, model_name, y_test, y_pred, class_names):
    print('-' * 80)
    print('Plot confusion matrix: ')
    print('Target: ', target)
    print('Model: ', model_name)

    # plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    df_conf_matrix = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(10, 10))
    hm = sns.heatmap(df_conf_matrix, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_conf_matrix.columns, xticklabels=df_conf_matrix.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()
    plt.show()

def Plot_Decision_Tree(target, model_name, model, feature_names):
    # plot the tree
    print('-' * 80)
    print('Target: ', target)
    print('Model: ', model_name)
    print('Plot the decision tree: ')
    plt.figure(figsize=(15, 10))
    tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=['No', 'Transfusions'],
                   rounded=True, fontsize=14)
    plt.show()

def Model_ROC_AUC_Score(target, model_name, model, X_test, y_test):
    print('-' * 80)
    print('Target: ', target)
    print('Model: ', model_name)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC-AUC score: {auc:.3f}')
    return auc

def Plot_ROC_AUC(target, model_name, model, X_test, y_test):
    # Plot ROC Area Under Curve
    print('Target: ', target)
    print('Model: ', model_name)
    print('Plot ROC Aarea Under Curve: ')

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC=' + str(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label=f'AUC-{target}: {auc:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Plot')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, auc, model_name

def Model_Results_Table(model_name_lst, parameters_lst, target, test_size, accuracy_lst, mean_accuracy_lst, k_folds, rmse_lst, f1_lst, auc_lst):

    # construct the table
    dict = {'Model Name': model_name_lst,
            'Parameters': parameters_lst,
            'Accuracy': accuracy_lst,
            'Mean Accuracy (' + str(k_folds) + ' folds)': mean_accuracy_lst,
            'RMSE': rmse_lst,
            'F1-score (macro avg)': f1_lst,
            'ROC-AUC score': auc_lst}
    results_table = pd.DataFrame([dict])
    # insert a column of target
    results_table.insert(2, 'Target','')
    results_table.insert(3, 'Test Size','')
    results_table['Target'] = target
    results_table['Test Size'] = test_size
    return results_table

def Plot_ROC_Combined(lst):
    plt.figure(figsize=(12, 12))

    for i in range(len(lst)):
        plt.plot(lst[i][0], lst[i][1], lw=2, label=f'AUC-{lst[i][3]}={lst[i][2]:.3f}')

    plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Plot - Comparison')
    plt.legend(loc="lower right")
    plt.show()

def Calc_Plot_VIF(df, n):   # n = num of features included
    df_vif = pd.DataFrame()
    df_vif["Features"] = df.columns
    # calculating VIF for each feature
    df_vif["VIF"] = [variance_inflation_factor(df.values, i)
                       for i in range(len(df.columns))]
    df_vif = df_vif.sort_values('VIF', ascending=False)[0:n]

    # plot
    df_vif.sort_values('VIF', ascending=True).plot(y='VIF', x='Features', kind='barh', figsize=(12, 6))
    plt.xlabel('Variance Inflation Factor')
    plt.ylabel('')
    plt.title(f'Feature VIF, features = {n}')
    plt.legend(loc='lower right')
    for index, value in enumerate(df_vif['VIF'].sort_values().round(1)):
        plt.text(value, index, str(value), ha='left', va='center')
    plt.show()

    return df_vif

def Calc_Top_Corr(df, n):    # n = # of top corr pair
    corr_matrix = df.corr()
    df_corr = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False))
    df_corr = pd.DataFrame(df_corr).reset_index()
    df_corr.columns=['Variable_1','Variable_2','Correlacion']
    df_sorted_corr = df_corr.reindex(df_corr.Correlacion.abs().sort_values(ascending=False).index).reset_index().drop(['index'],axis=1)
    return df_sorted_corr.head(n)


def Plot_Heatmap_Top_Corr(df, n, title):    #  n = corr coef threshold
    df_corr = df.corr()
    df_filtered = df_corr[((df_corr >= n) | (df_corr <= -n)) & (df_corr != 1)]
    plt.figure(figsize=(12, 8))

    sns.heatmap(df_filtered,
                annot=True, cmap='Reds', vmin=0, vmax=1, fmt='.1f',
                annot_kws={'color': 'yellow',
                           'ha': 'center',
                           # "bbox": {"facecolor": "yellow", "edgecolor": "black", "boxstyle": "round,pad=0.5"}   set background color
                           })
    plt.title(title)
    plt.show()

def Plot_Feature_Importances(df_f_importances):
    df_f_importances.plot(y='Features', x='Importance', kind='barh', figsize=(16, 9), fontsize=6)
    plt.xlabel('Feature Importances')
    plt.title('Feature Importances - Random Forest')
    plt.tight_layout()
    plt.show()
