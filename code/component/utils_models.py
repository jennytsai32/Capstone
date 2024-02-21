# This utilities include needed functions to check model results
# Functions included:
# 1. Predict()
# 2. Report()
# 3. Accuracy()
# 4. RMSE()
# 5. Confusion_Matrix()
# 6. Confusion_Matrix_Plot()
# 7. ROC_AUC_Score()
# 8. ROC_AUC_Plot()
# 9. Combined_ROC_Plot()
# 10. Single_Model_Results()
# 11. Combined_Model_Results()
# 12. Decision_Tree_Plot()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, root_mean_squared_error, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


def Predict(target, model_name,y_pred):
    # display results
    print('Target: ', target)
    print('Model: ', model_name)
    print(f'Y-test prediction: {y_pred}')
    print('-' * 80)


def Report(target, model_name, y_test, y_pred):
    print('Target: ', target)
    print('Model: ', model_name)
    print('Report: ' + '\n',
          classification_report(y_test, y_pred))
    print('-' * 80)


def Accuracy(target, model_name, k_folds, random_state, model, X, y):
    # perform cross validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf)
    mean_accuracy = scores.mean() * 100
    print('Target: ', target)
    print('Model: ', model_name)
    print(f'Model Accuracy (mean of {k_folds} folds): {mean_accuracy}')
    print('-' * 80)


def RMSE(target, model_name, y_test, y_pred):
    rmse = root_mean_squared_error(y_test, y_pred)
    print('Target: ', target)
    print('Model: ', model_name)
    print(f'RMSE: {rmse: .3f}')
    print('-' * 80)


def Confusion_Matrix(target, model_name, y_test, y_pred):
    print('Target: ', target)
    print('Model: ', model_name)
    print('Confusion Matrix: ' + '\n', confusion_matrix(y_test, y_pred))
    print('-' * 80)


def Confusion_Matrix_Plot(target, model_name, y_test, y_pred, class_name):
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

def Decision_Tree_Plot(target, model_name, model, feature_names):
    # plot the tree
    print('Target: ', target)
    print('Model: ', model_name)
    print('Plot the decision tree: ')
    plt.figure(figsize=(15, 10))
    tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=['Transfusions', 'No'],
                   rounded=True, fontsize=14)
    plt.show()


def ROC_AUC_Score(target, model_name, model, X_test, y_test):
    print('Target: ', target)
    print('Model: ', model_name)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC-AUC score: {auc:.3f}')
    print('-' * 80)


def ROC_AUC_Plot(target, model_name, model, X_test, y_test):
    # Plot ROC Area Under Curve
    print('Target: ', target)
    print('Model: ', model_name)
    print('Plot ROC Aarea Under Curve: ')

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC=' + str(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Plot')
    plt.legend(loc="lower right")
    plt.show()

def Single_Model_Results(target, parameters, model_name, model, random_state, k_folds, X, y, y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf)
    mean_accuracy = scores.mean() * 100
    rmse = root_mean_squared_error(y_test, y_pred)

    f1 = report['macro avg']['f1-score']

    # ROC-AUC
    y_pred_proba = self.model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    # construct the table
    dict = {'Model Name': model_name,
            'Parameters': parameters,
            'Target': target,
            'Mean Accuracy (' + str(self.k_folds) + ' folds)': mean_accuracy,
            'RMSE': rmse,
            'F1-score': f1,
            'ROC-AUC score': auc}
    results_table = pd.DataFrame([dict])
    return results_table

def Combined_Results_Table(lst_df):
    results_table = pd.concat(lst_df, ignore_index=True)
    return results_table

def Combined_ROC_Plot(lst):
    plt.figure(figsize=(10, 10))

    for i in range(len(lst)):
        plt.plot(lst[i][0], lst[i][1], lw=2, label=f'AUC-{lst[i][3]}={lst[i][2]:.3f}')

    plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Plot')
    plt.legend(loc="lower right")
    plt.show()
