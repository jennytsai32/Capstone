# build a class of basic classification models for our baseline model construction
# including Decision Tree, SVM, LogisticRegression, and XGBoost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, mean_squared_error, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


class SVM:

    def __init__(self, df, target, test_size, random_state, k_folds, kernel, C, gamma): # kernel can be linder or rbf; C=1.0; gamma=0.2 (for rbf)
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.k_folds = k_folds
        self.model_name = 'SVM - ' + kernel
        self.parameters = 'C=' + str(C) + ', gamma=' + str(gamma)

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[0]
        self.X_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[1]
        self.y_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[2]
        self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[3]

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns

        # get y_pred
        self.model = SVC(kernel=kernel, C=C, random_state=self.random_state, gamma=gamma)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        # build a model results table
        self.results_table = pd.DataFrame(columns=['Model Name', 'Parameters', 'Target', 'Mean Accuracy ('+str(self.k_folds)+' folds)', 'RMSE', 'F1-score'])

    def Predict(self):
        self.model.fit(self.X_train, self.y_train)

        # # make predictions
        # y_pred = self.model.predict(self.X_test)

        # display results
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print(f'Y-test prediction: {self.y_pred}')


    def Report(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Report: ' + '\n',
              classification_report(self.y_test, self.y_pred))

    def Accuracy(self):
        # perform cross validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print(f'Model Accuracy (mean of {self.k_folds} folds): {mean_accuracy}')


    def RMSE(self):
        rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print(f'RMSE: {rmse: .3f}')

    def Confusion_Matrix(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Confusion Matrix: ' + '\n', confusion_matrix(self.y_test, self.y_pred))

    def Confusion_Matrix_Plot(self):
        print('Plot confusion matrix: ')
        print('Target: ', self.target)
        print('Model: ', self.model_name)

        # plot confusion matrix
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        df_conf_matrix = pd.DataFrame(conf_matrix, index=self.class_names, columns=self.class_names)

        plt.figure(figsize=(10, 10))
        hm = sns.heatmap(df_conf_matrix, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_conf_matrix.columns, xticklabels=df_conf_matrix.columns)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.tight_layout()
        plt.show()

    def Decision_Tree_Plot(self):
        # plot the tree
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Plot the decision tree: ')
        plt.figure(figsize=(15, 10))
        tree.plot_tree(self.model, filled=True, feature_names=self.feature_names, class_names=['Transfusions','No'], rounded=True, fontsize=14)
        plt.show()

    def ROC_AUC_Plot(self):
        # Plot ROC Area Under Curve
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Plot ROC Aarea Under Curve: ')

        y_pred_proba = self.model.decision_function(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc = roc_auc_score(self.y_test, y_pred_proba)

        plt.figure(figsize=(10, 10))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def Display_Model_Results(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Model Results:' + '\n')
        report = classification_report(self.y_test, self.y_pred, output_dict=True)

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)

        f1 = report['macro avg']['f1-score']
        self.results_table.loc[len(self.results_table.index)] = [self.model_name, self.parameters, self.target, mean_accuracy, rmse, f1]

        return self.results_table