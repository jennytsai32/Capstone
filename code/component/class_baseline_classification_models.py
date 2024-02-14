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


class BaselineClassificationModels:

    def __init__(self, df, target, test_size, random_state, k_folds):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.k_folds = k_folds
        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[0]
        self.X_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[1]
        self.y_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[2]
        self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[3]

        # build a model results table
        self.results_table = pd.DataFrame(columns=['Model Name', 'Target', 'Mean Accuracy (10 folds)', 'RMSE', 'F1-score'])

    def Decision_Tree(self, criterion, max_depth, min_samples_leaf):      # usually max_depth=3, min_samples_leaf=5
        model_name = 'Decision Tree - ' + criterion
        model = DecisionTreeClassifier(criterion=criterion, random_state=self.random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        model.fit(self.X_train, self.y_train)

        # make predictions
        y_pred = model.predict(self.X_test)

        # results
        print('='*80)
        print('Target: ', self.target)
        print('Model: ', model_name)
        print('-' * 80 + '\n')
        print('Report: ' + '\n',
              classification_report(self.y_test, y_pred))
        print('-' * 80)
        print('Model Accuracy (without cross validation: ' + '\n',
              accuracy_score(self.y_test, y_pred)*100, '\n')
        print('-' * 80)

        # perform cross validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        print('Cross Validation Results (Accuracy): ' + '\n', scores * 100)
        print('-' * 80)
        print('Mean Accuracy (after cross-validation: ' + '\n', mean_accuracy)
        print('-' * 80)

        # MSE
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        print(f'RMSE: {rmse: .3f}')
        print('-' * 80)

        # confusion matrix
        print('Confusion Matrix: ' + '\n', confusion_matrix(self.y_test, y_pred))

        # get class_names and feature_names
        class_names = self.df[self.target].unique()
        feature_names = self.df.drop([self.target], axis=1).columns

        # plot confusion matrix
        print('-' * 80)
        print('Plot confusion matrix: ')
        conf_matrix = confusion_matrix(self.y_test, y_pred)
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

        # plot the tree
        print('-' * 80)
        print('Plot the decision tree: ')
        plt.figure(figsize=(15, 10))
        tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=['Transfusions','No'], rounded=True, fontsize=14)
        plt.show()

        print('=' * 80)
        print('END of Decision Tree model')
        print('=' * 80)

        # display the results table
        print('+'*90)
        print('Model Results and Comparison:' + '\n')
        report = classification_report(self.y_test, y_pred, output_dict=True)
        f1 = report['macro avg']['f1-score']
        self.results_table.loc[len(self.results_table.index)] = [model_name, self.target, mean_accuracy, rmse, f1]
        print(self.results_table)
        print('+'*90)

    def SVM(self, kernel, C, gamma):      # kernel can be linder or rbf; C=1.0; gamma=0.2 (for rbf)
        model_name = 'SVM - ' + kernel
        model = SVC(kernel=kernel, C=C, random_state=self.random_state, gamma=gamma)
        model.fit(self.X_train, self.y_train)

        # make predictions
        y_pred = model.predict(self.X_test)

        # results
        print('='*80)
        print('Target: ', self.target)
        print('Model: ', model_name)
        print('-' * 80 + '\n')
        print('Report: '+'\n',
              classification_report(self.y_test, y_pred))
        print('-' * 80)
        print('Model Accuracy: '+'\n',
              accuracy_score(self.y_test, y_pred)*100,'\n')
        print('-' * 80)

        # perform cross validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        print('Cross Validation Results (Accuracy): ' + '\n', scores * 100)
        print('-' * 80)
        print('Mean Accuracy (after cross-validation: ' + '\n', mean_accuracy)
        print('-' * 80)

        # MSE
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        print(f'RMSE: {rmse: .3f}')
        print('-' * 80)

        # confusion matrix
        print('Confusion Matrix: '+'\n', confusion_matrix(self.y_test, y_pred))

        # Plot ROC Area Under Curve
        print('-' * 80)
        print('Plot ROC Aarea Under Curve: ')

        y_pred_proba = model.decision_function(self.X_test)
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

        print('=' * 80)
        print('END of SVM model')
        print('=' * 80)

        # display the results table
        print('+'*90)
        print('Model Results and Comparison:' + '\n')
        report = classification_report(self.y_test, y_pred, output_dict=True)
        f1 = report['macro avg']['f1-score']
        self.results_table.loc[len(self.results_table.index)] = [model_name, self.target, mean_accuracy, rmse, f1]
        print(self.results_table)
        print('+'*90)

    def LogReg(self):
        model_name = 'LogisticRegression'
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)

        # make predictions
        y_pred = model.predict(self.X_test)

        # results
        print('='*80)
        print('Target: ', self.target)
        print('Model: ', model_name)
        print('-' * 80 + '\n')
        print('Report: '+'\n',
              classification_report(self.y_test, y_pred))
        print('-' * 80)
        print('Model Accuracy: '+'\n',
              accuracy_score(self.y_test, y_pred)*100,'\n')
        print('-' * 80)
        y_pred_score = model.predict_proba(self.X_test)
        print('ROC_AUC: ', roc_auc_score(self.y_test,y_pred_score[:,1]) * 100)
        print('\n')

        # perform cross validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        print('Cross Validation Results (Accuracy): ' + '\n', scores * 100)
        print('-' * 80)
        print('Mean Accuracy (after cross-validation: ' + '\n', mean_accuracy)
        print('-' * 80)

        # MSE
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        print(f'RMSE: {rmse: .3f}')
        print('-' * 80)

        # confusion matrix
        print('Confusion Matrix: '+'\n', confusion_matrix(self.y_test, y_pred))

        # Plot ROC Area Under Curve
        print('-' * 80)
        print('Plot ROC Aarea Under Curve: ')

        y_pred_proba = model.decision_function(self.X_test)
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

        print('=' * 80)
        print('END of SVM model')
        print('=' * 80)

        # display the results table
        print('+'*90)
        print('Model Results and Comparison:' + '\n')
        report = classification_report(self.y_test, y_pred, output_dict=True)
        f1 = report['macro avg']['f1-score']
        self.results_table.loc[len(self.results_table.index)] = [model_name, self.target, mean_accuracy, rmse, f1]
        print(self.results_table)
        print('+'*90)

    def XGBoost(self, n_estimators, eta):     # n_estimators defaul = 100; eta defaul = 0.3 (to get L shape increase n; to avoid long-tail, decrease eta)
        model_name = 'XGBoost'
        model = XGBClassifier(n_estimators = n_estimators, eta=eta)
        evalset = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        model.fit(self.X_train, self.y_train, eval_metric='logloss', eval_set=evalset)

        # make predictions
        y_pred = model.predict(self.X_test)

        # results
        print('='*80)
        print('Target: ', self.target)
        print('Model: ', model_name)
        print('-' * 80)
        print('Report: '+'\n',
              classification_report(self.y_test, y_pred))
        print('-' * 80)
        print('Model Accuracy: '+'\n',
              accuracy_score(self.y_test, y_pred)*100,'\n')
        print('-' * 80)

        # MSE
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        print(f'RMSE: {rmse: .3f}')
        print('-' * 80)

        # perform cross validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        print('Cross Validation Results (Accuracy): ' + '\n', scores * 100)
        print('-' * 80)
        print('Mean Accuracy (after cross-validation: ' + '\n', mean_accuracy)
        print('-' * 80)

        # confusion matrix
        print('Confusion Matrix: '+'\n', confusion_matrix(self.y_test, y_pred))

        # plot the learning curves
        print('-' * 80)
        print('Plot the learning curves')
        results = model.evals_result()
        plt.plot(results['validation_0']['logloss'], label='train')
        plt.plot(results['validation_1']['logloss'], label='test')
        plt.xlabel('# of iterations')
        plt.ylabel('Logloss')
        plt.legend()
        plt.show()

        print('=' * 80)
        print('END of XGBoost model')
        print('=' * 80)

        # display the results table
        print('+'*90)
        print('Model Results and Comparison:' + '\n')
        report = classification_report(self.y_test, y_pred, output_dict=True)
        f1 = report['macro avg']['f1-score']
        self.results_table.loc[len(self.results_table.index)] = [model_name, self.target, mean_accuracy, rmse, f1]
        print(self.results_table)
        print('+'*90)
