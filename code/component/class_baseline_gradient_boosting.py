# This class includes functions for GradientBoosting model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, root_mean_squared_error, f1_score
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns


class GradientBoosting:

    def __init__(self, df, target, test_size, random_state, k_folds, n_estimators, learning_rate):   # starting point: n_estimators=300, learning_rate=0.05
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.k_folds = k_folds
        self.model_name = 'Gradient Boosting'
        self.parameters = 'n_estimators='+str(n_estimators) +', learning_rate='+str(learning_rate)

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[0]
        self.X_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[1]
        self.y_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[2]
        self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[3]

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns

        # get y_pred
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, learning_rate=learning_rate)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def Predict(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print(f'Y-test prediction: {self.y_pred}')
        print('-' * 80)


    def Report(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Report: ' + '\n',
              classification_report(self.y_test, self.y_pred))
        print('-' * 80)

    def Accuracy(self):
        # perform cross validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print(f'Model Accuracy (mean of {self.k_folds} folds): {mean_accuracy}')
        print('-' * 80)


    def RMSE(self):
        rmse = root_mean_squared_error(self.y_test, self.y_pred)
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print(f'RMSE: {rmse: .3f}')
        print('-' * 80)

    def Confusion_Matrix(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Confusion Matrix: ' + '\n', confusion_matrix(self.y_test, self.y_pred))
        print('-' * 80)

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

    def ROC_AUC_Score(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        y_pred_proba = self.model.predict_proba(self.X_test)[:,1]
        auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f'ROC-AUC score: {auc:.3f}')
        print('-' * 80)

    def ROC_AUC_Plot(self):
        # Plot ROC Area Under Curve
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Plot ROC Aarea Under Curve: ')

        y_pred_proba = self.model.predict_proba(self.X_test)[:,1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc = roc_auc_score(self.y_test, y_pred_proba)

        plt.figure(figsize=(10, 10))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='AUC=' + str(auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Plot - {self.model_name}')
        plt.legend(loc="lower right")
        plt.show()

        return fpr, tpr, auc, self.model_name

    def Display_Model_Results(self):
        report = classification_report(self.y_test, self.y_pred, output_dict=True)

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        rmse = root_mean_squared_error(self.y_test, self.y_pred)

        f1 = report['macro avg']['f1-score']

        # ROC-AUC
        y_pred_proba = self.model.predict_proba(self.X_test)[:,1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc = roc_auc_score(self.y_test, y_pred_proba)

        # construct the table
        dict = {'Model Name': self.model_name,
                'Parameters': self.parameters,
                'Target': self.target,
                'Mean Accuracy ('+str(self.k_folds)+' folds)': mean_accuracy,
                'RMSE': rmse,
                'F1-score': f1,
                'ROC-AUC score': auc}
        results_table = pd.DataFrame([dict])
        return results_table

