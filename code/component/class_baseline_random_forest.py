# This class includes functions for RandomForest model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, root_mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

class RandomForest:

    def __init__(self, df, target, test_size, random_state, k_folds, n_estimators, feature_importances):   # starting point: n_estimators=100
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.k_folds = k_folds
        self.model_name = 'Random Forest'
        self.parameters = 'n_estimators='+str(n_estimators) + ', ' +str(feature_importances) + '_features'

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[0]
        self.X_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[1]
        self.y_train = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[2]
        self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)[3]

        # build model
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)

        # get feature importances
        self.importances = self.model.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        self.f_importances = pd.Series(self.importances, self.df.iloc[:, 1:].columns)

        # sort the array in descending order of the importances
        self.f_importances.sort_values(ascending=False, inplace=True)

        # select NEW X_train, X_test on k-features
        self.newX_train = self.X_train[:, self.model.feature_importances_.argsort()[::-1][:feature_importances]]
        self.newX_test = self.X_test[:, self.model.feature_importances_.argsort()[::-1][:feature_importances]]

        # build the model again
        self.model_k_features = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model_k_features.fit(self.newX_train, self.y_train)

        # make prediction
        self.y_pred = self.model_k_features.predict(self.newX_test)

        # get feature and class names
        self.class_names = self.df[self.target].unique()
        self.feature_names = self.f_importances.index

    def Random_Forest_Feature_Importances_Plot(self):
        self.f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
        plt.title('Feature Importances - Random Forest')
        plt.tight_layout()
        plt.show()

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

    def Decision_Tree_Plot(self):
        # plot the tree
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Plot the decision tree: ')
        plt.figure(figsize=(15, 10))
        tree.plot_tree(self.model_k_features, filled=True, feature_names=self.feature_names, class_names=['Transfusions','No'], rounded=True, fontsize=14)
        plt.show()

    def ROC_AUC_Score(self):
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        y_pred_proba = self.model_k_features.predict_proba(self.newX_test)[:,1]
        auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f'ROC-AUC score: {auc:.3f}')
        print('-' * 80)

    def ROC_AUC_Plot(self):
        # Plot ROC Area Under Curve
        print('Target: ', self.target)
        print('Model: ', self.model_name)
        print('Plot ROC Aarea Under Curve: ')

        y_pred_proba = self.model_k_features.predict_proba(self.newX_test)[:,1]
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
        scores = cross_val_score(self.model_k_features, self.X, self.y, cv=kf)
        mean_accuracy = scores.mean() * 100
        rmse = root_mean_squared_error(self.y_test, self.y_pred)

        f1 = report['macro avg']['f1-score']

        # ROC-AUC
        y_pred_proba = self.model_k_features.predict_proba(self.newX_test)[:,1]
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

