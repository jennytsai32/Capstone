import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, mean_squared_error
from sklearn.svm import SVC
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

# import clean data
#df=pd.read_csv(r'https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/CABG_20_recoded.csv',index_col=0)
df=pd.read_csv(r'https://github.com/jennytsai32/Capstone/blob/master/code/main_code/CABG_2018_2020_preselect.csv', index_col=0)
df=df.drop(['PUFYEAR'], axis=1)

X = df.values[:,1:]
y = df.values[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

model_name = 'XGBoost'
xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
}

model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=50)

# make predictions
y_pred = model.predict(xgb_test)

# results
print('='*80)
print('Target: OTHBLEED')
print('Model: XGBoost')
print('-' * 80)
# print('Report: '+'\n',
#       classification_report(self.y_test, y_pred))
print('-' * 80)
print('Model Accuracy: '+'\n',
      accuracy_score(y_test, y_pred)*100,'\n')
print('-' * 80)

# MSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse: .3f}')
print('-' * 80)

# perform cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=100)
scores = cross_val_score(model, X, y, cv=kf)
mean_accuracy = scores.mean() * 100
print('Cross Validation Results (Accuracy): ' + '\n', scores * 100)
print('-' * 80)
print('Mean Accuracy (after cross-validation: ' + '\n', mean_accuracy)
print('-' * 80)

# confusion matrix
print('Confusion Matrix: '+'\n', confusion_matrix(y_test, y_pred))

print('=' * 80)
print('END of XGBoost model')
print('=' * 80)

