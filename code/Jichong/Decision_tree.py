import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/CABG_20_recoded.csv',index_col=0)
df=df.drop(['PUFYEAR'], axis=1)

X = df.values[:,1:]
y = df.values[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: " + '\n',
      classification_report(y_test,y_pred_gini))

report = classification_report(y_test,y_pred_gini, output_dict=True)
f1 = report['macro avg']['f1-score']
print('F1-score: ', f1)
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print('-'*80 + '\n')

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = df.OTHBLEED.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(10,10))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


# display decision tree
feature_names = df.columns[1:]
class_names=['Transfusions','No']
plt.figure(figsize=(15, 10))
tree.plot_tree(clf_gini, filled=True, feature_names=feature_names, class_names=class_names, rounded=True, fontsize=14)
plt.show()

# perform cross validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=100)

scores = cross_val_score(clf_gini, X, y, cv=kf)
mean_accuracy = scores.mean() * 100
print('Cross validation results (Accuracy): ', scores * 100)
print("Mean Accuracy:", mean_accuracy)


#=======================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from Basic_Classification_Models import BasicClassificationModels

df=pd.read_csv(r'https://raw.githubusercontent.com/jennytsai32/Capstone/master/code/main_code/CABG_20_recoded.csv',index_col=0)
df=df.drop(['PUFYEAR'], axis=1)


# df=pd.read_csv(r'https://raw.githubusercontent.com/amir-jafari/Data-Mining/master/08-Decision_Tree/2-Example_Exercise/balance-scale.data.csv')
# df.columns=['target','Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']


data = BasicClassificationModels(df,'OTHBLEED',0.3,100)

data.Decision_Tree('gini',3,5)
data.SVM('linear',1.0,0)