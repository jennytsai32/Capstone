from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


df = pd.read_csv('/processed_data/2018_2022/CABG_5yr_preselect40.csv')
print(df.shape)



Y = np.array(df['OTHBLEED'])
X = np.array(df.loc[:, df.columns != 'OTHBLEED'])

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25, random_state=42)


tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, scoring='f1_weighted')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
