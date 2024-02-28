# This class includes functions for RandomForest model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class RandomForest:

    def __init__(self, df, target, test_size, random_state, n_estimators, feature_importances):   # starting point: n_estimators=100
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = 'Random Forest'
        self.parameters = 'n_estimators='+str(n_estimators) + ', ' +str(feature_importances) + '_features'

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

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

        #================================================
        # make prediction
        # self.y_pred = self.model_k_features.predict(self.newX_test)
        #================================================

        # get feature and class names
        self.class_names = self.df[self.target].unique()
        self.feature_names = self.f_importances.index