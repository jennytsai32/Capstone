# This class includes functions for XGBoost model

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class XGB:

    def __init__(self, df, target, test_size, random_state, n_estimators, eta):     # starting point: n_estimators defaul = 100; eta defaul = 0.3 (to get L shape increase n; to avoid long-tail, decrease eta)
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = 'XGBoost'
        self.parameters = 'n_estimators=' + str(n_estimators) + ', eta=' + str(eta)

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns

        # creating the classifier object
        self.model = XGBClassifier(n_estimators = n_estimators, eta=eta, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)