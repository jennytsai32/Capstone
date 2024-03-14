# This class includes functions for GradientBoosting model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoosting:

    def __init__(self, df, target, test_size, random_state, n_estimators, learning_rate):   # starting point: n_estimators=300, learning_rate=0.05
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = 'Gradient Boosting'
        self.parameters = 'n_estimators='+str(n_estimators) +', learning_rate='+str(learning_rate)

        self.X = df.drop([target], axis=1)
        self.y = df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns.to_list()

        # creating the classifier object
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, learning_rate=learning_rate)
        self.model.fit(self.X_train, self.y_train)