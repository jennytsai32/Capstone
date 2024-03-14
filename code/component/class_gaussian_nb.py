# This class includes functions for GaussianNB model

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class NaiveBayesGaussianNB:

    def __init__(self, df, target, test_size, random_state):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = 'GaussianNB'
        self.parameters = ''

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns.to_list()

        # creating the classifier object
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)