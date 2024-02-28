# This class includes functions for KNN model

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class KNN:

    def __init__(self, df, target, test_size, random_state, n_neighbors):   # starting point: n_neighbors = 3
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = 'KNN'
        self.parameters = 'n_neighbors='+str(n_neighbors)

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns

        # creating the classifier object
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.X_train, self.y_train)