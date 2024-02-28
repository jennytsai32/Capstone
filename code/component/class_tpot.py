# This class includes functions for TPOT

from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

class TPOT:

    def __init__(self, df, target, test_size, random_state, generations, population_size, verbosity):   # starting point: generations=5, population_size=20, verbosity=2
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = 'TPOT'
        self.parameters = f'generations={generations}, population_size={population_size}, verbosity={verbosity}'

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns

        # creating the classifier object
        self.model = TPOTClassifier(generations=generations, random_state=random_state, population_size=population_size, verbosity=verbosity)
        self.model.fit(self.X_train, self.y_train)

        # display results
        self.scores = self.model.score(self.X_test, self.y_test)