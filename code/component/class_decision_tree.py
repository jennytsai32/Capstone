# This class includes functions for Decision Tree model

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:

    def __init__(self, df, target, test_size, random_state, criterion, max_depth, min_samples_leaf):   # starting point: criterion = 'gini' or 'entropy', max_depth=3, min_samples_leaf=5
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model_name = 'Decision Tree - ' + criterion
        self.parameters = 'max_depth='+str(max_depth) +', min_samples_leaf='+str(min_samples_leaf)

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        self.class_names = self.df[self.target].unique()
        self.feature_names = self.df.drop([self.target], axis=1).columns.to_list()

        # creating the classifier object
        self.model = DecisionTreeClassifier(criterion=criterion, random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.model.fit(self.X_train, self.y_train)