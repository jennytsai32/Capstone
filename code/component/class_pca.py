from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA


class PCA_for_Feature_Selection:

    def __init__(self, df, target):
        self.df = df
        self.target = target

        self.X = df.drop([target], axis=1).values
        self.y = df[target].values

        # scale to 0-1
        self.X_SS = StandardScaler().fit_transform(self.X)

        # perform PCA
        self.pca = PCA(n_components='mle', svd_solver='full')
        self.pca.fit(self.X_SS)
        self.X_PCA = self.pca.transform(self.X_SS)

    def PCA_Reduced_Feature(self):
        print('Original Dimension: ', self.X_SS.shape)
        print('Transformed Dimension: ', self.X_PCA.shape)
        print(f'Reduced feature #: {self.X_SS.shape[1] - self.X_PCA.shape[1]}')

    def Explained_Variance_Ratio(self):
        print(f'Explained variance ratio of the reduced feature space: \n'
              f' {self.pca.explained_variance_ratio_}')

    def Reduced_Feature_Space_Plot(self):
        # Reduced Feature Space Plot
        plt.figure(figsize=(11, 6))
        x = np.arange(1, len(np.cumsum(self.pca.explained_variance_ratio_)) + 1, step=1)
        plt.xticks(x)
        plt.plot(x, np.cumsum(self.pca.explained_variance_ratio_))
        plt.title('PCA Plot - Cumulative Explained Variance vs Reduced Feature Space')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        plt.show()

    def Reduced_Feature_Space_Heatmap(self):
        # heatmap of reduced feature space
        # get the correlation coefficient matrix
        df_4g_reduced = pd.DataFrame(data=self.X_PCA)
        corr_matrix = df_4g_reduced.corr()
        print(corr_matrix)
        # plot the heatmap
        sns.heatmap(data=corr_matrix, annot=True, )
        plt.title('Correlation Coefficient Between Features\n - Reduced Feature Space by PCA')
        plt.show()

    def PCA_New_df(self):
        # construction of new (reduced dim) dataset
        a, b = self.X_PCA.shape
        column = []
        for i in range(b):
            column.append(f'Principal Col {i + 1}')
        df_PCA = pd.DataFrame(data=self.X_PCA, columns=column)
        return df_PCA