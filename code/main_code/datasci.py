# CS Foundations Final Project
# Team #8
# Jenny Tsai & Jiazu Zhang


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class datasci():

    def __init__(self, df):
        self.df = df
        self.shape = df.shape
        self.nrows = df.shape[0]
        self.ncols = df.shape[1]


    def size(self):
        '''
        returns number of columns and rows in an easy-to-read format.
        '''
        return f"There are {self.nrows} rows and {self.ncols} columns in the dataset."


    def missingReport(self, insight = False):
        '''
        returns a report of variables with missing values and missing value percentage in relations to its sample size. 
        If insight = True, prints recommendations for how to handle missing values.
        '''
        missingVal = self.df[self.df.columns[self.df.isnull().any()].tolist()].isnull().sum()
        missingPer = missingVal * 100 / self.nrows

        frame = {'Num of Nans': missingVal,
                'Percent of Nans': missingPer}

        missing_report = pd.DataFrame(frame).sort_values(by='Percent of Nans', ascending = False).round(2)

        if insight == True:
            print('INSIGHT:')
            print('The following variables contains over 70% missing values:')
            print(missing_report[missing_report['Percent of Nans']>=70].index.tolist())
            print()
            print('Further check the variables to see if there is any systematic issue.\nImputation is not recommended for these variables as it may lead to misleading results.')
            print()

        return missing_report
    

    def imputation(self, columns, impute = 'mean', insight=False):
        '''
        imputes the missing values with selected statistics.
        :: columns: list of column names with missing values to be imputed.
        :: impute: statistics to be imputed, including 'mean', 'median', 'mode', 'max', 'min', and 'other'. If 'other' is selected, users will be prompted to enter custom values to impute for each column.
        :: insight: default = False. If True, prints a table comparing statistics before vs. after imputation.
        '''
        if insight == True:
            target = str(input("Please enter the target variable:"))
            nan_bf = self.df[columns].isnull().sum()
            std_bf = self.df[columns].std()
            df1 = self.df[columns]
            df2 = self.df[target]
            df_new = pd.concat([df1, df2], axis=1)
            corr_bf = df_new.corr(method='pearson')

            nan_df = {'Before imputation': nan_bf}
            std_df = {'Before imputation': std_bf}

        if impute == 'mean':
            for col in columns:
                self.df[col].fillna(self.df[col].mean(), inplace = True)

        elif impute == 'median':
            for col in columns:
                self.df[col].fillna(self.df[col].median(), inplace = True)
        
        elif impute == 'mode':
            for col in columns:
                self.df[col].fillna(self.df[col].mode(), inplace = True)
        
        elif impute == 'max':
            for col in columns:
                self.df[col].fillna(self.df[col].max(), inplace = True)

        elif impute == 'min':
            for col in columns:
                self.df[col].fillna(self.df[col].min(), inplace = True)
    
        elif impute == 'other':

            for col in columns:
                print(f'Please enter the value to impute "{col}":')
                x = input()
                self.df[col].fillna(x, inplace = True)

        if insight == True:
            nan_af = self.df[columns].isnull().sum()
            nan_af = self.df[columns].isnull().sum()
            std_af = self.df[columns].std()
            corr_af = df_new.corr(method='pearson')

            nan_df['After imputation'] = nan_af
            std_df['After imputation'] = std_af
            print('---------------------------------------------------')
            print('Number of Nans:')
            print(pd.DataFrame(nan_df))
            print()
            print('---------------------------------------------------')
            print('Standard Deviation:')
            print(pd.DataFrame(std_df).round(2))
            print()
            print('---------------------------------------------------')
            print(f'Correlation with {target} before imputation:')
            print(pd.DataFrame(corr_bf).round(2))
            print()
            print(f'Correlation with {target} after imputation:')
            print(pd.DataFrame(corr_af).round(2))



    def recode(self, column, oldVal, newVal, inplace=False):
        '''
        recode the variable by replacing a list of old values with new values.
        :: column: column name whose values to be recoded.
        :: oldVal: list of original values within the variable/column.
        :: newVal: list of new values to replace the old values.
        :: inplace: default = False, which creates a new variable/column with new values. If True, new values replace the old values in the original variable/column.
        '''
        new_name = str(column) + '_NEW'
        self.df[new_name] = self.df[column].replace(oldVal, newVal)

        if inplace == True:
            self.df[column].replace(oldVal, newVal, inplace=True)
            return self.df[column].value_counts()
        
        else:
            return self.df[new_name].value_counts()
        

    def eda(self, column, insight=False):
        '''
        return a graph (histogram or bar chart) of the variable's distribution with descriptive statistics for exploratory data analysis.
        :: column: column name, can be continuous variable or categorical variable.
        :: insight: default = False. If True, prints recommendations for data imputation based on skewness of distribution of the variable.
        '''
        if isinstance(self.df[column][0], (np.int64, np.int32, np.float32, np.float64)):
    
            data = {'N':[len(self.df[column])],
                    '# of Nans':[self.df[column].isnull().sum()],
                    'Mean':[self.df[column].mean()],
                    'Std':[self.df[column].std()],
                    'Median':[self.df[column].median()],
                    'Max':[self.df[column].max()],
                    'Min':[self.df[column].min()],
                    'Skewness':[self.df[column].skew()]
                    }
            descr = pd.DataFrame.from_dict(data, orient='index', columns=[column]).round(2)

            plt.hist(self.df[column], color = "#108A99")
            plt.axvline(self.df[column].mean(), color='r', label='mean')
            plt.axvline(self.df[column].median(), color='blue', linestyle='dashed',label='median')
            plt.legend()
            plt.xlabel(column)

            plt.show()

            if insight==True:
                print('INSIGHT:')
                if self.df[column].skew() <= 0.5 and self.df[column].skew() >= -0.5:
                    print('The distribution is close to normal distribution. Mean can be a good estimate for data impuation.')
                else:
                    print('The distribution is skewed. Median can be a good estimate for data impuation.')

            return descr
        
        else:
            self.df[column].value_counts(sort=True).plot.bar(rot=0)
            plt.show()
            return self.df[column].value_counts()
        
        
    def featureSelection(self, columns, target):
        '''
        returns a bar chart with top 10 important features predicting the target variable.
        :: features: list of column names to be entered into the feature selection model (i.e., Random Forest).
        :: target: target variable name
        '''
        X = self.df[columns].values
        Y = self.df[target].values
        feature_names = self.df[columns].columns

        clf = RandomForestClassifier(n_estimators=100)

        clf.fit(X, Y)

        # plot feature importances
        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(clf.feature_importances_, feature_names)

        f_importances.sort_values(ascending=False, inplace=True)

        f_importances.plot(xlabel='Features', ylabel='Importance', kind='bar', rot=90)
    
        plt.tight_layout()
        plt.show()


    def column_tolist (self):
        x = self.df.columns
        temp = []
        for i in x:
            temp.append(i)
        return temp
        

    def file_comparsion(file1_path , file2_path):
        try: 
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                print('Comparison Started...')
                Indicator = True
                for line_num, (line1, line2) in enumerate(zip(f1, f2), start = 1):
                    if line1 != line2:
                        print(f'\nDifference found at line {line_num}: \n{file1_path}: {line1}\n{file2_path}: {line2}')
                        column1 = line1.split(',')
                        column2 = line2.split(',')
                        for col_num, (col1, col2) in enumerate(zip(column1, column2), start = 1):
                            if col1 != col2:
                                col2 = ('NULL' if col2 == '' else col2)
                                print(f"\nAt column {col_num}: \nOrignial Data: {col1} \nTarget Data: {col2}")
                        Indicator = False
            if Indicator:
                print('\{file1_path} and {file2_path} have same structure and content.')
        
            print('Comparison Completed!')
        except Exception as e:
            print(f'Error:{e}')
            return False  


# %%
