
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


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



    def recode(self, col, oldVal, newVal, inplace=False):
        '''
        recode the variable by replacing a list of old values with new values.
        :: col_list: list of column name whose values to be recoded.
        :: inplace: default = False, which creates a new variable/column with new values. If True, new values replace the old values in the original variable/column.
        '''
        new_name = str(col) + '_NEW'

        #res = {test_keys[i]: test_values[i] for i in range(len(test_keys))}

        if inplace == True:
            self.df[col].replace(oldVal, newVal, inplace=True)
            #return self.df[col].value_counts()
        
        else:
            self.df[new_name] = self.df[col].replace(oldVal, newVal)
            #return self.df[new_name].value_counts()
        

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

    def standardize(self, col_list):
        '''
        standardize a list of columns
        '''
        for col in col_list:
            self.df[col] = (self.df[col] - self.df[col].mean())/(self.df[col].std())

    # code obtained from Prof. Amir Jafari
    def remove_all_nan_columns(self):
        """
        Remove columns with all NaN values from a DataFrame
        """
        # Get columns with all NaNs
        all_nan_cols = [col for col in self.df.columns if self.df[col].isnull().all()]

        # Drop columns with all NaNs
        self.df.drop(all_nan_cols, axis=1, inplace=True)
    
    # code modified from Prof. Amir Jafari
    def impute_all(self, num_strategy ='mean', bool_strategy='most_frequent'):
        '''
        impute the all dataframe based on the column type.
        :: num_strategy: imputation strategy for numeric variables, including 'mean', 'median', 'mode', 'max'
        :: bool_strategy: imputation strategy for booleans, including 'most_frequent', 'all_true', 'all_false'
        '''

        # (1) impute numeric features:
        numeric_cols = self.df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns

        for col in numeric_cols:
            if num_strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif num_strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif num_strategy == 'mode':
                self.df[col].fillna((self.df[col].mode()), inplace=True)
            elif num_strategy == 'max':
                self.df[col].fillna(self.df[col].max(), inplace=True)
            else:
                print('Invalid imputation strategy. Using mean instead.')
                self.df[col].fillna(self.df[col].mean(), inplace=True)


        # (2) impute boolean features:
        bool_cols = []

        for col in self.df.columns:
            uniques = self.df[col].unique()
            has_nan = pd.isna(uniques).any()

            if has_nan:
                # Make sure NaN itself is not considered unique
                uniques = uniques[~pd.isna(uniques)]

            # Check 0s and 1s or Trues and Falses
            if set(uniques) <= {0, 1} or set(uniques) <= {True, False} or set(uniques) <= {'Yes', 'No'}:
                bool_cols.append(col)

        for col in bool_cols:

            if bool_strategy == 'most_frequent':
                most_freq = self.df[col].mode()[0]
                self.df[col].fillna(most_freq, inplace=True)

            elif bool_strategy == 'all_true':
                self.df[col] = self.df[col].astype(bool)
                self.df[col].fillna(True, inplace=True)

            elif bool_strategy == 'all_false':
                self.df[col] = self.df[col].astype(bool)
                self.df[col].fillna(False, inplace=True)

            else:
                print('Invalid strategy. Using most frequent.')
                most_freq = self.df[col].mode()[0]
                self.df[col].fillna(most_freq, inplace=True)

        # (3) impute categorical features:
                
        # Identify object/category columns
        object_cols = self.df.select_dtypes(include=['object', 'category'])
        MAP = []
        # Iterate through each object column
        for col in object_cols:
            # Check if there are NaN values
            if self.df[col].isnull().values.any():
                # Encode column as numeric with label encoder
                le = LabelEncoder()
                self.df[col] = self.df[col].astype(str)
                self.df[col] = le.fit_transform(self.df[col])

                # Get the max category label
                max_label = self.df[col].max()

                # Set NaN values to max + 1
                self.df[col] = self.df[col].fillna(max_label + 1)
                mappings = dict(zip(le.transform(le.classes_), le.classes_))
                MAP.append((col, mappings))
            else:
                le = LabelEncoder()
                self.df[col] = self.df[col].astype(str)
                self.df[col] = le.fit_transform(self.df[col])
                mappings = dict(zip(le.transform(le.classes_), le.classes_))
                MAP.append((col, mappings))

        return self.df, MAP


# **********************************
# Functions:
       
def file_compare(df1, df2):
    # find new columns in df2
    new_col = []
    for col in df2.columns:
        if col not in df1.columns:
            new_col.append(col)
    print("New columns:", new_col)

    # find old columns removed in df2
    rmv_col = []
    for col in df1.columns:
        if col not in df2.columns:
            rmv_col.append(col)
    print("Removed columns:", rmv_col)

    # find columns with the same name but different values
    dif_val = []
    obj_cols_df1 = df1.select_dtypes(include=['object', 'category'])
    obj_cols_df2 = df2.select_dtypes(include=['object', 'category'])
    common_cols = [x for x in obj_cols_df1 if x in obj_cols_df2]
    
    for col in common_cols:
        df1_unique = df1[col][~pd.isna(df1[col])].unique()
        df2_unique = df2[col][~pd.isna(df2[col])].unique()

        df1_unique = sorted([str(element) for element in df1_unique])
        df2_unique = sorted([str(element) for element in df2_unique])

        #df1_unique = df1[col].unique()
        #df2_unique = df2[col].unique()
        
        if len(df1_unique) != len(df2_unique):
            dif_val.append(col)

        else:
            result = all(x == y for x, y in zip(df1_unique, df2_unique))
            if result == False:
                dif_val.append(col)
            
    print('Same name different values:', dif_val)

    return new_col, rmv_col, dif_val

    

def glossary(key):
    content = {'DISCHDEST':'Discharge destination','PUFYEAR': 'Year of PUF', 'SEX': 'Gender', 'RACE_NEW': 'New Race', 'ETHNICITY_HISPANIC': 'Ethnicity Hispanic', 'CPT': 'CPT', 'INOUT': 'Inpatient/outpatient', 'AGE': 'Age of patient with patients over 89 coded as 90+', 'ANESTHES': 'Principal anesthesia technique', 'ELECTSURG': 'Elective Surgery (this variable name changes with 2021/2022)', 'HEIGHT': 'Height in inches', 'WEIGHT': 'Weight in lbs', 'DIABETES': 'Diabetes mellitus with oral agents or insulin', 'SMOKE': 'Current smoker within one year', 'DYSPNEA': 'Dyspnea', 'FNSTATUS2': 'Functional health status Prior to Surgery', 'VENTILAT': 'Ventilator dependent', 'HXCOPD': 'History of severe COPD', 'ASCITES': 'Ascites', 'HXCHF': 'Heart failure (CHF) in 30 days before surgery', 'HYPERMED': 'Hypertension requiring medication', 'RENAFAIL': 'Acute renal failure (pre-op)', 'DIALYSIS': 'Currently on dialysis (pre-op)', 'DISCANCR': 'Disseminated cancer', 'WNDINF': 'Open wound/wound infection', 'STEROID': 'Immunosuppressive Therapy', 'WTLOSS': 'Malnourishment', 'BLEEDIS': 'Bleeding disorder', 'TRANSFUS': 'Preop Transfusion of >= 1 unit of whole/packed RBCs in 72 hours prior to surgery', 'PRSODM': 'Pre-operative serum sodium', 'PRBUN': 'Pre-operative BUN', 'PRCREAT': 'Pre-operative serum creatinine', 'PRALBUM': 'Pre-operative serum albumin', 'PRBILI': 'Pre-operative total bilirubin', 'PRSGOT': 'Pre-operative SGOT', 'PRALKPH': 'Pre-operative alkaline phosphatase', 'PRWBC': 'Pre-operative WBC', 'PRHCT': 'Pre-operative hematocrit', 'PRPLATE': 'Pre-operative platelet count', 'PRPTT': 'Pre-operative PTT', 'PRINR': 'Pre-operative International Normalized Ratio (INR) of PT values', 'PRPT': 'Pre-operative PT', 'EMERGNCY': 'Emergency case', 'ASACLAS': 'ASA classification', 'OPTIME': 'Total operation time', 'TOTHLOS': 'Length of total hospital stay', 'CASEID': 'Case Identification Number', 'PRNCPTX': 'Principal operative procedure CPT code description', 'WORKRVU': 'Work Relative Value Unit', 'TRANST': 'Transfer status', 'ADMYR': 'Year of Admission', 'OPERYR': 'Year of Operation', 'SURGSPEC': 'Surgical Specialty', 'DPRNA': 'Days from Na Preoperative Labs to Operation', 'DPRBUN': 'Days from BUN Preoperative Labs to Operation', 'DPRCREAT': 'Days from Creatinine Preoperative Labs to Operation', 'DPRALBUM': 'Days from Albumin Preoperative Labs to Operation', 'DPRBILI': 'Days from Bilirubin Preoperative Labs to Operation', 'DPRSGOT': 'Days from SGOT Preoperative Labs to Operation', 'DPRALKPH': 'Days from ALKPHOS Preoperative Labs to Operation', 'DPRWBC': 'Days from WBC Preoperative Labs to Operation', 'DPRHCT': 'Days from HCT Preoperative Labs to Operation', 'DPRPLATE': 'Days from PlateCount Preoperative Labs to Operation', 'DPRPTT': 'Days from PTT Preoperative Labs to Operation', 'DPRPT': 'Days from PT Preoperative Labs to Operation', 'DPRINR': 'Days from INR Preoperative Labs to Operation', 'OTHERWRVU1': 'Other Work Relative Value Unit 1', 'OTHERWRVU2': 'Other Work Relative Value Unit 2', 'OTHERWRVU3': 'Other Work Relative Value Unit 3', 'OTHERWRVU4': 'Other Work Relative Value Unit 4', 'OTHERWRVU5': 'Other Work Relative Value Unit 5', 'OTHERWRVU6': 'Other Work Relative Value Unit 6', 'OTHERWRVU7': 'Other Work Relative Value Unit 7', 'OTHERWRVU8': 'Other Work Relative Value Unit 8', 'OTHERWRVU9': 'Other Work Relative Value Unit 9', 'OTHERWRVU10': 'Other Work Relative Value Unit 10', 'CONWRVU1': 'Concurrent Work Relative Value Unit 1', 'CONWRVU2': 'Concurrent Work Relative Value Unit 2', 'CONWRVU3': 'Concurrent Work Relative Value Unit 3', 'CONWRVU4': 'Concurrent Work Relative Value Unit 4', 'CONWRVU5': 'Concurrent Work Relative Value Unit 5', 'CONWRVU6': 'Concurrent Work Relative Value Unit 6', 'CONWRVU7': 'Concurrent Work Relative Value Unit 7', 'CONWRVU8': 'Concurrent Work Relative Value Unit 8', 'CONWRVU9': 'Concurrent Work Relative Value Unit 9', 'CONWRVU10': 'Concurrent Work Relative Value Unit 10', 'WNDCLAS': 'Wound classification', 'MORTPROB': 'Estimated Probability of Mortality', 'MORBPROB': 'Estimated Probability of Morbidity', 'HDISDT': 'Hospital discharge Year', 'YRDEATH': 'Year of death', 'ADMQTR': 'Quarter of Admission', 'HTOODAY': 'Days from Hospital Admission to Operation', 'NSUPINFEC': 'Number of Superficial Incisional SSI  Occurrences', 'SUPINFEC': 'Occurrences Superficial surgical site infection', 'SSSIPATOS': 'Superficial Incisional SSI PATOS', 'DSUPINFEC': 'Days from Operation until Superficial Incisional SSI Complication', 'NWNDINFD': 'Number of Deep Incisional SSI Occurrences', 'WNDINFD': 'Occurrences Deep Incisional SSI', 'DSSIPATOS': 'Deep Incisional SSI PATOS', 'DWNDINFD': 'Days from Operation until Deep Incisional SSI Complication', 'NORGSPCSSI': 'Number of Organ/Space SSI Occurrences', 'ORGSPCSSI': 'Occurrences Organ Space SSI', 'OSSIPATOS': 'Organ/Space SSI PATOS', 'DORGSPCSSI': 'Days from Operation until Organ/Space SSI Complication', 'NDEHIS': 'Number of Wound Disruption Occurrences', 'DEHIS': 'Occurrences Wound Disrupt', 'DDEHIS': 'Days from Operation until Wound Disruption Complication', 'NOUPNEUMO': 'Number of Pneumonia Occurrences', 'OUPNEUMO': 'Occurrences Pneumonia', 'PNAPATOS': 'Pneumonia PATOS', 'DOUPNEUMO': 'Days from Operation until Pneumonia Complication', 'NREINTUB': 'Number of Unplanned Intubation Occurrences', 'REINTUB': 'Occurrences Unplanned Intubation', 'DREINTUB': 'Days from Operation until Unplanned Intubation Complication', 'NPULEMBOL': 'Number of Pulmonary Embolism Occurrences', 'PULEMBOL': 'Occurrences Pulmonary Embolism', 'DPULEMBOL': 'Days from Operation until Pulmonary Embolism Complication', 'NFAILWEAN': 'Number of On Ventilator > 48 Hours Occurrences', 'FAILWEAN': 'Occurrences Ventilator > 48Hours', 'VENTPATOS': 'On Ventilator > 48 Hours PATOS', 'DFAILWEAN': 'Days from Operation until On Ventilator > 48 Hours Complication', 'NRENAINSF': 'Number of Progressive Renal Insufficiency Occurrences', 'RENAINSF': 'Occurrences Progressive Renal Insufficiency', 'DRENAINSF': 'Days from Operation until Progressive Renal Insufficiency Complication', 'NOPRENAFL': 'Number of Acute Renal Failure Occurrences', 'OPRENAFL': 'Occurrences Acute Renal Fail', 'DOPRENAFL': 'Days from Operation until Acute Renal Failure Complication', 'NURNINFEC': 'Number of Urinary Tract infection Occurrences', 'URNINFEC': 'Occurrences Urinary Tract Infection', 'UTIPATOS': 'UTI PATOS', 'DURNINFEC': 'Days from Operation until Urinary Tract Infection Complication', 'NCNSCVA': 'Number of Stroke/CVA Occurrences', 'CNSCVA': 'CVA/Stroke with neurological deficit', 'DCNSCVA': 'Days from Operation until Stroke/CVA Complication', 'NCDARREST': 'Number of Cardiac Arrest Requiring CPR Occurrences', 'CDARREST': 'Occurrences Cardiac Arrest Requiring CPR', 'DCDARREST': 'Days from Operation until Cardiac Arrest Requiring CPR Complication', 'NCDMI': 'Number of Myocardial Infarction Occurrences', 'CDMI': 'Occurrences Myocardial Infarction', 'DCDMI': 'Days from Operation until Myocardial Infarction Complication', 'NOTHBLEED': 'Number of Bleeding Transfusions Occurrences', 'OTHBLEED': 'Occurrences Bleeding Transfusions', 'DOTHBLEED': 'Days from Operation until Bleeding Transfusions Complication', 'NOTHDVT': 'Number of DVT/Thrombophlebitis Occurrences', 'OTHDVT': 'Occurrences DVT/Thrombophlebitis', 'DOTHDVT': 'Days from Operation until DVT/Thrombophlebitis Complication', 'NOTHSYSEP': 'Number of Sepsis Occurrences', 'OTHSYSEP': 'Occurrences Sepsis', 'SEPSISPATOS': 'Sepsis PATOS', 'DOTHSYSEP': 'Days from Operation until Sepsis Complication', 'NOTHSESHOCK': 'Number of Septic Shock Occurrences', 'OTHSESHOCK': 'Occurrences Septic Shock', 'SEPSHOCKPATOS': 'Septic Shock PATOS', 'DOTHSESHOCK': 'Days from Operation until Septic Shock Complication', 'PODIAG10': 'Post-op diagnosis (ICD 10)', 'PODIAGTX10': 'Post-op Diagnosis Text', 'RETURNOR': 'Return to OR', 'DOPERTOD': 'Days from Operation to Death', 'DOPTODIS': 'Days from Operation to Discharge', 'STILLINHOSP': 'Still in Hospital > 30 Days', 'REOPERATION1': 'Unplanned Reoperation 1', 'RETORPODAYS': 'Days from principal operative procedure to Unplanned Reoperation 1', 'RETOR2PODAYS': 'Days from principal operative procedure to Unplanned Reoperation 2', 'READMPODAYS1': 'Days from principal operative procedure to Any Readmission 1', 'READMPODAYS2': 'Days from principal operative procedure to Any Readmission 2', 'READMPODAYS3': 'Days from principal operative procedure to Any Readmission 3', 'READMPODAYS4': 'Days from principal operative procedure to Any Readmission 4', 'READMPODAYS5': 'Days from principal operative procedure to Any Readmission 5', 'WOUND_CLOSURE': 'Surgical wound closure', 'OTHCDIFF': 'Occurrences Clostridium Difficile (C.diff) Colitis', 'NOTHCDIFF': 'Number of C. diff Occurrences', 'DOTHCDIFF': 'Days from operation until C.diff Complication'}
    #content = {'PUFYEAR': 'Year of PUF', 'SEX': 'Gender', 'RACE_NEW': 'New Race', 'ETHNICITY_HISPANIC': 'Ethnicity Hispanic', 'INOUT': 'Inpatient/outpatient', 'AGE': 'Age of patient with patients over 89 coded as 90+', 'ANESTHES': 'Principal anesthesia technique', 'HEIGHT': 'Height in inches', 'WEIGHT': 'Weight in lbs', 'DIABETES': 'Diabetes mellitus with oral agents or insulin', 'SMOKE': 'Current smoker within one year', 'DYSPNEA': 'Dyspnea', 'FNSTATUS2': 'Functional health status Prior to Surgery', 'VENTILAT': 'Ventilator dependent', 'HXCOPD': 'History of severe COPD', 'ASCITES': 'Ascites', 'HXCHF': 'Heart failure (CHF) in 30 days before surgery', 'HYPERMED': 'Hypertension requiring medication', 'RENAFAIL': 'Acute renal failure (pre-op)', 'DIALYSIS': 'Currently on dialysis (pre-op)', 'DISCANCR': 'Disseminated cancer', 'WNDINF': 'Open wound/wound infection', 'STEROID': 'Immunosuppressive Therapy', 'WTLOSS': 'Malnourishment', 'BLEEDIS': 'Bleeding disorder', 'TRANSFUS': 'Preop Transfusion of >= 1 unit of whole/packed RBCs in 72 hours prior to surgery', 'PRSODM': 'Pre-operative serum sodium', 'PRBUN': 'Pre-operative BUN', 'PRCREAT': 'Pre-operative serum creatinine', 'PRALBUM': 'Pre-operative serum albumin', 'PRBILI': 'Pre-operative total bilirubin', 'PRSGOT': 'Pre-operative SGOT', 'PRALKPH': 'Pre-operative alkaline phosphatase', 'PRWBC': 'Pre-operative WBC', 'PRHCT': 'Pre-operative hematocrit', 'PRPLATE': 'Pre-operative platelet count', 'PRPTT': 'Pre-operative PTT', 'PRINR': 'Pre-operative International Normalized Ratio (INR) of PT values', 'PRPT': 'Pre-operative PT', 'EMERGNCY': 'Emergency case', 'ASACLAS': 'ASA classification', 'OPTIME': 'Total operation time', 'TOTHLOS': 'Length of total hospital stay'}
    print(content[key])
        



# %%
