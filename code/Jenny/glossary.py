#%%

import pandas as pd

df = pd.read_csv('var_def.csv')
#print(df.head())
#print(df.shape)


dict = {}
for i in range(df.shape[0]):
    dict[df.Name[i]]=df.Definition[i]
print(dict)

'''
def glossary(key):
    content = {'PUFYEAR': 'Year of PUF', 'SEX': 'Gender', 'RACE_NEW': 'New Race', 'ETHNICITY_HISPANIC': 'Ethnicity Hispanic', 'INOUT': 'Inpatient/outpatient', 'AGE': 'Age of patient with patients over 89 coded as 90+', 'ANESTHES': 'Principal anesthesia technique', 'HEIGHT': 'Height in inches', 'WEIGHT': 'Weight in lbs', 'DIABETES': 'Diabetes mellitus with oral agents or insulin', 'SMOKE': 'Current smoker within one year', 'DYSPNEA': 'Dyspnea', 'FNSTATUS2': 'Functional health status Prior to Surgery', 'VENTILAT': 'Ventilator dependent', 'HXCOPD': 'History of severe COPD', 'ASCITES': 'Ascites', 'HXCHF': 'Heart failure (CHF) in 30 days before surgery', 'HYPERMED': 'Hypertension requiring medication', 'RENAFAIL': 'Acute renal failure (pre-op)', 'DIALYSIS': 'Currently on dialysis (pre-op)', 'DISCANCR': 'Disseminated cancer', 'WNDINF': 'Open wound/wound infection', 'STEROID': 'Immunosuppressive Therapy', 'WTLOSS': 'Malnourishment', 'BLEEDIS': 'Bleeding disorder', 'TRANSFUS': 'Preop Transfusion of >= 1 unit of whole/packed RBCs in 72 hours prior to surgery', 'PRSODM': 'Pre-operative serum sodium', 'PRBUN': 'Pre-operative BUN', 'PRCREAT': 'Pre-operative serum creatinine', 'PRALBUM': 'Pre-operative serum albumin', 'PRBILI': 'Pre-operative total bilirubin', 'PRSGOT': 'Pre-operative SGOT', 'PRALKPH': 'Pre-operative alkaline phosphatase', 'PRWBC': 'Pre-operative WBC', 'PRHCT': 'Pre-operative hematocrit', 'PRPLATE': 'Pre-operative platelet count', 'PRPTT': 'Pre-operative PTT', 'PRINR': 'Pre-operative International Normalized Ratio (INR) of PT values', 'PRPT': 'Pre-operative PT', 'EMERGNCY': 'Emergency case', 'ASACLAS': 'ASA classification', 'OPTIME': 'Total operation time', 'TOTHLOS': 'Length of total hospital stay'}
    print(content[key])
'''
# %%
