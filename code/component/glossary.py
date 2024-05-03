#%%

import pandas as pd

df = pd.read_csv('/output/var_def.csv')
#print(df.head())
#print(df.shape)


dict = {}
for i in range(df.shape[0]):
    dict[df.Name[i]]=df.Definition[i]
print(dict)

'''
def glossary(key):
    content = dict
    print(content[key])
'''
# %%
