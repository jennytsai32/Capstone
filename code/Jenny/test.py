
#%%
import pandas as pd
from datasci import datasci

df = pd.read_csv('test.csv')

df.head()

#%%
df.dtypes

#%%
df['SICK'] = df['SICK'].astype('object')
print(df.dtypes)
print(df.SICK)
# %%
m = datasci(df)
m.size()

# %%

m.impute_all()

# %%
