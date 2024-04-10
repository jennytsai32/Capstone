#%%
from realtabformer import REaLTabFormer
import pandas as pd

df = pd.read_csv('processed_data/2015_2022/CABG_8yr_preselect41.csv', header=0)

df.head()
df.shape

#%%
df2 = df.iloc[0:50,:]
df2.shape


#%%
# Non-relational or parent table.
rtf_model = REaLTabFormer(
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

#%%
# Fit the model on the dataset.
# Additional parameters can be
# passed to the `.fit` method.
rtf_model.fit(df)

#%%
# Save the model to the current directory.
# A new directory `rtf_model/` will be created.
# In it, a directory with the model's
# experiment id `idXXXX` will also be created
# where the artefacts of the model will be stored.
rtf_model.save("output/rtf_model/")

# Generate synthetic data with the same
# number of observations as the real dataset.
samples = rtf_model.sample(n_samples=len(df2))

samples.to_csv('/processed_data/2015_2022/realtabformer_data.csv', index=False)

#%%
# Load the saved model. The directory to the
# experiment must be provided.
#rtf_model2 = REaLTabFormer.load_from_dir(path="output/rtf_model/id000017122473330058563584")