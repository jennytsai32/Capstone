#pip install DataSynthesizer

import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network

df = pd.read_csv('/processed_data/2015_2022/CABG_8yr_preselect41.csv', header=0)

print(df.head())
print(df.shape)


# Backup the original dataset
df_backup = df.copy()

# Specify categorical attributes
categorical_attributes = {'OTHBLEED':True,
                          "PUFYEAR": True,
                          "SEX": True,
                          "RACE_NEW": True,
                          "INOUT": True,
                          "ANESTHES": True,
                          "DIABETES": True,
                          "SMOKE": True,
                          "DYSPNEA": True,
                          "FNSTATUS2": True,
                          "VENTILAT": True,
                          "HXCOPD": True,
                          "ASCITES": True,
                          "HXCHF": True,
                          "HYPERMED": True,
                          "RENAFAIL": True,
                          "DIALYSIS": True,
                          "DISCANCR": True,
                          "WNDINF": True,
                          "STEROID": True,
                          "WTLOSS": True,
                          "BLEEDIS": True,
                          "TRANSFUS": True,
                          "EMERGNCY": True,
                          "ASACLAS": True
                          }

# Define privacy settings
epsilon = 0.1
degree_of_bayesian_network = 2
num_tuples_to_generate = 13534

# Initialize DataDescriber with category threshold
describer = DataDescriber(category_threshold=20)
print('Initialization completed.')

# Describe the dataset to create a Bayesian network
describer.describe_dataset_in_correlated_attribute_mode(dataset_file='CABG_8yr_preselect41.csv',
                                                        epsilon=epsilon,
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes
                                                        )

print('Describer work completed.')

# Save dataset description to a JSON file
description_file = 'CABG_description.json'
describer.save_dataset_description_to_file(description_file)

print('Saved data discription.')

# Display the Bayesian network
display_bayesian_network(describer.bayesian_network)

print('Display Baysian network completed.')

# Generate the dataset
generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)

print('Data generation completed.')

# Save synthetic data to a CSV file
synthetic_data_file = 'CABG_synthetic_Bayesian_8yr.csv'
generator.save_synthetic_data(synthetic_data_file)

print('Synthetic data saved.')

# Check synthetic data file
df_synthetic = pd.read_csv(synthetic_data_file)
print(df_synthetic.head())