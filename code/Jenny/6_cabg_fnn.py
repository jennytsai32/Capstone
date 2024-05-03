
# references: tensorflow documentation

"""## Import tensorflow and libraries"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


"""# Helper Functions"""

# convert dataframe to tf.data
def dataframe_to_dataset(dataframe, target):
  '''
  convert pandas dataframe to tf.data
  '''
  dataframe = dataframe.copy()
  if multi_class==False:
    labels = dataframe.pop(target)
  else:
    out = dataframe.pop(target).values
    labels = np.concatenate(out).ravel().reshape(dataframe.shape[0], -1)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  ds = ds.shuffle(buffer_size=len(dataframe))
  return ds

# FNN functions
def build_model_5_layer(output_func, n):
  '''
  build 5-layer binary classification model
  '''
  encoded_features = feature_space.get_encoded_features()
  x = keras.layers.Dense(32, activation="relu")(encoded_features)
  x = keras.layers.Dense(50, activation="relu")(x)
  x = keras.layers.Dropout(0.2)(x)
  predictions = keras.layers.Dense(n, activation=output_func)(x)
  model = keras.Model(inputs=encoded_features, outputs=predictions)

  return model


def build_model_multi_layer(output_func, n):
  '''
  build 7-layer binary classification model with more neurons
  '''
  encoded_features = feature_space.get_encoded_features()
  x = keras.layers.Dense(32, activation="relu")(encoded_features)
  x = keras.layers.Dense(50, activation="relu")(x)
  x = keras.layers.Dense(100, activation="relu")(x)
  x = keras.layers.Dense(50, activation="relu")(x)
  x = keras.layers.Dropout(0.2)(x)
  predictions = keras.layers.Dense(n, activation=output_func)(x)
  model = keras.Model(inputs=encoded_features, outputs=predictions)

  return model


def compile_model(model, opt, loss_func):
  '''
  complie model using the opitimizer and loss function of choice
  '''

  model.compile(
      optimizer=opt,
      loss="binary_crossentropy",
      metrics=['accuracy',
              tf.keras.metrics.F1Score(average='macro', threshold=0.5),
              tf.keras.metrics.RootMeanSquaredError(),
              tf.keras.metrics.AUC()])
  return model

# create lists to save best scores
best_val_acc = []
best_val_f1 = []
best_val_rmse = []
best_val_auc = []

# save best scores from training history
def save_best_scores():
  best_val_acc.append(max(history.history['val_accuracy']))
  best_val_f1.append(max(history.history['val_f1_score']))
  best_val_rmse.append(max(history.history['val_root_mean_squared_error']))
  best_val_auc.append(max(history.history['val_auc'+num]))

# encode target for multi-class modeling (for softmax)
def encode_target(Y):
  encoder = LabelEncoder()
  encoder.fit(Y)
  encoded_Y = encoder.transform(Y)
  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_y = to_categorical(encoded_Y)

  return dummy_y

"""# Hyperparamers"""

dataset_name = '8_year_data'
#dataset_name = 'realtabformer'
#dataset_name = 'datasynthesizer'
multi_class = True
batch_size = 32
train_val_split_ratio = 0.2

if multi_class==False:
  target = 'OTHBLEED'
  output_func='sigmoid'
  n=1
  loss_func="binary_crossentropy"

else:
  target = 'OTHBLEED_new'
  output_func='softmax'
  n=2
  loss_func="categorical_crossentropy"

"""# Data Preprocessing"""

# import data
df = pd.read_csv('/processed_data/2015_2022/CABG_8yr_preselect41.csv')
#df = pd.read_csv('/processed_data/2015_2022/realtabformer_data.csv')
#df = pd.read_csv('/processed_data/2015_2022/CABG_synthetic_Bayesian_8yr.csv')
print(df.head())
print(df.shape)

if multi_class==False:
  # convert target dtype to float32 for computing f1 score
  df['OTHBLEED'] = df['OTHBLEED'].astype(np.float32)

else:
  dummy_y = encode_target(df['OTHBLEED'])
  df['OTHBLEED_new'] = pd.Series(list(dummy_y))
  df = df.drop('OTHBLEED',axis=1)

# split into training and test data
val_dataframe = df.sample(frac=train_val_split_ratio, random_state=1337)
train_dataframe = df.drop(val_dataframe.index)

print(f"Using {len(train_dataframe)} samples for training and {len(val_dataframe)} for validation")


# convert dataframe to tf.data
train_ds = dataframe_to_dataset(train_dataframe, target)
val_ds = dataframe_to_dataset(val_dataframe, target)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

# batch
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# set up feature space
feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers (25)
        "PUFYEAR": "integer_categorical",
        "SEX": "integer_categorical",
        "RACE_NEW": "integer_categorical",
        "INOUT": "integer_categorical",
        "ANESTHES": "integer_categorical",
        "DIABETES": "integer_categorical",
        "SMOKE": "integer_categorical",
        "DYSPNEA": "integer_categorical",
        "FNSTATUS2": "integer_categorical",
        "VENTILAT": "integer_categorical",
        "HXCOPD": "integer_categorical",
        "ASCITES": "integer_categorical",
        "HXCHF": "integer_categorical",
        "HYPERMED": "integer_categorical",
        "RENAFAIL": "integer_categorical",
        "DIALYSIS": "integer_categorical",
        "DISCANCR": "integer_categorical",
        "WNDINF": "integer_categorical",
        "STEROID": "integer_categorical",
        "WTLOSS": "integer_categorical",
        "BLEEDIS": "integer_categorical",
        "TRANSFUS": "integer_categorical",
        "EMERGNCY": "integer_categorical",
        "ASACLAS": "integer_categorical",
        "OTHERCPT1": "integer_categorical",


        # Numerical features (16)
        "AGE": "float",
        "BMI": "float",
        "PRSODM": "float",
        "PRBUN": "float",
        "PRCREAT": "float",
        "PRALBUM": "float",
        "PRBILI": "float",
        "PRSGOT": "float",
        "PRALKPH": "float",
        "PRWBC": "float",
        "PRHCT": "float",
        "PRPLATE": "float",
        "PRPTT": "float",
        "PRINR": "float",
        "OPTIME": "float",
        "TOTHLOS": "float"
    },

    # one-hot encode all categorical features and concat all features into a single vector (one vector per sample).
    output_mode="concat",
)

# adapt the features (transform features based on feature space configuration)
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

# check feature size and dtype
for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)

# create train and val dataset and prefetch
preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

"""# Training

## 1. FNN with 5 layers, Optimizer = SGD
"""

# build and compile model
model = build_model_5_layer(output_func,n)
model = compile_model(model, opt='sgd',loss_func=loss_func)

# model summary
model.summary()

# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# create a figure to show loss and performance over training time
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

"""## 2. FNN with 5 layers, Optimizer = Adam"""

# build and compile model
model = build_model_5_layer(output_func=output_func,n=n)
model = compile_model(model, opt='adam',loss_func=loss_func)

# model summary
model.summary()


# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

"""## 3. FNN with 10 layers, more neurons, Optimizer = SGD"""

# build and compile model
model = build_model_multi_layer(output_func, n)
model = compile_model(model, opt='sgd',loss_func=loss_func)

# model summary
model.summary()


# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

"""## 4. FNN with 10 layers, more neurons, Optimizer = Adam"""

# build and compile model
model = build_model_multi_layer(output_func,n)
model = compile_model(model, opt='adam',loss_func=loss_func)

# model summary
model.summary()


# Train, evaluate and save the best model
history = model.fit(
    preprocessed_train_ds,
    epochs=100,
    validation_data=preprocessed_val_ds,
    verbose=2)

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
plt.show()

# save best scores from training
num="" if len(model.name)==5 else model.name[-2:]
save_best_scores()

# best scores summary from all models
summary = pd.DataFrame.from_dict(
    {'Model':['FNN-5layer-SGD-'+output_func,'FNN-5layer-Adam-'+output_func,'FNN-multilayer-SGD-'+output_func,'FNN-multilayer-Adam-'+output_func],
     'accuracy':best_val_acc,
     'f1_score':best_val_f1,
     'rMSE':best_val_rmse,
     'AUC':best_val_auc}
                       )
print(summary)

#summary.to_csv('/output/summary_'+output_func+'_'+dataset_name+'.csv')