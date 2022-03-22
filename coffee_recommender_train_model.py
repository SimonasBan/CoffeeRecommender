import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
import os
# -------
df = pd.read_csv('arabica_data_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df = df[['Aroma', 'Sweetness', 'Aftertaste', 'Acidity', 'Clean.Cup', 'Body', 'Cupper.Points']]
# ------------------
scaler = preprocessing.MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)
scaled_df = pd.DataFrame(d, columns=names)
# --------
input = scaled_df.drop('Cupper.Points', axis=1,inplace=False)
output = scaled_df['Cupper.Points']
X = input.to_numpy()
ya = np.array(output.to_numpy())
y = [[i] for i in ya]
# -------
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)
# -------
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('Cupper.Points')
test_labels = test_features.pop('Cupper.Points')
# -------
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
#----- Output clipper
from tensorflow.keras import backend
def clip(input, maxx, minn):
    return backend.clip(input, minn, maxx)
# -------
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(1),
      layers.Lambda(clip, output_shape="sigmoid", arguments={'maxx': 10, 'minn': 0})
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model
# -------
dnn_model = build_and_compile_model(normalizer)
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
# plot_loss(history)
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
# ----Result performance
print(test_results['dnn_model'])
# ---draw predictions plot-----
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Cupper points]')
plt.ylabel('Predictions [Cupper points]')
lims = [6, 10]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
# -------
# Save dnn model to a local file
directory = os.getcwd()
dnn_model.save(f"{directory}/models")
# -----tests---
test1_res = dnn_model.predict([7, 7, 7, 7, 7,7])
print(f"Test 7, 7, 7, 7, 7,7 results: {test1_res}")
test2_res = dnn_model.predict([9, 7, 7, 7, 7,7])
print(f"Test 9, 7, 7, 7, 7,7 results: {test2_res}")
test3_res = dnn_model.predict([7, 9, 7, 7, 7,7])
print(f"Test 7, 9, 7, 7, 7,7 results: {test3_res}")
test4_res = dnn_model.predict([7, 7, 9, 7, 7,7])
print(f"Test 7, 7, 9, 7, 7,7 results: {test4_res}")
test5_res = dnn_model.predict([7, 7, 7, 9, 7,7])
print(f"Test 7, 7, 7, 9, 7,7 results: {test5_res}")
test6_res = dnn_model.predict([7, 7, 7, 7, 9, 7])
print(f"Test 7, 7, 7, 7, 9, 7 results: {test6_res}")
test7_res = dnn_model.predict([7, 7, 7, 7, 7, 9])
print(f"Test 7, 7, 7, 7, 7, 9 results: {test7_res}")
test8_res = dnn_model.predict([10, 10, 10, 10, 10, 10])
print(f"Test 10, 10, 10, 10, 10, 10 results: {test8_res}")