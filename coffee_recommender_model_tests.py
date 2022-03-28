import os
from tensorflow import keras
# Setup
directory = os.getcwd()
model = keras.models.load_model(f"{directory}/models")
# -----
# -----Testing of individual parameters
test1_res = model.predict([7, 7, 7, 7, 7,7])
print(f"Test 7, 7, 7, 7, 7,7 results: {test1_res}")