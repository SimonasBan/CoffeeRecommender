import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import seaborn as sns
df = pd.read_csv('arabica_data_cleaned.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df = df[['Aroma', 'Sweetness', 'Aftertaste', 'Acidity', 'Clean.Cup', 'Body', 'Cupper.Points']]
df.head()
