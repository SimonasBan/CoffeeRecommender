import matplotlib.pyplot as plt
import numpy as np

test_len = np.arange(0,11,1)
# a = plt.axes(None)
plt.scatter(test_len, test_len)
plt.xlabel(f"parametrui suteikta reikšmė")
plt.ylabel('Išvestis')
plt.show()