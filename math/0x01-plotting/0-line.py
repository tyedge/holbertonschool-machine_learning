#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
plt.plot(y, color='red', linestyle='solid')
plt.axis([0, 10, None, None])
plt.show()
