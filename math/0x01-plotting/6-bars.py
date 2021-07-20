#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
names = ['Farrah', 'Fred', 'Felicia']
bars = [0, 1, 2]
wid = 0.5

plt.bar(names, fruit[0], color='red', width=wid, label='apples')
plt.bar(names, fruit[1], bottom=fruit[0], color='yellow', width=wid,
        label='bananas')
plt.bar(names, fruit[2], bottom=fruit[0] + fruit[1], color='#ff8000',
        width=wid, label='oranges')
plt.bar(names, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2],
        color='#ffe5b4', width=wid, label='peaches')
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.xticks(bars, names)
plt.yticks(np.arange(0, 90, step=10))
plt.legend(loc='upper right')

plt.show()
