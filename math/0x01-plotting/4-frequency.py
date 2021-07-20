#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
beans = np.arange(0, 101, step=10)
plt.hist(student_grades, bins=beans, edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.axis([0, 100, 0, 30])
plt.xticks(beans)
plt.show()
