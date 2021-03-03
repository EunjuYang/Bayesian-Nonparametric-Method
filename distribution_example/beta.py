from scipy.stats import beta
from numpy import linspace, exp
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

rho = linspace(.01, .99, 99)
a = [0.5, 2, 1, 2]
b = [0.5, 5, 1, 2]
plt.figure()
for i in range(len(a)):
  density = beta(a[i], b[i]).pdf(rho)
  plt.plot(rho, density)

plt.xlabel('rho')
plt.ylabel('density')
plt.legend(['0.5, 0.5', '2, 5', '1, 1', '2, 2'])
plt.show()

l_rho1, l_rho2 = [], []
a, b = 1, 1,
for _ in range(5):

  rho1 = np.random.beta(a,b)
  rho2 = 1 - rho1
  print(rho1, rho2)
  plt.figure()
  plt.bar([1, 2], [rho1, rho2], width=0.2)
  plt.xlabel('rho')
  plt.xlim(0, 3)
  plt.ylim(0, 1)
  plt.title(f'a: {a} b: {b}')
  plt.show()