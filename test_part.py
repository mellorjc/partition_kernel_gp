from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from part_kernel import FastKernel
from numpy import arange, linspace, sin, vstack, argsort
import matplotlib.pyplot as plt

digits = load_boston()
data = digits['data']
data = linspace(0, 10, 500)
#data = vstack([data, data])
labels = digits['target']
labels = sin(data)
train_data, test_data, train_labels, test_labels = train_test_split(data,
                                                                    labels,
                                                                    test_size=0.1)
model = FastKernel(train_data)
model._select_centers(data)
y = arange(10)
# v1 = model.K2(data[:10, :], data[:10, :]).dot(y)
# v2 = model.Ky(data[:10, :], y)
v1 = model.K2(data[:10], data[:10]).dot(y)
v2 = model.Ky(data[:10], y)
print(v1)
print(v2)
p = model.predict_mean(test_data[:20], train_data, train_labels)
print(p)
print(test_labels[:20])

plt.plot(data, labels)
xs = argsort(test_data[:20])
plt.plot(test_data[:20][xs], p[xs])
plt.show()
