import pandas as pd
import numpy as np

from classifiers.perceptron import Perceptron

import matplotlib.pyplot as plt
# URL of the Iris dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

df = pd.read_csv(url, header=None, names=column_names, encoding='utf-8')

print(df.shape)


y= df.iloc[0:100, -1].values
y=np.where(y=='Iris-setosa', 1, 0)
X= df.iloc[0:100, 0:2].values

pp = Perceptron()
pp.fit(X,y)

print(pp.errors_)

#plot function given by GPT
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x')
    colors = ('lightblue', 'lightgreen')
    cmap = ListedColormap(colors)

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'Class {cl}')

    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Sepal width [cm]')
    plt.legend(loc='upper left')
    plt.show()

# Call this after training
plot_decision_regions(X, y, classifier=pp)
