---
layout: home_layout
---
### [](#header-2)What is Multi Layer Perceptron

A multilayer perceptron [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a class of feedforward artificial neural network. An MLP consists of at least three layers of nodes. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

Multilayer perceptrons are sometimes colloquially referred to as **vanilla** neural networks, especially when they have a single hidden layer.

In the case of regression problems, the average of the predicted attribute may be returned. In the case of classification, the most prevalent class may be returned.

![Img](https://cdn-images-1.medium.com/max/2000/1*bhFifratH9DjKqMBTeQG5A.gif){:height="36px" width="36px"}

### [](#header-3)MLP Simple Implementation using Keras

Import keras
```python

from __future__ import print_function
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
```

Load dataset
```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Explore the shape of dataset
```python
print (x_train.shape)
# (60000, 28, 28)
```