import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import time


# This  function  learns  parameters  for  the  neural  network and  returns  the model 
# − X is  the  training  data
# − y  is  the  training  labels 
# - nnhdim :  Number of  nodes  in  the  hidden  layer
# - numpasses :  Number of  passes  through  the  training  data  for  gradient descent
# − printloss :  If  True ,  print  the  loss  every  1000  iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):

    # Initialize input layer weights
    W = np.zeros_like(np.multiply(X, nn_hdim))
    W = np.random.uniform(0, 10, (W.shape[0], W.shape[1]))

    # Initialize output layer weights

    eta = 0.01

    # for iterator in range(1, num_passes):
    #     G = np.zeros_like()













# def plot_decision_boundary(pred_func, X, y):
#     # Set min and max values and give it some padding
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     h = 0.01
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole grid
#     Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

# Generate dataset
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# Generate outputs
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('HiddenLayerSize%d' %nn_hdim)
    model = build_model(X, y, nn_hdim)
    # plot_decision_boundary(lambda x: predict(model, x), X, y)

plt.show()