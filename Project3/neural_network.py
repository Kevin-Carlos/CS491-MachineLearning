import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import time

def softmax(vals):
  numerator = np.exp(vals)
  denominator = np.sum(numerator, axis=1, keepdims=True)

  result = numerator / denominator

  return result

# Helper  function  to  predict  an output  (0  or  1)
# model  is  the  current  version  of  the  model {’W1’: W1, ’b1’: b1 , ’W2’: W2, ’b2’: b2 } It’s a dictionary
# x  is  one sample  (without  the  label)
def predict(model, x):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  
  a = x.dot(W1) + b1
  h = np.tanh(a)
  z = h.dot(W2) + b2
  y_hat = softmax(z)

  prediction = np.argmax(y_hat, axis=1)

  return prediction 

def calculate_loss(model, X, y):
  loss = 0
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  new_y = encode(y)

  a = X.dot(W1) + b1
  h = np.tanh(a)
  z = h.dot(W2) + b2
  y_hat = softmax(z)

  # Calculate the loss
  for i in range(X.shape[0]):
    for j in range(new_y.shape[1]):
      loss += -(new_y[i][j] * np.log(y_hat[i][j]))
  
  loss /= X.shape[0]


  return loss

def encode(y):
  new_y = np.zeros((200,2), dtype=int)
  for i in range(len(y)):
    if (y[i] == 0):
      new_y[i][0] = 1
    else:
      new_y[i][1] = 1
  
  return new_y

# This  function  learns  parameters  for  the  neural  network and  returns  the model 
# − X is  the  training  data
# − y  is  the  training  labels 
# - nnhdim :  Number of  nodes  in  the  hidden  layer
# - numpasses :  Number of  passes  through  the  training  data  for  gradient descent
# − printloss :  If  True ,  print  the  loss  every  1000  iterations
eta = 0.03
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):

  # Randomly initialize input layer weights
  W1 = np.random.rand(2, nn_hdim)
  b1 = np.random.rand(1, nn_hdim)
  W2 = np.random.rand(nn_hdim, 2)
  b2 = np.random.rand(1, 2)

  # Encode the y's
  new_y = encode(y)

  for i in range(num_passes):
    # Forward propagation
    a = X.dot(W1) + b1
    h = np.tanh(a)
    z = h.dot(W2) + b2
    y_hat = softmax(z)

    # Try encoding the y_hats
    new_y_hat = np.zeros((200,2), dtype=int)
    for k in range(y_hat.shape[0]):
      if (np.argmax(y_hat[k]) == 0):
        new_y_hat[k][0] = 1
      else:
        new_y_hat[k][1] = 1

    # dLdy_hat = new_y_hat - new_y
    dLdy_hat = y_hat - new_y
    dLda = (1 - pow(h, 2)) * dLdy_hat.dot(W2.T)
    dLdW2 = (h.T).dot(dLdy_hat)
    dLdb2 = np.sum(dLdy_hat)
    dLdW1 = (X.T).dot(dLda)
    dLdb1 = np.sum(dLda)

    # Update weights/biases
    W1 = W1 - (eta * dLdW1)
    W2 = W2 - (eta * dLdW2)
    b1 = b1 - (eta * dLdb1)
    b2 = b2 - (eta * dLdb2)

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
    if (i % 1000 == 0):
      print_loss = True
    
    if (print_loss):
      print("Loss after iteration %i: %f" %(i, calculate_loss(model, X, y)))
      print_loss = False

  return model



def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and the training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[: ,0], X[: ,1], s=40, c=y, cmap=plt.cm.Spectral)

# Generate dataset
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[: ,0], X[: ,1], s=40, c=y, cmap=plt.cm.Spectral)

# Generate outputs
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = build_model(X, y, nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x), X, y)

plt.tight_layout()
plt.show()
plt.savefig("./outputfig.png")