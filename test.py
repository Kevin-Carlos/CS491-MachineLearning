import decision_trees as dt
import numpy as np

# Test Data
# X = np.array([[0,1], [0,0], [1,0], [0,0], [1,1]]) # Training Set 1
# Y = np.array([[1], [0], [0], [0], [1]]) # Training labels 1

X = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 1]])
Y = np.array([[0], [1], [1], [0], [0], [1], [0], [0], [1], [0]])

dt.DT_train_binary(X, Y, 1)