#
# Authors:  Kevin Carlos and James Schnebly
#
# Due date: 9/26/19 @ 3:30 pm
#
# Details:  Implement a decision tree using these functions:
#               DT_train_binary(X,Y,max_depth)
#               DT_test_binary(X,Y,DT)
#               DT_train_binary_best(X_train, Y_train, X_val, Y_val)
#               DT_make_prediction(x,DT)
#               DT_train_real(X,Y,max_depth)
#               DT_test_real(X,Y,DT)
#               DT_train_real_best(X_train,Y_train,X_val,Y_val)

import numpy as np
import math

# Test Data
# X = np.array([[0,1], [0,0], [1,0], [0,0], [1,1]]) #Training Set 1
# Y = np.array([[1], [0], [0], [0], [1]]) # Training labels 1

# Entropy Test Data
X = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0]])
Y = np.array([[0], [1], [1], [0], [0], [1], [0], [0], [1], [0]])


# function that takes in Data in X and Y format and returns the entropy of the set
def calcEntropy(X, Y):
    # Calculate Entropy for the entire set
    no = 0
    yes = 0

    for item in Y:
        if (item == 0):
            no += 1
        if (item == 1):
            yes += 1

    totalCount = no + yes

    entropy = -(no/totalCount) * math.log2(no/totalCount) - (yes/totalCount) * math.log2(yes/totalCount)
    return entropy

#
# X = feature data as a 2D array, each row is a single sample
# Y = training labels as a 2D array, each row is a single label
# max_depth = max depth for the resulting DT
#
# Entropy:  H() = Summation(c element of C) of -p(c) log_2 p(c)
# IG:           = H - Summation(t) of p(t)H(t)
#
def DT_train_binary(X,Y,max_depth):

    # H() = H
    H = calcEntropy(X, Y)

    num_features = X.shape[1]
    splits = 0

    if (max_depth != -1 and max_depth != 0):
        while (splits < max_depth):
            highest_IG = -4567890
            for num in range(num_features):
                # Subset Data to get L and R sides
                X_l =
                X_r =
                Y_l =
                Y_r =

                # Calculate probability of being on each side
                p_L = 0
                p_R = 0

                # Calculate Entropy of each side
                h_l = calcEntropy(X_l, Y_l)
                h_r = calcEntropy(X_r, Y_r)

                # Calculate the Information Gain For the Split
                IG = 0

                # Test if highest IG
                if (IG > highest_IG):
                    highest_IG = IG


            # Take away values that have been labeled after split (if any) and recompute H() and split
            splits += 1


