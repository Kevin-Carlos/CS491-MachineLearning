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


#
# X = feature data as a 2D array, each row is a single sample
# Y = training labels as a 2D array, each row is a single label
# max_depth = max depth for the resulting DT
#
# Entropy:  H() = Summation(c element of C) of -p(c) log_2 p(c)
# IG:           = H - Summation(t) of p(t)H(t)
#
def DT_train_binary(X,Y,max_depth):
