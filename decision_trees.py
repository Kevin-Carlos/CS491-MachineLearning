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
X = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 1]])
Y = np.array([[0], [1], [1], [0], [0], [1], [0], [0], [1], [0]])


# function that takes in Data in X and Y format and returns the entropy of the set
def calcEntropy(X, Y):
    # Calculate Entropy
    no = 0
    yes = 0

    for item in Y:
        if (item == 0):
            no += 1
        if (item == 1):
            yes += 1

    totalCount = no + yes

    yesSide1 = (yes/totalCount) 
    noSide1 = (no/totalCount) 

    try:
        yesSide2 = math.log2(yes/totalCount)
    except:
        yesSide2 = 0

    try:
        noSide2 = math.log2(no/totalCount)
    except:
        noSide2 = 0

    yesSide = yesSide1 * yesSide2
    noSide = noSide1 * noSide2

    entropy = -1 * noSide - yesSide
    return entropy


def calcIG(H, p_L, p_R, H_l, H_r):
    # Calculate IG for split
    summation = p_R * H_r + p_L * H_l
    IG = H - summation
    return IG

def split(X, Y, max_depth, num_splits, featureList):
    ### Base Cases
    # ------------

    # If max depth is hit OR the feature List is empty aka no more features to split on return leaf node classifying the higher percentage class
    if num_splits == max_depth or not featureList:
        no = 0
        yes = 0
        for item in Y:
            if (item == 0):
                no += 1
            if (item == 1):
                yes += 1
        if yes >= no:
            return ([1, 1], None, None)
        else:
            return ([1, 0], None, None)


    H = calcEntropy(X, Y) # H()
    print("H():", H)

    # If Entropy == 0 return leaf node classifying the only class in the data
    if H == 0:
        return ([1, int(Y[0])], None, None )

    ### END BASE CASES

    highest_IG = -99999999

    for num in featureList:
        X_l = []
        X_r = []
        Y_l = []
        Y_r = []
        i = 0
        trindex_l = []
        trindex_r = []

        for row in X:
            # Subset Data to get L and R sides
            if row[num] == 0:
                # This row belongs in X_l
                X_l.append(row)
                Y_l.append(Y[i])

            elif row[num] == 1:
                # This row belongs in X_r
                X_r.append(row)
                Y_r.append(Y[i])
            
            i += 1

        X_l = np.array(X_l)
        X_r = np.array(X_r)
        Y_l = np.array(Y_l)
        Y_r = np.array(Y_r)

        # Calculate probability of being on each side (row wise)
        samples_r = X_r.shape[0]
        samples_l = X_l.shape[0]
        total_samples = samples_l + samples_r

        p_L = samples_l / total_samples
        p_R = samples_r / total_samples

        # Calculate Entropy of each side
        # print("Entropy on each side for feature", num + 1)

        h_l = calcEntropy(X_l, Y_l)
        h_r = calcEntropy(X_r, Y_r)

        # print("\tEntropy for L:", h_l)
        # print("\tEntropy for R:", h_r)
        # Calculate the Information Gain For the Split
        IG = calcIG(H, p_L, p_R, h_l, h_r)
        # print("IG FOR FEATURE", num + 1, "SPLIT:", IG)

        # Test if highest IG
        if (IG > highest_IG):
            highest_IG = IG
            highest_IG_feature_num = num
            best_X_l = X_l
            best_X_r = X_r
            best_Y_l = Y_l
            best_Y_r = Y_r
            # print("Feature number", num + 1, "has highest IG with a value of", IG)

    num_splits += 1
    # print("Splitting on Feature number", highest_IG_feature_num + 1)

    # Take the feature out of the feature list
    featureList.remove(highest_IG_feature_num)
    # print(featureList)

    # return a recursive call to fill out the tree
    print("Samples Going Left:", best_X_l.shape[0])
    print("Samples Going Right:", best_X_r.shape[0])
    return ([0, highest_IG_feature_num + 1], split(best_X_l, best_Y_l, max_depth, num_splits, featureList), split(best_X_r, best_Y_r, max_depth, num_splits, featureList))

#
# X = feature data as a 2D array, each row is a single sample
# Y = training labels as a 2D array, each row is a single label
# max_depth = max depth for the resulting DT
#
# Entropy:  H() = Summation(c element of C) of -p(c) log_2 p(c)
# IG:           = H - Summation(t) of p(t)H(t)
#
def DT_train_binary(X,Y,max_depth):

    DT = split(X, Y, max_depth, 0, list(range(X.shape[1])))

    return(DT)

print(DT_train_binary(X, Y, 10))