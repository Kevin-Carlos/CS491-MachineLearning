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
    # Calculate Entropy
    no = 0
    yes = 0

    for item in Y:
        if (item == 0):
            no += 1
        if (item == 1):
            yes += 1

    totalCount = no + yes
    try:
        entropy = -(no/totalCount) * math.log2(no/totalCount) - (yes/totalCount) * math.log2(yes/totalCount)
        return entropy
    except:
        return 0 

def calcIG(H, p_L, p_R, H_l, H_r):
    # Calculate IG for split
    summation = p_R * H_r + p_L * H_l
    IG = H - summation
    return IG

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
            highest_IG = -9999
            for num in range(num_features):
                X_l = []
                X_r = []
                Y_l = []
                Y_r = []
                i = 0

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

                # Calculate probability of being on each side
                samples_r = X_r.shape[0]
                samples_l = X_l.shape[0]
                total_samples = samples_l + samples_r

                p_L = samples_l / total_samples
                p_R = samples_r / total_samples

                # Calculate Entropy of each side
                h_l = calcEntropy(X_l, Y_l)
                h_r = calcEntropy(X_r, Y_r)

                # Calculate the Information Gain For the Split
                IG = calcIG(H, p_L, p_R, h_l, h_r)

                # Test if highest IG
                if (IG > highest_IG):
                    highest_IG = IG
                    highest_IG_feature_num = num
                    print("Feature number", num + 1, "has highest IG with a value of", IG)


            # Take away values that have been labeled after split (if any) and recompute H() and split

            splits += 1
            print("Splitting on feature number", highest_IG_feature_num + 1)
        return("Training Complete")

print(DT_train_binary(X, Y, 1))