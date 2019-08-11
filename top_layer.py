import os
import sys
import math
import numpy as np
import traceback

learning_rate = 0.01

print("Loading input data")
input_primary = np.load(sys.argv[1])
input_secondary = np.load(sys.argv[2])
print("Loading test data")
test_primary = np.load(sys.argv[3])
test_secondary = np.load(sys.argv[4])


#Partition the data to avoid memory error
input_primary = input_primary[:15000]
input_secondary = input_secondary[:15000]
test_primary = test_primary[:15000]
test_secondary = test_secondary[:15000]


#Implements the sigmoid activation function of a vector
def SigmoidVec(input):
    e = np.exp(np.multiply(-1, input))
    denom = np.add(1, e)
    return np.divide(1, denom)

#Implements Softmax activation function of a vector
def SoftmaxVec(b):
    e = np.exp(b - np.max(b))
    return e / e.sum()

#Forward Pass of Cross-Entropy
def CrossEntropyForward(y, yh):
    yt = -np.transpose(y)
    return np.dot(yt, np.log(yh))

#Backprop Pass of Cross Entropy
def CrossEntropyBackward(y, yh, b, gb):
    return -(np.dot(gb, np.divide(y, yh)))

#Backprop Pass of Sigmoid Function
def SigmoidBackward(a, b, g_b):
    one_minb = np.subtract(1, b)
    prod = np.multiply(b, one_minb).reshape(1,len(b))
    return np.multiply(g_b.reshape(1, len(b)), prod)



def ComputeTestError(Wp, Ws, Wtop, primary, secondary, hidden_primary, hidden_secondary, hidden_top_layer):
    #Add bias units
    primary = np.insert(primary, 0, 1, axis = 1)
    secondary = np.insert(secondary, 0, 1, axis = 1)

    #Sample hidden states for primary
    hp_pos = SigmoidVec(np.dot(primary, Wp))
    hp_pos[:,0] = 1
    hp_sampled =  hp_pos > np.random.rand(len(hp_pos), hidden_primary + 1)
    hp_pos = None

    #Generate samples for secondary
    hs_pos = SigmoidVec(np.dot(secondary, Ws))
    hs_pos[:,0] = 1
    hs_sampled =  hs_pos > np.random.rand(len(hs_pos), hidden_secondary + 1)
    hs_pos = None

    #Concatenate hidden state vectors from primary and secondary
    vtop_pos = np.hstack((hp_sampled, hs_sampled))
    hp_sampled, hs_sampled = None, None
    #Add bias units
    vtop_pos = np.insert(vtop_pos, 0, 1, axis = 1)
    htop_pos = SigmoidVec(np.dot(vtop_pos, Wtop))
    vtop_pos = None
    htop_pos[:,0] = 1
    htop_sampled =  htop_pos > np.random.rand(len(htop_pos),  hidden_top_layer + 1)
    htop_pos = None

    #Generate "Negatives" starting from the Top
    vtop_neg = SigmoidVec(np.dot(htop_sampled, np.transpose(Wtop)))
    vtop_neg[:,0] = 1

    #Un-concatenate top layer and split into primary + secondary vectors
    hp_sampled = vtop_neg[:,:hidden_primary + 1]
    hs_sampled = vtop_neg[:, hidden_secondary + 1:-1]
    vtop_neg = None

    #Negatives for Primary
    vp_neg = SigmoidVec(np.dot(hp_sampled, np.transpose(Wp)))
    vp_neg[:,0] = 1

    #Negatives for Secondary
    vs_neg = SigmoidVec(np.dot(hs_sampled, np.transpose(Ws)))
    vs_neg[:,0] = 1

    hp_sampled, hs_sampled = None, None

    #Compute Mean-Squared-Error
    error_p = np.sum((primary - vp_neg) ** 2) / len(primary)
    error_s = np.sum((secondary - vs_neg) ** 2) / len(secondary)

    return error_p, error_s


def TrainRBM(primary, secondary, hidden_primary, hidden_secondary, hidden_top_layer, num_epochs):
    Wp = np.random.normal(0, .01, size = (len(primary[0]) + 1, hidden_primary + 1))
    Ws = np.random.normal(0, .01, size = (len(secondary[0]) + 1, hidden_secondary + 1))
    Wtop = np.random.normal(0, .01, size = (hidden_primary + hidden_secondary + 3, hidden_top_layer + 1))
    Wlabel = np.random.normal(0, .01, size = (hidden_top_layer + 1, 1))

    #Add bias units
    primary = np.insert(primary, 0, 1, axis = 1)
    secondary = np.insert(secondary, 0, 1, axis = 1)



    error_prev = 100000
    train_err, test_err = [], []

    for e in range(num_epochs):
        #Sample hidden states for primary
        hp_pos = SigmoidVec(np.dot(primary, Wp))
        hp_pos[:,0] = 1
        hp_sampled =  hp_pos > np.random.rand(len(hp_pos), hidden_primary + 1)
        Gp_pos = np.dot(np.transpose(primary), hp_pos)
        hp_pos = None

        #Generate samples for secondary
        hs_pos = SigmoidVec(np.dot(secondary, Ws))
        hs_pos[:,0] = 1
        hs_sampled =  hs_pos > np.random.rand(len(hs_pos), hidden_secondary + 1)
        Gs_pos = np.dot(np.transpose(secondary), hs_pos)
        hs_pos = None

        #Concatenate hidden state vectors from primary and secondary
        vtop_pos = np.hstack((hp_sampled, hs_sampled))
        #Add bias units
        vtop_pos = np.insert(vtop_pos, 0, 1, axis = 1)
        htop_pos = SigmoidVec(np.dot(vtop_pos, Wtop))
        htop_pos[:,0] = 1
        htop_sampled =  htop_pos > np.random.rand(len(htop_pos),  hidden_top_layer + 1)

        Gtop_pos = np.dot(np.transpose(vtop_pos), htop_pos)
        vtop_pos = None

        #Determine Label probabilities, update label weights with backpropagation using a cross-entropy objective function
        """print("Computing Label + Backprop Step")
        print(htop_sampled.shape)
        print(Wlabel.shape)
        y_sampled = np.dot(htop_sampled, Wlabel)

        print("Sigmoid of y_sampled")
        y_probs = SigmoidVec(y_sampled)
        print("Cross Entropy Step")
        print(y_probs)
        cross_entropy = CrossEntropyForward(np.ones(y_probs.shape), y_probs)[0][0]
        print("Epoch %s: Cross-Entropy = %s" % (e, cross_entropy))

        print("Backward CE")
        dy_probs = CrossEntropyBackward(np.ones(y_probs.shape), y_probs, cross_entropy, 1)
        print(dy_probs.shape)
        print("Backward y_sampled")
        dy_sampled = SigmoidBackward(y_sampled, y_probs, dy_probs)
        print(dy_sampled.shape)
        print("Backward Wlabel")
        dWlabel = np.dot(dy_sampled, htop_sampled)
        print(dWlabel.shape)
        #Update by gradient descent
        print("Gradient Descent Update")
        Wlabel = np.subtract(Wlabel, (np.multiply(learning_rate, np.transpose(dWlabel))))
        print("Wlabel", Wlabel.shape)
        y_sampled, y_probs, dy_probs, dy_sampled, dWlabel = None, None, None, None, None

        print("Backprop Step Complete")"""

        #Generate "Negatives" starting from the Top
        vtop_neg = SigmoidVec(np.dot(htop_sampled, np.transpose(Wtop)))
        vtop_neg[:,0] = 1
        htop_neg = SigmoidVec(np.dot(vtop_neg, Wtop))
        Gtop_neg = np.dot(np.transpose(vtop_neg), htop_neg)
        htop_neg = None

        #Un-concatenate top layer and split into primary + secondary vectors
        hp_sampled = vtop_neg[:,:hidden_primary + 1]
        hs_sampled = vtop_neg[:, hidden_secondary + 1:-1]
        vtop_neg = None

        #Negatives for Primary
        vp_neg = SigmoidVec(np.dot(hp_sampled, np.transpose(Wp)))
        vp_neg[:,0] = 1
        hp_neg = SigmoidVec(np.dot(vp_neg, Wp))
        Gp_neg = np.dot(np.transpose(vp_neg), hp_neg)
        hp_sampled = None

        #Negatives for Secondary
        vs_neg = SigmoidVec(np.dot(hs_sampled, np.transpose(Ws)))
        vs_neg[:,0] = 1
        hs_neg = SigmoidVec(np.dot(vs_neg, Ws))
        Gs_neg = np.dot(np.transpose(vs_neg), hs_neg)
        hs_sampled = None


        #Update Weights
        Wp += learning_rate * ((Gp_pos - Gp_neg) / len(primary))
        Ws += learning_rate * ((Gs_pos - Gs_neg) / len(secondary))
        Wtop += learning_rate * ((Gtop_pos - Gtop_neg) / len(primary))

        error_p = np.sum((primary - vp_neg) ** 2) / len(primary)
        error_s = np.sum((secondary - vs_neg) ** 2) / len(secondary)

        diff = error_prev - (error_p + error_s)
        if diff <= 0.001:
            print("Converged. Halting...")
            break
        else:
            train_err.append([error_p, error_s])
            test_errp, test_errs = ComputeTestError(Wp, Ws, Wtop, test_primary, test_secondary, hidden_primary, hidden_secondary, hidden_top_layer)
            test_err.append([test_errp, test_errs])
            print("Epoch %s: error (train) == %s %s, error (test) == %s %s" % (e, error_p, error_s, test_errp, test_errs))
            error_prev = error_p + error_s
    return Wp, Ws, Wtop, train_err, test_err


#print("Debugging Softmax")
#DebugSoftmax(input_primary, 500)
print("Training RBM")
Wp, Ws, Wtop, train_err, test_err = TrainRBM(input_primary, input_secondary, 500, 500, 100, 500)

np.savetxt("train_err.txt", np.array(train_err))
np.savetxt("test_err.txt", np.array(test_err))

np.save("Wp", Wp)
np.save("Ws", Ws)
np.save("Wtop", Wtop)
print("Saving to outputs")
