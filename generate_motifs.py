import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


Wp, Ws, Wtop = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3])

num_h = len(Wtop[0])
hidden_primary = len(Wp[0])
hidden_secondary = len(Ws[0])

train_err, test_err = np.loadtxt(sys.argv[4]), np.loadtxt(sys.argv[5])
prim_dict, sec_dict = sys.argv[6], sys.argv[7]
alpha, beta = float(sys.argv[7]), float(sys.argv[8])


p_dict, s_dict = {}, {}

with open(prim_dict, 'r') as fn:
    for i, line in enumerate(fn):
        p_dict[i] = line.rstrip()

with open(sec_dict, 'r') as fn:
    for i, line in enumerate(fn):
        s_dict[i] = line.rstrip()

#Implements the sigmoid activation function of a vector
def SigmoidVec(input):
    e = np.exp(np.multiply(-1, input))
    denom = np.add(1, e)
    return np.divide(1, denom)

#Implements Softmax activation function of a vector
def SoftmaxVec(b):
    e = np.exp(b - np.max(b))
    return e / e.sum()

#Generate 15,000 sampled motifs, initialized from random htop (throw the first sample)
htop_sampled = np.rint(np.random.rand(num_h))
htop_sampled[0] = 1
sampled_p = []
sampled_s = []

i = 0
while i <= 15000:
    #Generate sample starting from the Top
    vtop_neg = SigmoidVec(np.dot(htop_sampled, np.transpose(Wtop)))
    vtop_neg[0] = 1
    htop_neg = SigmoidVec(np.dot(vtop_neg, Wtop))
    htop_neg = None

    #Un-concatenate top layer and split into primary + secondary vectors
    hp_sampled = vtop_neg[:hidden_primary]
    hs_sampled = vtop_neg[hidden_secondary:-1]
    vtop_neg = None

    #Negatives for Primary
    vp_neg = SigmoidVec(np.dot(hp_sampled, np.transpose(Wp)))
    vp_neg[0] = 1
    hp_neg = SigmoidVec(np.dot(vp_neg, Wp))
    hp_sampled = None

    #Negatives for Secondary
    vs_neg = SigmoidVec(np.dot(hs_sampled, np.transpose(Ws)))
    vs_neg[0] = 1
    hs_neg = SigmoidVec(np.dot(vs_neg, Ws))
    hs_sampled = None

    #Now we sampled the highest probability "word" from each of the two sets
    p_samp = SoftmaxVec(vp_neg[1:])
    s_samp = SoftmaxVec(vs_neg[1:])

    p_threshold = (1 / len(p_samp)) + (alpha * np.std(p_samp))
    s_threshold = (1 / len(s_samp)) + (beta * np.std(s_samp))


    sampled_p.extend(np.ndarray.tolist(np.ndarray.flatten(np.argwhere(p_samp > p_threshold))))
    sampled_s.extend(np.ndarray.tolist(np.ndarray.flatten(np.argwhere(s_samp > s_threshold))))



    #Now regenerate h_top from sampled v
    #Sample hidden states for primary
    hp_sampled =  hp_neg > np.random.rand(len(hp_neg))
    hp_neg = None

    #Generate samples for secondary
    hs_sampled =  hs_neg > np.random.rand(len(hs_neg))
    hs_neg = None

    #Concatenate hidden state vectors from primary and secondary
    vtop = np.append(hp_sampled, hs_sampled)
    #Add bias units
    vtop = np.insert(vtop, 0, 1, axis = 0)
    htop = SigmoidVec(np.dot(vtop, Wtop))
    htop[0] = 1
    htop_sampled =  htop > np.random.rand(len(htop))
    vtop = None
    i += 1

sampled_p = np.array(sampled_p, dtype = int)
sampled_s = np.array(sampled_s, dtype = int)

#Now, using our vectors and dictionaries, we can construct 'artificial' reads to pass into WebLogo
p_string = []
s_string = []
for i in range(len(sampled_p)):
    p_string.append(p_dict[sampled_p[i]])
for j in range(len(sampled_s)):
    s_string.append(s_dict[sampled_s[j]])

with open("primary_motifs.txt", 'w') as fn:
    for str in p_string:
        fn.write("%s\n" % str)
with open("secondary_motifs.txt", 'w') as fn:
    for str in s_string:
        fn.write("%s\n" % str)


#Now we will graph the train/test error as a function of number of iterations
primary_train = np.transpose(train_err)[0]
sec_train = np.transpose(train_err)[1]
primary_test = np.transpose(test_err)[0]
sec_test = np.transpose(test_err)[1]
plt.clf()
plt.plot(np.arange(len(primary_train)), primary_train, 'b', label='primary (train)')
plt.plot(np.arange(len(primary_test)), primary_test, 'r', label='primary (test)')
plt.plot(np.arange(len(sec_train)), sec_train, 'g', label='secondary (train)')
plt.plot(np.arange(len(sec_test)), sec_test, 'm', label='secondary (test)')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Reconstruction Error (MSE)')
plt.show()
