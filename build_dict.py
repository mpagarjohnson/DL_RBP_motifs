import itertools
import sys


#If run_flag == 1, Builds a dictionary of all k-mers, taking k and the outfile name as CL inputs
#If run_flag == 2, Builds a dictionary using the secondary structure alphabet, of size 8.

fn = sys.argv[1]
run_flag = int(sys.argv[2])
try:
    k = int(sys.argv[3])
except:
    print("k not specified, using default value of 6.")
    k = 6



if run_flag == 1:
    prod = itertools.product('AUGC', repeat = k)
    dict = []

    for tuple in prod:
        string = ''.join(tuple)
        dict.append(string)

    with open(fn, 'w') as f:
        for i in dict:
            f.write(i + "\n")
        print("Saved to", fn)
elif run_flag == 2:
    prod = itertools.product('ESHBM', repeat = k)
    dict = []

    for tuple in prod:
        string = ''.join(tuple)
        dict.append(string)

    with open(fn, 'w') as f:
        for i in dict:
            f.write(i + "\n")
        print("Saved to", fn)
