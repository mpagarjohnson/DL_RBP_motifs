import os
import sys
import numpy as np



input_file = sys.argv[1]
outfile = sys.argv[2]
#control_flag signifies whether or not the experiment is the control data (1 for control, 0 for experimental),
#which will be used to construct the most common motifs.
try:
    control_flag = int(sys.argv[3])
except:
    raise SystemError("Invalid input for control flag: must be set to 0 or 1\n")

if control_flag > 2:
    raise SystemError("Invalid input for control flag: must be set to 0 or 1\n")


try:
    dict_path = sys.argv[4]
except:
    if control_flag == 1:
        raise SystemError("Must provide dictionary path if control flag set to 1\n")
    elif control_flag == 0:
        pass


if control_flag == 0:
    print("Using common_motifs.txt")


k = 0
num_partition = 4 ** 4


#Given a dot-bracket string representing RNA secondary structures,
#Determine structural motifs (abstract shape representations)
def FindShapes(seq):
    shapes = list(seq)
    in_loop = False
    in_multi_loop = False
    num_stems = 0
    num_multi_stems = 0
    idx = 0
    while idx < len(shapes):
        if in_loop == False:
            #If we are not in a loop, unpaired bases are external
            if shapes[idx] == '.':
                shapes[idx] = 'E'
                idx += 1
            #Otherwise, we start the stem portion of a loop
            else:
                shapes[idx] = 'S'
                in_loop = True
                num_stems += 1
                idx += 1
        #In a loop, but not in a multiloop
        elif in_multi_loop == False:
            if shapes[idx] == '.':            #Unpaired base following a stem
                if shapes[idx - 1] == 'S':
                    #Unpaired bases following the stems are forming a loop
                    if shapes[idx + 1] == '.':
                        shapes[idx] = 'H'
                    #If there is only one unpaired base, we classify it as a bulge
                    else:
                        shapes[idx] = 'B'
                    idx += 1
                #Unpaired base following loop continues the loop
                elif shapes[idx - 1] == 'H':
                    shapes[idx] = 'H'
                    idx += 1
            #A base paired downstream following an interior loop portion starts a multiloop
            elif shapes[idx] == '(':
                if shapes[idx - 1] == 'H':
                    in_multi_loop = True
                    num_multi_stems += 1
                    shapes[idx] = 'S'
                    idx += 1
                #If the previous base is part of the stem, or a bulge, continue the stem
                elif shapes[idx - 1] == 'S' or shapes[idx-1] == 'B':
                    num_stems += 1
                    shapes[idx] = 'S'
                    idx += 1
            #An upstream paired base ')' is the other side of some stem, so should reduce # stems
            elif shapes[idx] == ')':
                if num_stems > 1:
                    shapes[idx] = 'S'
                    num_stems -= 1
                else:
                    shapes[idx] = 'S'
                    num_stems -= 1
                    in_loop = False
                idx += 1
        #In a multiloop
        else:
            if shapes[idx] == '.':
                if shapes[idx - 1] == 'S':
                    #Unpaired bases following the stems are forming a loop
                    if shapes[idx + 1] == '.':
                        shapes[idx] = 'M'
                    #If there is only one unpaired base, we classify it as a bulge
                    else:
                        shapes[idx] = 'B'
                    idx += 1
                #Unpaired base following loop continues the loop
                elif shapes[idx - 1] == 'M':
                    shapes[idx] = 'M'
                    idx += 1
            elif shapes[idx] == '(':
                num_multi_stems += 1
                shapes[idx] = 'S'
                idx += 1
            elif shapes[idx] == ')':
                if num_multi_stems > 1:
                    shapes[idx] = 'S'
                    num_multi_stems -= 1
                else:
                    shapes[idx] = 'S'
                    num_multi_stems -= 1
                    in_multi_loop = False
                idx += 1
    return ''.join(shapes)


#Create array of all motifs and count through entire input file. Return most common n motifs in total, or all non-zero motifs if fewer than n exist.
def GenerateAllMotifs(dict, reads, k):
    counts = np.zeros(len(dict))
    for string in reads:
        n = len(string)
        for i in range(n-k+1):
            substr = string[i:i+k]
            if substr in dict:
                counts[dict[substr]] += 1
            else:
                pass
    if np.sum(counts > 0) <= num_partition:
        return np.argwhere(counts != 0).reshape(np.sum(counts > 0))
    else:
        return np.argpartition(counts, -num_partition)[-num_partition:]

#take a read as a sequence of secondary abstract structures, using a sliding window of 8 and step size 1, compute counts of each motif in the string
def FindMotifs(dict, string, k):
    n = len(string)
    counts = np.zeros(len(dict), dtype = bool)
    #Pattern matching
    for i in range(n-k+1):
        substr = string[i:i+k]
        if substr in dict:
            counts[dict[substr]] = True
        else:
            pass

    return counts




#Read input file and strip out unnecessary things
input_reads = []
with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        if i % 2 == 1:
            input_reads.append(line.split(' ')[0])


#Process input data from dot-bracket notation to the abstract structure representation
"""with open("debug.txt", 'w') as fn:
    for i in range(len(input_reads)):
        input_reads[i] = FindShapes(input_reads[i])
        fn.write(input_reads[i])"""

#If we need to construct the most common motifs, read dictionary as a list, then create map from indices to 8-letter structural motifs
if control_flag == 1:
    list_of_motifs = []
    dict_motif = {}
    with open(dict_path, 'r') as f:
        for i, line in enumerate(f):
            text = line.rstrip()
            k = len(text)
            list_of_motifs.append(text)
            dict_motif[text] = i
    list_of_motifs = np.array(list_of_motifs, dtype = str)

    #Generate list of most common motifs and save to common_motifs.txt (to avoid MemoryError down the road)
    most_common_indices = GenerateAllMotifs(dict_motif, input_reads, k)
    #print(most_common_indices)
    most_common_motifs = np.sort(list_of_motifs[most_common_indices])
    np.savetxt("common_motifs.txt", most_common_motifs, fmt = "%s")

    #Now, update the list of motifs and the motif dictionary to only contain the most common
    dict_motif = {}
    list_of_motifs = []
    for i, s in enumerate(most_common_motifs):
        dict_motif[s] = i

#Otherwise, we rely on previously generated common_motifs.txt
else:
    dict_motif = {}
    try:
        list_of_motifs = np.loadtxt("common_motifs.txt", dtype = str)
    except:
        raise OSError("common_motifs.txt not found \n")
    for i, s in enumerate(list_of_motifs):
        dict_motif[s] = i


#Save counts of common motifs for each read in new array and save to outfile.npy
count_vec = []
for read in input_reads:
    count_vec.append(FindMotifs(dict_motif, read, k))

count_vec = np.array(count_vec, dtype = bool)
print("Writing to", outfile + ".npy")
np.save(outfile, count_vec)
