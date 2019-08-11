import os
import sys
import numpy as np


#Read Command Line Args
input_data_name = sys.argv[1]
dictfile_name = sys.argv[2]
outfile_name = sys.argv[3]


#Read k-mer dictionary as a CL input
k = 0
dict = {}
with open(dictfile_name, 'r') as f:
    for i, line in enumerate(f):
        dict[line.rstrip()] = i
        k = len(line.rstrip())

#READ FASTQ File (Code copied from https://www.biostars.org/p/317524/)
def process(lines=None):
    ks = ['name', 'sequence', 'optional', 'quality']
    return {k: v for k, v in zip(ks, lines)}

try:
    fn = input_data_name
except IndexError as ie:
    raise SystemError("Error: Specify file name\n")

if not os.path.exists(fn):
    raise SystemError("Error: File does not exist\n")

#End copied code

n = 4


#take a read as RNA sequence and, using a sliding window of k and step size 1, compute counts of each k-mer in the string
def ParseRead(dict, string, k):
    #First, change all T to U
    string = string.replace('T', 'U')
    not_found_flag = False
    #print(string)
    n = len(string)
    counts = np.zeros(len(dict), dtype = bool)
    for i in range(n-k+1):
        substr = string[i:i+k]
        if substr in dict:
            counts[dict[substr]] = True
        else:
            not_found_flag = True
            pass
            #print("ERROR: string not found:", substr)

    return counts, not_found_flag



reads = []

with open(fn, 'r') as fh:
    lines = []
    numEntries = 0
    count_vec = []
    test_set_counts = []
    raw_reads = []
    test_set_reads = []
    for line in fh:
        curr_counts = np.zeros(len(dict), dtype = bool)

        if numEntries > 200000:
            break

        lines.append(line.rstrip())
        if len(lines) == n:
            record = process(lines)
            #Only use reads where we are sure of every base pair (nothing with N)
            if 'N' not in record['sequence']:
                curr_counts, nf_flag = ParseRead(dict, record['sequence'], k)
                if nf_flag == False:
                    if numEntries < 100000:
                        numEntries += 1
                        count_vec.append(curr_counts)
                        raw_reads.append(record['sequence'])
                    elif numEntries == 100000:
                        numEntries += 1
                        count_vec = np.array(count_vec, dtype = bool)
                        raw_reads = np.array(raw_reads, dtype = str)
                        print("Writing to", outfile_name + ".npy")
                        print("Writing raw reads to", outfile_name + "_reads.seq")
                        np.save(outfile_name, count_vec)
                        np.savetxt(outfile_name + "_reads.seq", raw_reads, fmt = "%s")
                        count_vec, raw_reads = None, None
                    else:
                        numEntries += 1
                        test_set_counts.append(curr_counts)
                        test_set_reads.append(record['sequence'])
            lines = []
    test_set_counts = np.array(test_set_counts, dtype = bool)
    test_set_reads = np.array(test_set_reads, dtype = str)
    print("Writing test set to", outfile_name + "_test.npy")
    print("Writing test set reads to", outfile_name + "_reads_test.seq")
    np.save(outfile_name + "_test", test_set_counts)
    np.savetxt(outfile_name + "_reads_test.seq", test_set_reads, fmt = "%s")
