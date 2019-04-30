import os
import sys
import numpy as np



#Read k-mer dictionary as a CL input
k = 0
dict = {}
with open(sys.argv[2], 'r') as f:
    for i, line in enumerate(f):
        dict[line.rstrip()] = i
        k = len(line.rstrip())





#READ FASTQ File (Code copied from https://www.biostars.org/p/317524/)
def process(lines=None):
    ks = ['name', 'sequence', 'optional', 'quality']
    return {k: v for k, v in zip(ks, lines)}

try:
    fn = sys.argv[1]
except IndexError as ie:
    raise SystemError("Error: Specify file name\n")

if not os.path.exists(fn):
    raise SystemError("Error: File does not exist\n")

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
    raw_reads = []
    for line in fh:
        curr_counts = np.zeros(len(dict), dtype = bool)

        if numEntries > 150000:
            break

        lines.append(line.rstrip())
        if len(lines) == n:
            record = process(lines)
            if 'N' not in record['sequence']:
                curr_counts, nf_flag = ParseRead(dict, record['sequence'], k)
                if nf_flag == False:
                    numEntries += 1
                    count_vec.append(curr_counts)
                    raw_reads.append(record['sequence'])

            lines = []
    count_vec = np.array(count_vec, dtype = bool)
    raw_reads = np.array(raw_reads, dtype = str)






outfile_name = sys.argv[3]
print("Writing to", outfile_name + ".npy")
print("Writing raw reads to", outfile_name + "_reads.seq")
np.save(outfile_name, count_vec)
np.savetxt(outfile_name + "_reads.txt", raw_reads, fmt = "%s")
