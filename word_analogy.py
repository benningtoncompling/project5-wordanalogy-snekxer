"""
Write a script called word_analogy.py, that solves analogies such as "dog is to cat as
puppy is to ___". The script should run with the following command:
./word_analogy.py <vector_file> <input_directory> <output_directory> <eval_file>
<should_normalize> <similarity_type>

The vector file is a text file with the format:
word v1 v2 v3 v4 ... v300
v1-300 are the entries of the 300 dimensional word embedding vector

Step 1 will be to read in this vector file into a dictionary where the keys are the words
and the values are lists with 300 float entries each.

The input directory will be a path to a directory with several test files. Each test file has
an analogy problem on each line. The format is:
A B C D
A-D are words where A is related to B in the same way that C is related to D.
Some examples include:
Athens Greece Beijing China
amazing amazingly apparent apparently
falling fell knowing knew
"""

import sys
import os
import numpy

vector_file = sys.argv[1]
input_directory = sys.argv[2]
output_directory = sys.argv[3]
eval_file = output_directory + "/" + sys.argv[4]
should_normalize = sys.argv[5]
similarity_type = sys.argv[6]

vectors_in_file = open(vector_file, "r", encoding='UTF-8').read().splitlines()
vectors_in_file = [vector.split(" ") for vector in vectors_in_file]

vector_dict = {}
for vector in vectors_in_file:
    vector_dict[vector[0]] = numpy.array(vector[1:]).astype(float)

"""
Step 2 is to read in these analogy problems, ignoring D. Using only A, B, and C, find
the best candidate for D. This will be the word with the most similar vector to C_vec +
B_vec - A_vec, according to a given similarity metric.

Whether or not the vectors should be normalized before the analogy calculation (C + B -
A) is given by should_normalize:
0 - if <should_normalize> is 0, don't normalize, us the vectors as is
1 - if <should_normalize> is 1, normalize before the C + B - A calculation. This
means if the vector is v1, v2, v3... v300, you should use v1/mag, v2/mag, v3/mag...
v300/mag. Where mag is the magnitude of the vector, square root of (v1^2 + v2^2 +
v3^2 + ... + v300^2).

"""


def normalize():
    normalized_vector_dict = {}
    for word, vectors in vector_dict.items():
        mag = numpy.sqrt(sum(numpy.power(vectors, 2)))
        normalized = []
        for vector in vectors:
            normalized.append(vector / mag)
        normalized_vector_dict[word] = numpy.array(normalized).astype(float)
    return normalized_vector_dict


"""
The similarity metrics are indicated by <similarity_type>:
    #compare C+B-A to vector values of all entries
0 - if similarity_type is 0, use Euclidean distance (the smaller, the more similar) 
1 - if similarity_type is 1, use Manhattan distance (the smaller, the more similar)
2 - if similarity_type is 2, use cosine distance (the larger, the more similar)
"""


# distance calculations generate a new vector that has the distances between C+B-A and each word/vector in model.
# according to the type of distance, the larger or smaller one of all distances is picked, and the word associated
# with that distance is the one that goes into the analogy
# this assumes that the input vector is a numpy array as float

def euclidean(words_vector):
    # euclidean distance = sqrt( sum( ( C+B-A[i] - vector[i])^2 ) )
    distance = {}
    for key, values in vector_dict.items():
        distance[key] = numpy.sqrt(sum(numpy.power(words_vector - values, 2)))
    word = min(distance.items(), key=lambda k: k[1])
    return word[0]


def manhattan(words_vector):
    # manhattan distance = sum( abs( ( C+B-A[i] - vector[i] ) ) )
    distance = {}
    for key, values in vector_dict.items():
        distance[key] = sum(numpy.absolute(words_vector - values))
    word = min(distance.items(), key=lambda k: k[1])
    return word[0]


def cosine(words_vector):
    #cosine similarity = sum( C+B-A[i] * vector[i] ) / sqrt(sum(C+B-A[i]^2)) * sqrt(sum(vector[i]^2))
    distance = {}
    for key, values in vector_dict.items():
        distance[key] = sum(words_vector * values) / (numpy.sqrt(sum(numpy.power(words_vector, 2))) * numpy.sqrt(sum(numpy.power(values, 2))))
    word = max(distance.items(), key=lambda k: k[1])
    return word[0]

"""
You should generate one output file in output_directory for every input file in
input_directory. The input_file and output_file should have the same name, but be in
different directories. Each output file should be of the format:
A B C D'
A-C are the same as the input file A-C, D' is the word your code generated to solve the
analogy. This means that each file in the input_directory should have a corresponding
file in the output_directory, with the same number of lines, and the same first three
words on each line. Only the last word of each line may be different.

Step 3 is to run your analogy solver on all the input files, save the solved analogies to
output files, and calculate the accuracy for each file.

In addition to the output files in the output directory, you should write the accuracy for
each file to a separate output file (eval_file). The format of this file should be as shown
in sample_eval.txt
"""

files = {}
for filename in os.listdir(input_directory):
    if filename.startswith('.'):
        continue
    if not filename.endswith('.txt'):
        continue
    filepath = os.path.join(input_directory, filename)
    file = numpy.array(open(filepath, 'r', encoding='UTF-8').readlines())
    file = [words.split(" ") for words in file]
    files[filename] = file

if should_normalize == 1:
    vector_dict = normalize()

output_dict = {}

with open(eval_file, "w", encoding='UTF-8') as output:
    for filename, file in files.items():
        file_analogies = []
        matched_analogies = 0
        for file_words in file:
            # C+B-A
            if file_words[0] not in vector_dict.keys() or file_words[1] not in vector_dict.keys()or file_words[2] not in vector_dict.keys():
                continue
            else:
                words_vector = vector_dict[file_words[2]]  + vector_dict[file_words[1]] - vector_dict[file_words[0]]
            if similarity_type == 0:
                word = euclidean(words_vector)
            elif similarity_type == 1:
                word = manhattan(words_vector)
            elif similarity_type == 2:
                word = cosine(words_vector)
            print("predicted: " + word + " actual: " + file_words[3].strip("\n"))
            file_analogies.append([file_words[0], file_words[1], file_words[2], word])
            if word == file_words[3].strip("\n"):
                print("match")
                matched_analogies += 1
        output_dict[filename] = file_analogies
        total_analogies = len(file_analogies)
        accuracy = (matched_analogies * 100)/total_analogies
        output.write(filename + "\n" + "ACCURACY TOP1: " + str(accuracy) + "% (" + str(matched_analogies) + "/" + str(total_analogies) + ")\n")
output.close()


for filename, file in output_dict.items():
    filepath = os.path.join(output_directory, filename)
    with open(filepath, "w", encoding='UTF-8') as output:
        for lines in file:
            output.write(lines[0] + " " + lines[1] + " " + lines[2] + " " + lines[3] + "\n")
    output.close()
