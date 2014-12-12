#!python3
'''
Ian Leaman
12/12/2014

dependencies:
    - numpy.

'''

import numpy as np


P = 0.85  # Random surfer weight
K = 40  # Maximum itterations
E = 1e-5  # Tolerance
X = .1  # Experimental value


# An abstraction for the word frequency tracker dictionary
class word_freq_table():
    def __init__(self):
        self.word_frequencies = {}
        self.filePath = None

    def insert_word(self, word):
        if word in self.word_frequencies:
            self.word_frequencies[word] += 1
        else:
            self.word_frequencies[word] = 1

    # Shortcut for generating frequency tables for many files
    def generate_from_file(self, filePath):
        self.filePath = filePath
        # Read in file and compute term frequency
        with open(filePath, 'r', encoding='ascii', errors='ignore') as f:

                for line in f:
                    for word in line.split():
                        # Filter stopwords
                        self.insert_word(word)

    def terms(self):
        return self.word_frequencies.keys()

    def __str__(self):
        return self.filePath + ": " + str(self.word_frequencies)

    def __contains__(self, search):
        return search in self.word_frequencies

    def __getitem__(self, key):
        # return self.word_frequencies[key]  # add default 0 here instead?
        return self.word_frequencies.get(key, 0)


# Uses random surfer and the sum of the matrix to generate probabilites and
# ensure probability matrix is irriducible
def standard_probability(column):
    # print(column)
    sumed = np.matrix.sum(column)
    # return sumed
    return P * (column / sumed) + (1-P) * (1 / len(column))


# An experiment to test alternatives to random surfer while maintining
# stochastic probability matrix properties
def weighted_standard_probability(column):
    # print(column)
    sumed = np.matrix.sum(column)
    # return sumed
    probability = P * (column / sumed)
    + (1-P) * ((column + X) / np.matrix.sum(column + X))
    # print(probability)
    return(probability)


def get_all_keys(tables):
    terms = set()
    for table in tables:
        # Apparently this is faster
        terms |= set(table.terms())

    return list(terms)


def generate_frequency_matrix(terms, documents, default=0):
    matrix_array = []

    # O(n^2)
    # Each column is a document and each row is a term
    for term in terms:
        doc_terms_array = []
        for doc in documents:
            doc_terms_array.append(doc[term])
        matrix_array.append(doc_terms_array)

    return np.matrix(matrix_array)


# Accepts an adjacency matrix and a function which operates per column of the
# matrix to transform it into a probability matrix.
# Function must return a matrix with the same dims as the input
def generate_probability_matrix(matrix, func):
    # Create a new probability Matrix
    probability_matrix = np.matrix.copy(matrix).astype(float)

    if matrix.shape[0] == matrix.shape[1]:
        pass
    elif matrix.shape[0] > matrix.shape[1]:
        # append to right
        zeros = np.zeros((matrix.shape[0], matrix.shape[0] - matrix.shape[1]))
        zeros[:, :] = 1 / matrix.shape[0]
        probability_matrix = np.concatenate([probability_matrix, zeros],
                                            axis=1)
    elif matrix.shape[1] > matrix.shape[0]:
        # append to bottom
        zeros = np.zeros((matrix.shape[1] - matrix.shape[0], matrix.shape[1]))
        zeros[:, :] = 1 / matrix.shape[1]
        probability_matrix = np.concatenate([probability_matrix, zeros])

    # Iterate over columns, or n
    for col in range(0, probability_matrix.shape[1]):
        probability_matrix[:, col] = func(probability_matrix[:, col])

    return probability_matrix


def generate_stationary_probability_vector(probability_matrix):
    p_vector = np.zeros((probability_matrix.shape[0], 1))
    p_vector[0, 0] = 1
    p_vector_old = -1
    i = 0
    while i < K or np.all(abs(p_vector_old - p_vector) <= E):
        p_vector = probability_matrix * p_vector
        # print(type(p_vector))
        i += 1
    print("Took", i, "iterations")
    return p_vector


def main():
    fileNames = ["friedman_articles/" + str(x) for x in range(1, 11)]
    frequency_tables = [word_freq_table()] * len(fileNames)

    for table, doc in zip(frequency_tables, fileNames):
        table.generate_from_file(doc)

    # Create term, document adjacency matrix.
    terms = get_all_keys(frequency_tables)

    matrix = generate_frequency_matrix(terms=terms, documents=frequency_tables)
    matrix = generate_probability_matrix(matrix, standard_probability)

    ranks = generate_stationary_probability_vector(matrix)

    # Sort terms and matricies in reverse minimum order.
    sortedLists = [list(x) for x in zip(*sorted(zip(ranks, terms),
                                        key=lambda pair: -pair[0]))]
    # stopWords = load_stopwords()
    # print(stopWords)
    i = 0
    for rank, term in zip(sortedLists[0], sortedLists[1]):
        # if term in stopWords:
        #     continue
        print(str(i + 1) + ": \"" + str(term) + "\" with rank:", rank[0, 0])
        i += 1
        if i == 100:
            break


# def load_stopwords():
#     l = []
#     with open("simple_stopwords", "r") as f:
#         for entry in f:
#             l.append(entry.strip("\n"))
#     return l


main()
