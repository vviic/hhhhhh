# DNA序列转变为数字特征

import numpy as np

embeddings = np.load(r'embedding_matrix.npy')


def kmer_to_number(kmer):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    number = 0
    for char in kmer:
        number = number * 4 + mapping[char]
    return number + 1


def generate_kmers(dna_sequence, k):
    kmers = []
    n = len(dna_sequence)
    for i in range(n - k + 1):
        kmer = dna_sequence[i:i + k]
        kmer_number = kmer_to_number(kmer)
        kmers.append(kmer_number)
    return kmers


def map_kmers_to_embedding(kmers_numbers, embedding):
    mapped_embeddings = np.array([embedding[number - 1] for number in kmers_numbers])
    return mapped_embeddings


def dna_to_one_hot(dna_sequence):
    base_to_one_hot = {
        'A': np.array([1, 0, 0, 0]),
        'T': np.array([0, 1, 0, 0]),
        'C': np.array([0, 0, 1, 0]),
        'G': np.array([0, 0, 0, 1])
    }
    one_hot_sequence = np.zeros((len(dna_sequence), 4))

    for i, base in enumerate(dna_sequence):
        one_hot_sequence[i] = base_to_one_hot[base]

    return one_hot_sequence


def matrix_encoding(seq, k):
    seq = seq.upper()
    kmers_numbers = generate_kmers(seq, k)
    matrix = map_kmers_to_embedding(kmers_numbers, embeddings)
    return matrix


def one_hot_matrix(seq):
    seq = seq.upper()
    matrix = dna_to_one_hot(seq)
    return matrix

# test_Seq = 'ATGCGATCGTTTTATTAT'

# dna2vec = matrix_encoding(test_Seq, 6)
# one_hot = dna_to_one_hot(test_Seq)


def ncp_encoding(seq):
    mapping = {
        'A': [1.0, 0.0, 0.0],
        'C': [0.0, 0.0, 1.0],
        'G': [0.0, 1.0, 1.0],
        'T': [1.0, 1.0, 0.0]
    }
    default = [0.5, 0.5, 0.5]
    encoded = [mapping.get(base.upper(), default) for base in seq]
    return np.array(encoded)

def dpp_encoding(seq):
    ncp = ncp_encoding(seq) # (L, 3)
    L = len(ncp)
    if L == 0:
        return np.empty((0, 6))

    # For sequences with only one nucleotide, repeat its NCP encoding to form a 6-dim vector
    if L == 1:
        return np.array([np.concatenate((ncp[0], ncp[0]))])

    # For L > 1:
    # Create a shifted version of ncp for concatenation
    # ncp_shifted will hold ncp[1], ncp[2], ..., ncp[L-1], ncp[L-1] ('自身填充' for the last position)
    ncp_shifted = np.empty_like(ncp)
    ncp_shifted[:-1] = ncp[1:]
    ncp_shifted[-1] = ncp[-1] 

    dpp = np.concatenate((ncp, ncp_shifted), axis=1) # Resulting shape (L, 6)
    return dpp
