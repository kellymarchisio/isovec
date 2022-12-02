###############################################################################
#
# This file originates from:
# https://github.com/cambridgeltl/iso-study/blob/master/scripts/gh_script.py
# And contains pieces from the other scripts in:
# github.com/cambridgeltl/iso-study/ such as evs_script.py
#
# Some changes by Kelly Marchisio for use with Python3 on the CLSP Grid
#   (June 2021).
#
###############################################################################

# -*- coding: utf-8 -*-
import numpy as np
import torch
import sys
from sklearn.preprocessing import normalize

FREQ = 5000
HOMO_DIM = 1

def load_word_vectors(file_destination, max=200001):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector
    dimensionality.
    """
    print("Loading vectors from", file_destination, file=sys.stderr)
    input_dic = {}

    with open(file_destination, "r") as in_file:
        lines = in_file.readlines()

    in_file.close()

    words = []
    vectors = []
    for line in lines[1:max]:
        item = line.strip().split()
        dkey = item.pop(0)
        words.append(dkey)
        vector = np.array(item, dtype='float32')
        vectors.append(vector)
        #print np.mean(vector)

    npvectors = np.vstack(vectors)

    # Our words are stored in the list words and...
    # ...our vectors are stored in the 2D array npvectors

    # 1. Length normalize
    npvectors = normalize(npvectors, axis=1, norm='l2')

    # 2. Mean centering dimesionwise
    npvectors = npvectors - npvectors.mean(0)

    # 3. Length normalize again
    npvectors = normalize(npvectors, axis=1, norm='l2')

    print("vectors loaded from", file_destination, file=sys.stderr)
    return words, np.asarray(npvectors)

def distance_matrix(xx_freq, xx_vec):
    """
    This function computes distance matrices from the embedding matrices
    """
    # List of vectors (arrays)
    xx_vectors = []
    for word in xx_freq[:FREQ]:
        xx_vectors.append(xx_vec[word])
    # nd.array(2-dimensional, like xx_vectors in the function above)
    xx_embed_temp = np.vstack(xx_vectors)
    xx_embed = torch.from_numpy(xx_embed_temp)
    xx_dist = torch.sqrt(2 - 2 * torch.clamp(torch.mm(xx_embed, torch.t(xx_embed)), -1., 1.))
    xx_matrix = xx_dist.cpu().numpy()
    return xx_matrix

def diffble_rs_distance(v1_matrix, v2_matrix, device='cpu', unsupervised=False):
    assert len(v1_matrix) == len(v2_matrix)
    assert torch.isclose(torch.norm(v1_matrix[0]), torch.tensor(1.0))
    assert torch.isclose(torch.norm(v2_matrix[0]), torch.tensor(1.0))

    #dot prods
    v1_dotprod = v1_matrix @ v1_matrix.T #[1000,1000]
    v2_dotprod = v2_matrix @ v2_matrix.T

    #remove dot products with themselves (adds false correlation)
    v1_dotprod.fill_diagonal_(0)
    v2_dotprod.fill_diagonal_(0)

    #mean center & flatten
    if unsupervised:
        # Sort the correspondences, b/c we don't know the word alignments.
        # This is our novel contribution.
        v1_cent, _ = torch.sort((v1_dotprod - torch.mean(v1_dotprod)).flatten())
        v2_cent, _ = torch.sort((v2_dotprod - torch.mean(v2_dotprod)).flatten())
    else:
        v1_cent = (v1_dotprod - torch.mean(v1_dotprod)).flatten()
        v2_cent = (v2_dotprod - torch.mean(v2_dotprod)).flatten()

    v1_normed = torch.nn.functional.normalize(v1_cent, dim=0)
    v2_normed = torch.nn.functional.normalize(v2_cent, dim=0)

    cost = torch.ones(1).to(device) - torch.dot(v1_normed, v2_normed)

    return cost

def select_k(spectrum, min_energy = 0.9):
    running_tot = 0
    total = torch.sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_tot += spectrum[i]
        if running_tot/total > min_energy:
            return i+1
    return len(spectrum)

def diffble_evs_distance(v1_matrix, v2_matrix, device):
    assert len(v1_matrix) == len(v2_matrix)
    assert torch.isclose(torch.norm(v1_matrix[0]), torch.tensor(1.0))
    assert torch.isclose(torch.norm(v2_matrix[0]), torch.tensor(1.0))

    v1_dotprod = v1_matrix @ v1_matrix.T #[1000,1000]
    v2_dotprod = v2_matrix @ v2_matrix.T

    #remove dot products with themselves (adds 1 to degree matrix)
    v1_dotprod.fill_diagonal_(0)
    v2_dotprod.fill_diagonal_(0)

    diag_v1 = torch.diag(torch.sum(v1_dotprod, 1)).to(device)
    diag_v2 = torch.diag(torch.sum(v2_dotprod, 1)).to(device)

    laplacian_v1 = diag_v1 - v1_dotprod
    laplacian_v2 = diag_v2 - v2_dotprod

    eigvals_v1 = torch.linalg.eigvalsh(laplacian_v1)
    eigvals_v2 = torch.linalg.eigvalsh(laplacian_v2)

    k1 = select_k(eigvals_v1)
    k2 = select_k(eigvals_v2)
    k = min(k1, k2)

    sq_diff = (eigvals_v1[:k] - eigvals_v2[:k])**2
    #print(sq_diff)
    similarity = sum(sq_diff) / len(sq_diff) #select k after test

    return similarity

def main():
    # Get vectors first and words sorted by frequency
    en_freq, en_vec = load_word_vectors(sys.argv[1])
    de_freq, de_vec = load_word_vectors(sys.argv[2])

    # Step 1. Compute distance matrices from the top FREQ words
    # a) Source and b) Target
    en_matrix = distance_matrix(en_freq, en_vec)
    de_matrix = distance_matrix(de_freq, de_vec)


# The code starts here
if __name__=='__main__':
    main()
