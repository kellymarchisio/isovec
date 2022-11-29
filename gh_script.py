
###############################################################################
#
# This file is from:
# https://github.com/cambridgeltl/iso-study/blob/master/scripts/gh_script.py
# And contains pieces from the other scripts in:
# github.com/cambridgeltl/iso-study/ such as evs_script.py
#
# Some changes by Kelly Marchisio for use with Python3 on the CLSP Grid
#   (June 2021).
# Note: compute_diagram and comput_distance are the same code as in BLISS:
#   https://github.com/joelmoniz/BLISS/blob/master/gh/gh.ipynb
#
###############################################################################

# -*- coding: utf-8 -*-
import numpy as np
import eagerpy as ep
import torch, gudhi
import sys, time, codecs
#import matplotlib.pyplot as plt
from gudhi.wasserstein import wasserstein_distance
from gudhi import bottleneck_distance
from bisect import bisect_left
from hopcroftkarp import HopcroftKarp
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

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

def compute_diagram_wass(x, homo_dim=1):
    """
    This function computes the persistence diagram on the basis of the distance matrix
    and the homology dimension.

    While compute_diagram requires a distance matrix, compute_diagram_wass
    accepts a lists of points only. (Fully connected now -- can add a parameter
    to RipsComplex to only include nearest_neighbors below a certain distance.

    Adapted from the below by Kelly for the 0th dimension:
    https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-PyTorch-optimization.ipynb.
        Their comment says:
          Same as the finite part of st.persistence_intervals_in_dimension(1),
          but differentiable
    Can print the below to verify that diag0 and
    persistence_intervals_in_dimension(0) are (approximately) the same.
        print('diag0 is', diag0)
        print('rips_tree.persistence_intervals_in_dimension(0) would have been:',
                rips_tree.persistence_intervals_in_dimension(0))
    """
    # Making the rips_complex is the slow part.
    rips_tree = gudhi.RipsComplex(points=x).create_simplex_tree(max_dimension=homo_dim)
    rips_tree.persistence()
    i = rips_tree.flag_persistence_generators()
    if len(i[0]) > 0:
        i0 = torch.tensor(i[0])
    else:
        i0 = torch.empty((0, 3), dtype=int)
    diag0 = torch.norm(x[i0[:, (0, 1)]] - x[i0[:, (0, 2)]], dim=-1)
    return diag0

def compute_diagram(x, homo_dim=1):
    """
    This function computes the persistence diagram on the basis of the distance matrix
    and the homology dimension
    """
    rips_tree = gudhi.RipsComplex(x).create_simplex_tree(max_dimension=homo_dim)
    rips_diag = rips_tree.persistence()
    return [rips_tree.persistence_intervals_in_dimension(w) for w in range(homo_dim)]

def compute_distance(x, y, homo_dim = 1, device='cpu'):
    start_time = time.time()
    diag_x = compute_diagram_wass(x, homo_dim=homo_dim)
    diag_y = compute_diagram_wass(y, homo_dim=homo_dim)
    diag_x.to(device)
    diag_y.to(device)
    return diffble_gh_distance(diag_x, diag_y, matching=False, device=device)
    #return min([diffble_gh_distance(x, y, matching=False, device=device) for (x, y) in zip(diag_x, diag_y)])

def diffble_gh_distance(S, T, matching=False, device='cpu'):
    # adapted from https://github.com/scikit-tda/persim/blob/master/persim/bottleneck.py
    #S = torch.unsqueeze(S,0)
    #T = torch.unsqueeze(T,0)
    S_size = np.prod(S.shape)
    M = min(S.shape[0], S_size)
    if S_size > 0:
        S = S[torch.isfinite(S[:,1]), :]
        if S.shape[0] < M:
            warnings.warn(
                "dgm1 has points with non-finite death times;"+
                "ignoring those points"
            )
            M = S.shape[0]
    T_size = np.prod(T.shape)
    N = min(T.shape[0], T_size)
    if T_size > 0:
        T = T[torch.isfinite(T[:,1]), :]
        if T.shape[0] < N:
            warnings.warn(
                "dgm2 has points with non-finite death times;"+
                "ignoring those points"
            )
            N = T.shape[0]
    #does not go here
    if M == 0:
        S = torch.FloatTensor([0,0])
        M = 1
    if N == 0:
        T = torch.FloatTensor([0,0])
        N = 1

    Sb, Sd = S[:,0], S[:,1]
    Tb, Td = T[:,0], T[:,1]
    D1 = torch.abs(Sb[:, None] - Tb[None, :])
    D2 = torch.abs(Sd[:, None] - Td[None, :])
    DUL = torch.maximum(D1, D2)

    D = torch.zeros(M+N, M+N).to(device)
    D[0:M, 0:N] = DUL
    UR = np.inf * torch.ones(M,M).to(device)
    UR[range(len(UR)), range(len(UR))] = 0.5 * (S[:,1] - S[:,0]) * torch.ones(M).to(device)
    #UR.fill_diagonal_(0.5 * (S[:,1] - S[:,0]).item())
    D[0:M, N::] = UR

    UL = np.inf * torch.ones(N, N).to(device)
    #UL.fill_diagonal_(0.5 * (T[:,1] - T[:,0]).item())
    UL[range(len(UL)), range(len(UL))] = 0.5 * (T[:,1] - T[:,0]) * torch.ones(N).to(device)
    D[M::, 0:N] = UL

    #torch.sort -> torch.unique -> torch.flatten
    #torch.unique is not diffble -- skipping..
    vals = torch.flatten(D)
    vals[vals == float("Inf")] = 0
    return torch.max(vals)
    # sorted, _ = torch.sort(torch.flatten(D))

    # ds = sorted[0:-1] # Everything but np.inf
    # print(ds)
    # bdist = ds[-1]
    # return bdist

def compute_diffble_wass_distance(x, y, homo_dim = 1):
    '''
        Computing Wasserstein Distance as in:
        https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-PyTorch-optimization.ipynb.

        There some stuff about staying on the unit disk in the demo that I
        (Kelly) didn't implement b/c I don't know if I have to.
    '''
    start_time = time.time()
    diag_x = compute_diagram_wass(x, homo_dim=homo_dim)
    diag_y = compute_diagram_wass(y, homo_dim=homo_dim)
    ## print('diag_y', diag_y)
    #print("Filteration graph: %.3f" % (time.time() - start_time))
    return wasserstein_distance(diag_x, diag_y, order=1, enable_autodiff=True)


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

    # Step 2. Get the actual distance based on matrices and the
    # persistance diagrams
    print("Gromov-Hausdorff: ", compute_distance(en_matrix, de_matrix))

# The code starts here
if __name__=='__main__':
    main()
