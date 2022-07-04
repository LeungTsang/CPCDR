import os 
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2_model.modeling.backbone import build_backbone
from detectron2_model.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2_model.config import get_cfg
from model import *

    
def build_SwAV(config):
    cfg = get_cfg()
    cfg.merge_from_file(config.cfg_file)

    backbone = build_backbone(cfg)
    sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
    projector = MLP(in_channels = 128, out_channels=config.projector_out_channels, layer_num = config.projector_layer_num)
    prototypes = nn.Conv2d(in_channels=projector.out_channels, out_channels=config.num_prototypes, kernel_size=1, padding=0, bias = False)

    model = SwAVModel(backbone, sem_seg_head, projector, prototypes)

    return model


def sec_to_hm_str(t):

    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0, silent = True):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    #print("Computing pairwise distances...")
    device=X.device
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros((n, n), device=device)
    beta = torch.ones((n, 1), device=device)
    logU = torch.log(torch.tensor([perplexity],device=device))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0 and not silent:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50, silent = True):
    #print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    if not silent:
        print((n, d))
    X = X - torch.mean(X, 0)

    l, M = torch.eig(torch.mm(X.t(), X),True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0 and i+1<d:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, Y=None, no_dims=2, initial_dims=50, perplexity=30.0, eta = 500, max_iter = 1000, stop_P = 100, silent = True):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    device = X.device
    print(X.shape)
    if X.shape[1]>initial_dims:
        X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    #max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    #eta = 500
    min_gain = 0.01
    if Y is None:
        Y = torch.randn((n, no_dims),device=device)
    dY = torch.zeros((n, no_dims),device=device)
    iY = torch.zeros((n, no_dims),device=device)
    gains = torch.ones((n, no_dims),device=device)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21],device=device))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12],device=device))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y