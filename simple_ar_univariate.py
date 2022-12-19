import numpy as np
from scipy import stats
import math


def naive_simple_accept_reject_univariate(N, M, target, proposal):
    """ Construct N samples from a univariate target distribution using the simple accept-reject method.
    Args:
        N: Number of samples
        M: Scaling constant in envelope function
        target: Target function we wish to sample from
        proposal: Proposal function used in envelope function
    Returns:
        samples: (N, 1) vector of samples from target function 
        accept_prob: Acceptance probability 
    """
    
    samples = np.zeros(N)
    i = 0
    j = 0

    while i < N:
        # Step 1: Draw from uniform and proposal
        u, x = np.random.uniform(low=0, high=1), proposal.rvs(1)

        # Step 2: Accept-reject
        if u * (M * proposal.pdf(x)) <= target.pdf(x):
            samples[i] = x
            i += 1
        
        # Count total iterations
        j +=1
    
    accept_prob = N / j

    return samples, accept_prob


def simple_accept_reject_univariate(N, M, target, proposal):
    """ Construct N samples from a univariate target distribution using the simple accept-reject method. Vectorized version.
    Args:
        N: Number of samples
        M: Scaling constant in envelope function
        target: Target function we wish to sample from
        proposal: Proposal function used in envelope function
    Returns:
        samples: (N, 1) vector of samples from target function 
        accept_prob: Acceptance probability 
    """

    # Number of excess samples based on theoretical rejection frequency. Added for vectorization
    no_samples = math.ceil(N * ((1 / (M * target.norm_const()) + 0.05) * 10))

    # Step 1: Draw from uniform and proposal
    u, x = np.random.uniform(low=0, high=1, size=no_samples), proposal.rvs(no_samples)

    # Step 2: Accept-reject
    bool_arr = u * (M * proposal.pdf(x)) <= target.pdf(x)
    samples = x[bool_arr]

    if len(samples) >= N:
        return samples[0:N], len(samples) / len(x)

    i = len(samples)
    j = len(x)
    
    while i < N:
        # Step 1: Draw from uniform and proposal
        u, x = np.random.uniform(low=0, high=1), proposal.rvs(1)

        # Step 2: Accept-reject
        if u * (M * proposal.pdf(x)) <= target.pdf(x):
            samples = np.append(samples, x)
            i += 1
        
        # Count total iterations
        j +=1
    
    accept_prob = N / j

    return samples, accept_prob
