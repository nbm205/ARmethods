import numpy as np
from scipy import stats

def naive_simple_accept_reject_multivariate(N, M, D, target, proposal):
    """ Construct N samples from a multivariate target distribution using the simple accept-reject method.
    Args:
        N: Number of samples
        M: Scaling constant in envelope function
        target: Target function we wish to sample from
        proposal: Proposal function used in envelope function
    Returns:
        samples: (N, 1) vector of samples from target function 
        accept_prob: Acceptance probability 
    """
    
    samples = np.zeros((N, D))
    i = 0
    j = 0

    while i < N:
        # Step 1: Draw from uniform and proposal
        u, x = np.random.uniform(low=0, high=1), proposal.rvs(size=1)

        # Step 2: Accept-reject
        if u * (M * proposal.pdf(x)) <= target.pdf(x):
            samples[i, :] = x
            i += 1
        
        # Count total iterations
        j +=1
    
    accept_prob = N / j

    return samples, accept_prob
