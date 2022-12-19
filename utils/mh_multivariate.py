import numpy as np
from scipy import stats


def naive_independent_metropolis_hastings_multivariate(x0, N, D, burn_in, target, proposal):
    """ Construct N samples from a multivariate target distribution using the Independent
    Metropolis-Hastings algorithm
    Args:
        x0: Initial state
        N: Number of samples
        D: Dimensions
        nurn_in: Burn-in period
        target: Target function
        proposal: Proposal function
    Returns:
        samples: (N, D) vector of samples from target function 
        MH_accept_prob: Metropolis-Hastings acceptance rate
    """

    samples = np.zeros((N + burn_in, D))
    i = 0
    j = 0

    while i < burn_in:
        # Step 1: Draw from proposal
        x = proposal.rvs(size=1)

        # Step 2: Calculate probability of choosing x
        p = min((target.pdf(x) / target.pdf(x0)) * ((proposal.pdf(x0)) / (proposal.pdf(x))), 1)

        # Step 3: Accept-reject
        u = np.random.uniform(low=0, high=1)
        if u <= p:
            x0 = x

        i += 1
    
    while i < N + burn_in:
        # Step 1: Draw from proposal
        x = proposal.rvs(size=1)

        # Step 2: Calculate probability of choosing x
        p = min((target.pdf(x) / target.pdf(x0)) * ((proposal.pdf(x0)) / (proposal.pdf(x))), 1)

        # Step 3: Accept-reject
        u = np.random.uniform(low=0, high=1)
        if u <= p:
            x0 = x

        samples[i - burn_in - 1, :] = x0

        if (x0 == x).all():
            j += 1
        i += 1

    MH_accept_prob = j / N

    return samples, MH_accept_prob

def naive_rw_metropolis_hastings_multivariate(x0, N, D, burn_in, target, proposal):
    """ Construct N samples from a multivariate target distribution using the Random Walk
    Metropolis-Hastings algorithm
    Args:
        x0: Initial state
        N: Number of samples
        D: Dimensions
        nurn_in: Burn-in period
        target: Target function
        proposal: Proposal function
    Returns:
        samples: (N, D) vector of samples from target function
        MH_accept_prob: Metropolis-Hastings acceptance rate 
    """
    
    samples = np.zeros((N + burn_in, D))
    i = 0
    j = 0

    while i < burn_in:

        # Step 1: Draw from proposal
        proposal.loc = x0
        x = proposal.rvs(size=1)

        # Step 2: Calculate probability of choosing x
        g_x_x0 = proposal.pdf(x)
        proposal.loc = x
        g_x0_x = proposal.pdf(x0)
        p = min((target.pdf(x) / target.pdf(x0)) * (g_x0_x / g_x_x0), 1)

        # Step 3: Accept-reject
        u = np.random.uniform(low=0, high=1)
        if u <= p:
            x0 = x

        i += 1

    while i < N + burn_in:

        # Step 1: Draw from proposal
        proposal.loc = x0
        x = proposal.rvs(size=1)

        # Step 2: Calculate probability of choosing x
        g_x_x0 = proposal.pdf(x)
        proposal.loc = x
        g_x0_x = proposal.pdf(x0)
        p = min((target.pdf(x) / target.pdf(x0)) * (g_x0_x / g_x_x0), 1)

        # Step 2: Accept-reject
        u = np.random.uniform(low=0, high=1)
        if u <= p:
            x0 = x

        samples[i - burn_in - 1, :] = x0

        if (x0 == x).all():
            j += 1
            
        i += 1

    MH_accept_prob = j / N

    return samples, MH_accept_prob
