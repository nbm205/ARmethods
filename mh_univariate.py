import numpy as np
from scipy import stats


def naive_independent_metropolis_hastings_univariate(x0, N, burn_in, target, proposal):
    """ Construct N samples from a univariate target distribution using the Independent MH method.
    Args:
        x0: Initial state
        N: Number of samples
        M: Scaling constant in envelope function
        burn_in: Burn-in period
        target: Target function we wish to sample from
        proposal: Proposal function used in envelope function
    Returns:
        samples: (N, 1) vector of samples from target function 
    """
    
    samples = np.zeros(N)
    i = 0
    j = 0

    while i < burn_in:
        # Step 1: Draw from proposal
        x = proposal.rvs(1)[0]

        # Step 2: Calculate probability of choosing x
        p = min((target.pdf(x) / target.pdf(x0)) * ((proposal.pdf(x0)) / (proposal.pdf(x))), 1)

        # Step 3: Accept-reject
        u = np.random.uniform(low=0, high=1)
        if u <= p:
            x0 = x

        i += 1

    while i < N + burn_in:
        # Step 1: Draw from proposal
        x = proposal.rvs(1)[0]

        # Step 2: Calculate probability of choosing x
        p = min((target.pdf(x) / target.pdf(x0)) * ((proposal.pdf(x0)) / (proposal.pdf(x))), 1)

        # Step 3: Accept-reject
        u = np.random.uniform(low=0, high=1)
        if u <= p:
            x0 = x

        samples[i - burn_in - 1] = x0

        if x0 == x:
            j += 1
        i += 1

    MH_accept_prob = j / N

    return samples, MH_accept_prob

def independent_metropolis_hastings_univariate(x0, N, burn_in, target, proposal):
    """ Construct N samples from a univariate target distribution using the Independent MH method.
    Args:
        x0: Initial state
        N: Number of samples
        M: Scaling constant in envelope
        burn_in: Burn-in period
        target: Target density we wish to sample from
        proposal: Proposal density used in envelope 
    Returns:
        samples: (N, 1) vector of samples from target distribution 
    """
    
    samples = np.zeros(N)

    # Get uniforms for performing discrete choice in step 3
    u = np.random.uniform(low=0, high=1, size=N + burn_in)

    # Step 1: Draw from proposal
    x = proposal.rvs(N + burn_in)

    i = 0
    j = 0

    while i < burn_in:
        # Step 2: Calculate probability of choosing x
        p = min((target.pdf(x[i]) / target.pdf(x0)) * ((proposal.pdf(x0)) / (proposal.pdf(x[i]))), 1)

        # Step 3: Accept-reject
        if u[i] <= p:
            x0 = x[i]

        i += 1

    while i < N + burn_in:

        # Step 2: Calculate probability of choosing x
        p = min((target.pdf(x[i]) / target.pdf(x0)) * ((proposal.pdf(x0)) / (proposal.pdf(x[i]))), 1)

        # Step 2: Accept-reject
        if u[i] <= p:
            x0 = x[i]

        samples[i - burn_in - 1] = x0

        if x0 == x[i]:
            j += 1
            
        i += 1

    MH_accept_prob = j / N

    return samples, MH_accept_prob

def rw_metropolis_hastings_univariate(x0, N, burn_in, target, proposal):
    """ Construct N samples from a univariate target distribution using the Random Walk MH method.
    Args:
        x0: Initial state
        N: Number of samples
        M: Scaling constant in envelope 
        burn_in: Burn-in period
        target: Target density we wish to sample from
        proposal: Proposal density used in envelope 
    Returns:
        samples: (N, 1) vector of samples from target density 
    """
    
    samples = np.zeros(N)

    # Get uniforms for performing discrete choice in step 3
    u = np.random.uniform(low=0, high=1, size=N + burn_in)

    i = 0
    j = 0

    while i < burn_in:

        # Step 1: Draw from proposal
        proposal.loc = x0
        x = proposal.rvs(1)[0]

        # Step 2: Calculate probability of choosing x
        g_x_x0 = proposal.pdf(x)
        proposal.loc = x
        g_x0_x = proposal.pdf(x0)
        p = min((target.pdf(x) / target.pdf(x0)) * (g_x0_x / g_x_x0), 1)

        # Step 3: Accept-reject
        if u[i] <= p:
            x0 = x

        i += 1

    while i < N + burn_in:

        # Step 1: Draw from proposal
        proposal.loc = x0
        x = proposal.rvs(1)[0]

        # Step 2: Calculate probability of choosing x
        g_x_x0 = proposal.pdf(x)
        proposal.loc = x
        g_x0_x = proposal.pdf(x0)
        p = min((target.pdf(x) / target.pdf(x0)) * (g_x0_x / g_x_x0), 1)

        # Step 2: Accept-reject
        if u[i] <= p:
            x0 = x

        samples[i - burn_in - 1] = x0

        if x0 == x:
            j += 1
            
        i += 1

    MH_accept_prob = j / N

    return samples, MH_accept_prob

def naive_rw_metropolis_hastings_univariate(x0, N, burn_in, target, proposal):
    """ Construct N samples from a univariate target distribution using the Random Walk MH.
    Args:
        x0: Initial state
        N: Number of samples
        M: Scaling constant in envelope
        burn_in: Burn-in period
        target: Target density we wish to sample from
        proposal: Proposal density used in envelope
    Returns:
        samples: (N, 1) vector of samples from target distribution 
    """
    
    samples = np.zeros(N)
    i = 0
    j = 0

    while i < burn_in:

        # Step 1: Draw from proposal
        proposal.loc = x0
        x = proposal.rvs(1)[0]

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
        x = proposal.rvs(1)[0]

        # Step 2: Calculate probability of choosing x
        g_x_x0 = proposal.pdf(x)
        proposal.loc = x
        g_x0_x = proposal.pdf(x0)
        p = min((target.pdf(x) / target.pdf(x0)) * (g_x0_x / g_x_x0), 1)

        # Step 2: Accept-reject
        u = np.random.uniform(low=0, high=1)
        if u <= p:
            x0 = x

        samples[i - burn_in - 1] = x0

        if x0 == x:
            j += 1
            
        i += 1

    MH_accept_prob = j / N

    return samples, MH_accept_prob


    

