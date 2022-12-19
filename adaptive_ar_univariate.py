import numpy as np
import matplotlib.pyplot as plt
import math
import numba as nb
from numba import jit

@jit(nopython=True)
def exp_u(x, xs, hs, dhdxs):
    """Evaluates the exponential piecewise linear envelope function at a given point.
    Args:
        x: Input point
        xs: Abscissae as an ordered set
        hs: Intersection points
        dhdxs: Derivative of log-density
    Returns:
        exp(u(x))
    """
    z = (hs[1:] - hs[:-1] - xs[1:] * dhdxs[1:] + xs[:-1] * dhdxs[:-1]) / (dhdxs[:-1] - dhdxs[1:])
    i = np.searchsorted(z, x)
    
    return np.exp(dhdxs[i] * (x - xs[i]) + hs[i])
    
@jit(nopython=True)
def exp_l(x, xs, hs):
    """Evaluates the exponential piecewise linear squeezing function at a given point.
    Args:
        x: Input point
        xs: Abscissae as an ordered set
        hs: Intersection points
    Returns:
        exp(l(x))
    """
    # If x lies outside of the domain for l_k(x)
    if np.all(x < xs) or np.all(x > xs):
        return -np.inf
    
    else:
        i = np.searchsorted(xs, x)
        l = ((xs[i] - x) * hs[i-1] + (x - xs[i-1]) * hs[i]) / (xs[i] - xs[i-1])

        return np.exp(l)

def fixed_exp_u(x, xs, hs, dhdxs, z, rows):
    """Evaluates the exponential piecewise linear envelope function at a given point.
    Args:
        x: Input point
        xs: Abscissae as an ordered set
        hs: Intersection points
        dhdxs: Derivative of log-density
    Returns:
        exp(u(x))
    """
    i = np.searchsorted(z, x)
    
    return np.exp(dhdxs[rows, i] * (x - xs[rows, i]) + hs[rows, i])
    

def fixed_exp_l(x, xs, hs, rows):
    """Evaluates the exponential piecewise linear squeezing function at a given point.
    Args:
        x: Input point
        xs: Abscissae as an ordered set
        hs: Intersection points
    Returns:
        exp(l(x))
    """

    i = np.searchsorted(xs[0, :], x)
    l = x.copy()
    l[np.logical_or(0 == i, i > len(xs[0, :]) - 1)] = -np.inf
    bool_arr = ~np.logical_or(0 == i, i > len(xs[0, :]) - 1)
    l[bool_arr] = ((xs[rows[bool_arr], i[bool_arr]] - x[bool_arr]) * hs[rows[bool_arr], i[bool_arr]-1] + (x[bool_arr] - xs[rows[bool_arr], i[bool_arr]-1]) * hs[rows[bool_arr], i[bool_arr]]) / (xs[rows[bool_arr], i[bool_arr]] - xs[rows[bool_arr], i[bool_arr]-1])

    return np.exp(l)

@jit(nopython=True)
def sample_envelope(xs, hs, dhdxs, u, z_limits):
    """Samples from s_k(x)"""
    
    # Compute intersection points
    z = (hs[1:] - hs[:-1] - xs[1:] * dhdxs[1:] + xs[:-1] * dhdxs[:-1]) / (dhdxs[:-1] - dhdxs[1:])
    z.sort()
    
    # Concatenate end points
    limits = np.concatenate((np.array([z_limits[0]]), z, np.array([z_limits[1]])))
    limits = np.stack((limits[:-1], limits[1:]), axis=1)

    # Compute numerator for unnormalized probabilities
    probs_unormalized = np.exp(hs - xs * dhdxs + limits[:, 1] * dhdxs) - np.exp(hs - xs * dhdxs + limits[:, 0] * dhdxs)
    
    # Catch instances where dhdx is zero in denominator
    i_nonzero, i_zero = np.where(dhdxs != 0.), np.where(dhdxs == 0.)
    probs_unormalized[i_nonzero] = probs_unormalized[i_nonzero] / dhdxs[i_nonzero]
    probs_unormalized[i_zero] = (np.exp(hs) * (limits[:, 1] - limits[:, 0]))[i_zero]

    # Compute normalized probabilities
    probs = probs_unormalized / np.sum(probs_unormalized)

    # Pick i'th piecewise exponential according to their probability mass
    cumsum = np.cumsum(probs)
    rdm_unif = np.random.rand(1)
    i = np.searchsorted(cumsum, rdm_unif)[0] 
    
    # Invert i^th piecewise exponential CDF to get a sample from that interval
    if dhdxs[i] == 0.:
        return (u * probs_unormalized[i]) / np.exp(hs[i]) + limits[i, 0]
        
    else:
        x = (np.log(u * probs_unormalized[i] * dhdxs[i] + np.exp(hs[i] - xs[i] * dhdxs[i] + limits[i, 0] * dhdxs[i])) + xs[i] * dhdxs[i] - hs[i])
        x = x / dhdxs[i] 
    
        return x

def adaptive_rejection_sampling(N, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')]):
    """ Construct N samples from a univariate target distribution using adaptive rejction sampling.
    Args:
        N: Number of samples
        log_prob: h(x) and h'(x)
        xs: Initial T_k
        z_limits: Bounds on domain of target density
    Returns:
        samples: (N, 1) vector of samples from target
        accept_prob: Acceptance probability 
    """

    samples = np.zeros(N)
    no_samples = math.ceil(N * 1.05) # Number of samples based on theoretical rejection frequency. Added 0.05 in hopes that while loop can be avoided below

    # Compute h(x) and h'(x)
    hs, dhdxs = log_prob(xs)

    # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
    w = np.random.uniform(low=0, high=1, size=(no_samples, 2))

    i = 0
    j = 0
    while i < N:
        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[j, 0], z_limits)

        eu = exp_u(x, xs, hs, dhdxs)
        el = exp_l(x, xs, hs)

        # Squeezing test
        if w[j, 1] * eu <= el:
            samples[i] = x
            i += 1

        else:
            # Evaluate h and h'
            h, dhdx = log_prob(x)

            # Rejection test
            if w[j, 1] * eu <= np.exp(h):
                samples[i] = x
                i += 1
                
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        j += 1
    
    accept_prob = N / j
        
    return samples, accept_prob

def naive_adaptive_rejection_sampling(N, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')]):
    """ Construct N samples from a univariate target distribution using adaptive rejction sampling.
    Args:
        N: Number of samples
        log_prob: h(x) and h'(x)
        xs: Initial T_k
        z_limits: Bounds on domain of target density
    Returns:
        samples: (N, 1) vector of samples from target
        accept_prob: Acceptance probability 
    """

    samples = np.zeros(N)

    # Compute h(x) and h'(x)
    hs, dhdxs = log_prob(xs)

    i = 0
    j = 0
    while i < N:

        # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
        w = np.random.uniform(low=0, high=1, size=2)

        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[0], z_limits)

        eu = exp_u(x, xs, hs, dhdxs)
        el = exp_l(x, xs, hs)

        # Squeezing test
        if w[1] * eu <= el:
            samples[i] = x
            i += 1

        else:
            # Evaluate h and h'
            h, dhdx = log_prob(x)

            # Rejection test
            if w[1] * eu <= np.exp(h):
                samples[i] = x
                i += 1
                
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        j += 1
    
    accept_prob = N / j
        
    return samples, accept_prob

def naive_fixed_adaptive_rejection_sampling(N, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')]):
    """ Construct N samples from a univariate target distribution using adaptive rejction sampling.
    Args:
        N: Number of samples
        log_prob: h(x) and h'(x)
        xs: Initial T_k
        z_limits: Bounds on domain of target density
    Returns:
        samples: (N, 1) vector of samples from target
        accept_prob: Acceptance probability 
    """

    samples = np.zeros(N)

    # Compute h(x) and h'(x)
    hs, dhdxs = log_prob(xs)

    i = 0
    j = 0
    while i < N:

        # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
        w = np.random.uniform(low=0, high=1, size=2)

        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[0], z_limits)

        eu = exp_u(x, xs, hs, dhdxs)
        el = exp_l(x, xs, hs)

        # Squeezing test
        if w[1] * eu <= el:
            samples[i] = x
            i += 1

        else:
            # Evaluate h and h'
            h, dhdx = log_prob(x)

            # Rejection test
            if w[1] * eu <= np.exp(h):
                samples[i] = x
                i += 1

        j += 1
    
    accept_prob = N / j
        
    return samples, accept_prob

@jit(nopython=True)
def fixed_envelope(xs, hs, dhdxs, z_limits, no_samples):
    """Sample from fixed s_k(x)"""

    # Compute intersection points
    z = (hs[1:] - hs[:-1] - xs[1:] * dhdxs[1:] + xs[:-1] * dhdxs[:-1]) / (dhdxs[:-1] - dhdxs[1:])
    z.sort()

    # Concatenate end points
    limits = np.concatenate((np.array([z_limits[0]]), z, np.array([z_limits[1]])))
    limits = np.stack((limits[:-1], limits[1:]), axis=1)

    probs_unormalized = np.exp(hs - xs * dhdxs + limits[:, 1] * dhdxs) - np.exp(hs - xs * dhdxs + limits[:, 0] * dhdxs)
    probs_unormalized = probs_unormalized.reshape((len(probs_unormalized), 1)) / dhdxs.reshape((len(dhdxs), 1))

    # Compute normalized probabilities
    probs = probs_unormalized / np.sum(probs_unormalized)

    # Tile remaining quantities
    limits = np.transpose(np.repeat(limits[:,0], no_samples).reshape((len(limits), no_samples)))
    probs_unormalized = np.transpose(np.repeat(probs_unormalized, no_samples).reshape((len(probs_unormalized), no_samples)))
    xs = np.transpose(np.repeat(xs, no_samples).reshape((len(xs), no_samples)))
    hs, dhdxs = np.transpose(np.repeat(hs, no_samples).reshape((len(hs), no_samples))), np.transpose(np.repeat(dhdxs, no_samples).reshape((len(dhdxs), no_samples)))

    # Pick i'th piecewise exponential according to their probability mass 
    cumsum = np.cumsum(probs)
    cumsum = np.transpose(np.repeat(cumsum, no_samples).reshape((len(cumsum), no_samples)))
    u = np.random.rand(no_samples)
    m,n = cumsum.shape
    max_num = np.maximum(np.max(cumsum) - np.min(cumsum), np.max(u) - np.min(u)) + 1
    r = max_num*np.arange(cumsum.shape[0]).reshape((m, 1))
    p = np.searchsorted( (cumsum+r).ravel(), (u+r[:, 0]).ravel() ).reshape(m,-1)
    i = (p - n*(np.arange(m).reshape((m, 1))))[:, 0] 

    rows = np.arange(len(xs)) # Keep track of indices

    return probs_unormalized, i, limits, xs, hs, dhdxs, z, rows

def fixed_adaptive_rejection_sampling(N, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')]):
    """ Construct N samples from a univariate target distribution using adaptive rejction sampling. Vectorized version.
    Args:
        N: Number of samples
        log_prob: h(x) and h'(x)
        xs: Initial T_k
        z_limits: Bounds on domain of target density
    Returns:
        samples: (N, 1) vector of samples from target
        accept_prob: Acceptance probability 
    """
    
    # Number of samples based on theoretical rejection frequency. Added 0.05 in hopes that while loop can be avoided below
    no_samples = math.ceil(N * 1.05)

    # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
    w = np.random.uniform(low=0, high=1, size=(no_samples, 2))

    # Compute h(x) and h'(x)
    hs, dhdxs = log_prob(xs)

    # probs_unormalized, i, limits, xs, hs, dhdxs, z, rows = fixed_sample_envelope(xs, hs, dhdxs, z_limits, no_samples)
    probs_unormalized, i, limits, xs, hs, dhdxs, z, rows = fixed_envelope(xs, hs, dhdxs, z_limits, no_samples)

    # Invert i^th piecewise exponential CDF to get a sample from that interval
    x = (np.log(w[:, 0] * probs_unormalized[rows, i] * dhdxs[rows, i] + np.exp(hs[rows, i] - xs[rows, i] * dhdxs[rows, i] + limits[rows, i] * dhdxs[rows, i])) + xs[rows, i] * dhdxs[rows, i] - hs[rows, i]) / dhdxs[rows, i]

    # Evaluate envelope
    eu = fixed_exp_u(x, xs, hs, dhdxs, z, rows)
    
    # Evaluate h(x) and h'(x)
    h, dhdxs = log_prob(x)
    
    # Rejection test
    rej_arr = w[:, 1] * eu <= np.exp(h)
    samples = x[rej_arr]
        
    return samples[0:N], len(samples) / len(x)

def adaptive_rejection_sampling_plotting(N, x_plot, x_lim, y_lim, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')], plot='linear', fixed=False):
    """Plotting function for envelopes"""
    hs, dhdxs = log_prob(xs)
    
    samples = np.zeros(N)
    no_samples = math.ceil(N * 1.05) # Number of samples based on theoretical rejection frequency. Added 0.05 in hopes that while loop can be avoided below

    # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
    w = np.random.uniform(low=0, high=1, size=no_samples)
    u = np.random.uniform(low=0, high=1, size=no_samples)

    abscissaes = [2, 3, 5, 10, 20, 50]
    if fixed == True:
        abscissaes = [len(xs)]
        
    i = 0
    j = 0

    while i < N:

        if len(xs) in abscissaes:
            abscissaes.remove(len(xs))

            # Compute log probabilities, exp(u), and exp(l) over x-grid
            log_probs = [log_prob(x)[0] for x in x_plot]
            euu = [exp_u(x, xs, hs, dhdxs) for x in x_plot]
            if z_limits[0] != float('-inf'):
                euu[0] = 1e-8
            ell = [exp_l(x, xs, hs) for x in x_plot]

            if plot == 'linear':
                # Plot h(x), the log envelope, and log squeezing functions
                fig, ax = plt.subplots(figsize=(10, 6)) 
                ax.scatter(xs, hs, color='dimgray', alpha=0.8, zorder=3)
                ax.plot(x_plot, log_probs, color='black', label='$\ln~f$')
                ax.plot(x_plot, np.log(euu), color='maroon', label='$ln envelope$')

                # Handle negative infinity
                ell = np.log(np.maximum(np.array(ell), np.zeros(np.shape(ell)) + 1e-8))
                ax.plot(x_plot, ell, color='darkcyan', label='$ln squeezing$')

                # Plot formatting
                plt.grid()
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.xlabel('$x$', fontsize=12)
                plt.ylabel('$\log~f(x)$', fontsize=12)
                plt.title(f'With {len(xs)} points after {i} iterations', fontsize=12)
                plt.legend()
                fig.tight_layout()
                plt.show()
            elif plot == 'exp':
                # Plot h(x), the envelope and squeezing functions
                fig, ax = plt.subplots(figsize=(10, 6)) 
                ax.scatter(xs, np.exp(hs), color='dimgray', alpha=0.8, zorder=3)
                ax.plot(x_plot, np.exp(log_probs), color='black', label='$f$')
                ax.plot(x_plot, euu, color='maroon', label='$envelope$')

                # Handle negative infinity
                ell = np.maximum(np.array(ell), np.zeros(np.shape(ell)) + 1e-8)
                ax.plot(x_plot, ell, color='darkcyan', label='$squeezing$')

                # Plot formatting
                plt.grid()
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.xlabel('$x$', fontsize=12)
                plt.ylabel('$f(x)$', fontsize=12)
                plt.title(f'With {len(xs)} points after {i} iterations', fontsize=12)
                plt.legend()
                fig.tight_layout()
                plt.show()

        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, u[i], z_limits)
        eu = exp_u(x, xs, hs, dhdxs)
        el = exp_l(x, xs, hs)

        # Squeezing test
        if w[i] * eu <= el:
            samples[i] = x
            i += 1

        else:
            # Evaluate h and h'
            h, dhdx = log_prob(x)

            # Rejection test
            if w[i] * eu <= np.exp(h):
                samples[i] = x
                i += 1
                
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        j += 1
    
    accept_prob = N / j
        
    return samples, accept_prob

def adaptive_rejection_sampling_no_squeezing(N, log_prob, xs=np.array([-1.,1.])):

    hs, dhdxs = log_prob(xs)
    
    samples = np.zeros(N)
    i = 0
    j = 0
    while i < N:
        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs)
        eu = exp_u(x, xs, hs, dhdxs)

        # Draw uniform (0,1)
        w = np.random.rand()

        # Evaluate h and h'
        h, dhdx = log_prob(x)

        # Rejection test
        if w * eu <= np.exp(h):
            samples[i] = x
            i += 1
            
        # Update
        idx = np.searchsorted(xs, x)
        xs = np.insert(xs, idx, x)
        hs = np.insert(hs, idx, h)
        dhdxs = np.insert(dhdxs, idx, dhdx)

        j += 1
    
    accept_prob = N / j
        
    return samples, accept_prob

def adaptive_rejection_metropolis_sampling(N, x0, burn_in, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')]):
    """ Construct N samples from a univariate target distribution using adaptive rejction sampling.
    Args:
        N: Number of samples
        x0: Initial value
        burn_in: Burn-in period
        log_prob: h(x) and h'(x)
        xs: Initial T_k
        z_limits: Bounds on domain of target density
    Returns:
        samples: (N, 1) vector of samples from target
        accept_prob: Acceptance probability 
        mh_accept_prob: Metropolis-Hastings acceptance rate
    """

    samples = np.zeros(N + burn_in)
    no_samples = math.ceil((N + burn_in) * 1.1) # Number of samples based on theoretical rejection frequency. Added 0.05 in hopes that while loop can be avoided below

    # Compute h(x) and h'(x)
    hs, dhdxs = log_prob(xs)

    # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
    w = np.random.uniform(low=0, high=1, size=(no_samples, 3))

    i = 0
    j = 0
    k = 0
    while k < burn_in:
        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[k, 0], z_limits)
        eu = exp_u(x, xs, hs, dhdxs)

        # Evaluate h and h'
        h, dhdx = log_prob(x)

        # Rejection test
        if w[k, 1] * eu <= np.exp(h):
            h0, _ = log_prob(x0)
            eu0 = exp_u(x0, xs, hs, dhdxs)
            p = min((np.exp(h) / np.exp(h0)) * ((min(np.exp(h0), eu0)) / (min(np.exp(h), eu))), 1)

            # Step 3: MH rejection step
            if w[k, 2] < p:
                x0 = x

        else:
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        # Count total iterations    
        k += 1

    k_burn_in = k
    i = k

    while i < N + burn_in:

        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[k, 0], z_limits)
        eu = exp_u(x, xs, hs, dhdxs)

        # Evaluate h and h'
        h, dhdx = log_prob(x)

        # Rejection test
        if w[k, 1] * eu <= np.exp(h):
            h0, _ = log_prob(x0)
            eu0 = exp_u(x0, xs, hs, dhdxs)
            p = min((np.exp(h) / np.exp(h0)) * ((min(np.exp(h0), eu0)) / (min(np.exp(h), eu))), 1)
            i += 1

            # Step 3: MH rejection step
            if w[k, 2] <= p:
                x0 = x
                j += 1

            samples[i - burn_in - 1] = x0
        else:
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        # Count total iterations    
        k += 1
    
    accept_prob = (i-k_burn_in) / (k-k_burn_in)
    mh_accept_prob = j / (i-k_burn_in)
        
    return samples, accept_prob, mh_accept_prob

def naive_adaptive_rejection_metropolis_sampling(N, x0, burn_in, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')]):
    """ Construct N samples from a univariate target distribution using adaptive rejction sampling.
    Args:
        N: Number of samples
        x0: Initial value
        burn_in: Burn-in period
        log_prob: h(x) and h'(x)
        xs: Initial T_k
        z_limits: Bounds on domain of target density
    Returns:
        samples: (N, 1) vector of samples from target
        accept_prob: Acceptance probability 
        mh_accept_prob: Metropolis-Hastings acceptance rate
    """

    samples = np.zeros(N + burn_in)

    # Compute h(x) and h'(x)
    hs, dhdxs = log_prob(xs)

    i = 0
    j = 0
    k = 0
    while k < burn_in:

        # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
        w = np.random.uniform(low=0, high=1, size=3)

        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[0], z_limits)
        eu = exp_u(x, xs, hs, dhdxs)

        # Evaluate h and h'
        h, dhdx = log_prob(x)

        # Rejection test
        if w[1] * eu <= np.exp(h):
            h0, _ = log_prob(x0)
            eu0 = exp_u(x0, xs, hs, dhdxs)
            p = min((np.exp(h) / np.exp(h0)) * ((min(np.exp(h0), eu0)) / (min(np.exp(h), eu))), 1)

            # Step 3: MH rejection step
            if w[2] < p:
                x0 = x

        else:
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        # Count total iterations    
        k += 1

    k_burn_in = k
    i = k

    while i < N + burn_in:

        # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
        w = np.random.uniform(low=0, high=1, size=3)

        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[0], z_limits)
        eu = exp_u(x, xs, hs, dhdxs)

        # Evaluate h and h'
        h, dhdx = log_prob(x)

        # Rejection test
        if w[1] * eu <= np.exp(h):
            h0, _ = log_prob(x0)
            eu0 = exp_u(x0, xs, hs, dhdxs)
            p = min((np.exp(h) / np.exp(h0)) * ((min(np.exp(h0), eu0)) / (min(np.exp(h), eu))), 1)
            i += 1

            # Step 3: MH rejection step
            if w[2] <= p:
                x0 = x
                j += 1

            samples[i - burn_in - 1] = x0
        else:
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        # Count total iterations    
        k += 1

    accept_prob = (i-k_burn_in) / (k-k_burn_in)
    mh_accept_prob = j / (i-k_burn_in)
        
    return samples, accept_prob, mh_accept_prob

def adaptive_rejection_metropolis_sampling_plotting(N, x0, x_plot, x_lim, y_lim, log_prob, xs=np.array([-1.,1.]), z_limits=[float('-inf'), float('inf')], plot='linear'):
    """Plotting function for Adaptive Rejection Metropolis Sampling"""

    hs, dhdxs = log_prob(xs)
    
    samples = np.zeros(N)
    no_samples = math.ceil(N * 1.1) # Number of samples based on theoretical rejection frequency. Added 0.05 in hopes that while loop can be avoided below

    # Draw two vectors of uniform (0,1) for sample_envelope and rejection test
    w = np.random.uniform(low=0, high=1, size=(no_samples, 3))

    abscissaes = [2, 3, 5, 10, 20, 120]
    i = 0
    j = 0
    k = 0
    while i < N:

        if len(xs) in abscissaes:
            abscissaes.remove(len(xs))

            # Compute log probabilities, exp(u), and exp(l) over x-grid
            log_probs = [log_prob(x)[0] for x in x_plot]
            euu = [exp_u(x, xs, hs, dhdxs) for x in x_plot]

            if plot == 'linear':
                # Plot h(x), the log envelope, and log squeezing functions
                fig, ax = plt.subplots(figsize=(10, 6)) 
                ax.scatter(xs, hs, color='dimgray', alpha=0.8, zorder=3)
                ax.plot(x_plot, log_probs, color='black', label='$\ln~f$')
                ax.plot(x_plot, np.log(euu), color='maroon', label='$ln envelope$')

                # Plot formatting
                plt.grid()
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.xlabel('$x$', fontsize=12)
                plt.ylabel('$\log~f(x)$', fontsize=12)
                plt.title(f'With {len(xs)} points after {k} iterations', fontsize=12)
                plt.legend()
                fig.tight_layout()
                plt.show()

            elif plot == 'exp':
                # Plot h(x), the envelope and squeezing functions
                fig, ax = plt.subplots(figsize=(10, 6)) 
                ax.scatter(xs, np.exp(hs), color='dimgray', alpha=0.8, zorder=3)
                ax.plot(x_plot, np.exp(log_probs), color='black', label='$f$')
                ax.plot(x_plot, euu, color='maroon', label='$envelope$')

                # Plot formatting
                plt.grid()
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.xlabel('$x$', fontsize=12)
                plt.ylabel('$f(x)$', fontsize=12)
                plt.title(f'With {len(xs)} points after {k} iterations', fontsize=12)
                plt.legend()
                fig.tight_layout()
                plt.show()

        # Sample envelope and evaluate
        x = sample_envelope(xs, hs, dhdxs, w[k, 0], z_limits)
        eu = exp_u(x, xs, hs, dhdxs)

        # Evaluate h and h'
        h, dhdx = log_prob(x)

        # Rejection test
        if w[k, 1] * eu <= np.exp(h):
            h0, _ = log_prob(x0)
            eu0 = exp_u(x0, xs, hs, dhdxs)
            p = min((np.exp(h) / np.exp(h0)) * ((min(np.exp(h0), eu0)) / (min(np.exp(h), eu))), 1)
            i += 1

            # Step 3: MH rejection step
            if w[k, 2] <= p:
                x0 = x
                j += 1

            samples[i - 1] = x0
        else:
            # Update
            idx = np.searchsorted(xs, x)
            xs = np.insert(xs, idx, x)
            hs = np.insert(hs, idx, h)
            dhdxs = np.insert(dhdxs, idx, dhdx)

        # Total iterations
        k += 1
        if k > N * 2:
            print("terminated")
            accept_prob = (i-k) / (k)
            mh_accept_prob = j / (i-k)
            return samples, accept_prob, mh_accept_prob
    
    accept_prob = i / k
    mh_accept_prob = j / i
        
    return samples, accept_prob, mh_accept_prob
