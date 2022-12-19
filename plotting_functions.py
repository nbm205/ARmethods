import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

"""Several plotting functions for illustrating relevant aspects of the accept-reject methods"""

def envelope_plot(x_plot, M, target, proposal, title=''):
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(x_plot, 
            target.pdf(x_plot), 
            label=r'$f(x)$', color='black')
    ax.plot(x_plot, 
            proposal.pdf(x_plot),
            label='proposal', color='darkcyan')
    ax.plot(x_plot, 
            M*proposal.pdf(x_plot),
            label='envelope', color='maroon')          
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title)
    ax.legend()
    fig.tight_layout()

    return fig

def pdf_kde_plot(samples, x_plot, target, title='', bins=30):
    pdf = target.pdf(x_plot)
    kde = stats.gaussian_kde(samples)
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.hist(samples,
            density=True,
            bins=bins,
            color='lightsteelblue',
            alpha=0.8,
            edgecolor='black')
    ax.plot(x_plot,
            pdf,
            color='black',
            label='pdf')
    ax.plot(x_plot, 
            kde(x_plot),
            color='maroon',
            label='kde')

    # Format plot
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title)
    plt.legend()
    fig.tight_layout()

    return fig

def trace_plot(samples, title=''):
    t_plot = 1 + np.arange(len(samples))
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(t_plot,
            samples,
            color='black')

    # Format plot
    plt.xlabel('t', fontsize=12)
    plt.ylabel('$x_t$', fontsize=12)
    plt.title(title)
    fig.tight_layout()

    return fig

def autocorrelation_plot(samples, lags=10, title=''):
    fig, ax = plt.subplots(figsize=(10, 6)) 
    fig = pd.plotting.autocorrelation_plot(pd.Series(samples))
    fig.set_xlim([0, lags])

    # Format plot
    plt.title(title)

    return fig

def envelope3D_plot(x_grid, y_grid, M, target, proposal, title=''):

    pos = np.empty(x_grid.shape + (2,))
    pos[:, :, 0] = x_grid
    pos[:, :, 1] = y_grid


    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca(projection='3d')
    surf1 = ax.plot_surface(x_grid, y_grid, target.pdf(pos), color='Yellow', alpha=0.8)
    surf2 = ax.plot_surface(x_grid, y_grid, M * proposal.pdf(pos), color='Blue', alpha=0.2)
    ax.view_init(azim=60, elev=16)
    # Insert legends
    handles, labels = ax.get_legend_handles_labels()
    patch1 = mpatches.Patch(color='Green', label='target', alpha=0.8)
    patch2 = mpatches.Patch(color='Blue', label='envelope', alpha=0.2)  
    handles.extend([patch1, patch2])
    ax.legend(handles=handles)
    plt.title(title)
    fig.tight_layout()
    return fig

def contour_plot(samples, x_grid, y_grid, target, title=''):

    pos = np.empty(x_grid.shape + (2,))
    pos[:, :, 0] = x_grid
    pos[:, :, 1] = y_grid

    levels = np.linspace(0, 0.15, 5)
    levels[0] = 0.01

    fig = plt.figure()
    ax = fig.gca()

    # Contours for true pdf
    cset = ax.contour(x_grid, y_grid, target.pdf(pos),
                levels=levels, alpha=0.9, colors='black')
    ax.clabel(cset, inline=False, fontsize=10)

    # Peform the kernel density estimate
    pos2 = np.vstack([x_grid.ravel(), y_grid.ravel()])
    kernel = stats.gaussian_kde(samples.T)
    f = np.reshape(kernel(pos2).T, x_grid.shape)
    cset1 = ax.contour(x_grid, y_grid, f,
                levels=levels, alpha=0.9, colors='maroon')

    # Insert legends
    handles, labels = ax.get_legend_handles_labels()
    line1 = Line2D([0], [0], label='pdf', color='black')
    line2 = Line2D([0], [0], label='kde', color='maroon')
    handles.extend([line1, line2])
    ax.legend(handles=handles)

    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    return fig
