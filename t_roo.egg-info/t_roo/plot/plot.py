import h5py
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
pltparams = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16,
        "figure.constrained_layout.use": False}
plt.rcParams.update(pltparams)
import corner

from ..backends import HDFBackend

default_corner_kwargs = dict(bins=40, 
                        smooth=True, 
                        label_kwargs=dict(fontsize=14),
                        title_kwargs=dict(fontsize=14), 
                        quantiles=[],
                        levels=[0.68, 0.95],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=3, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red",
                        labelpad=0.2)

latex_labels = dict(chirp_mass="$\\mathcal{M}$ [$M_\\odot$]",
                    mass_ratio="$q$",
                    chi_1="$\\chi_1$",
                    chi_2="$\\chi_2$",
                    a_1="$a_1$",
                    a_2="$a_2$",
                    chi_eff="$\\chi_eff$",
                    lambda_1="$\\Lambda_1$",
                    lambda_2="$\\Lambda_2$",
                    theta_jn="$\\theta_{\\mathrm{JN}}$ [rad]",
                    dec="$\\delta$ [rad]",
                    ra="$\\alpha$ [rad]",
                    luminosity_distance="$d_L$ [Mpc]",
                    redshift="$z$",
                    psi="$\\Psi$ [rad]",
                    phase="$\\varphi_c$ [rad]",
                    geocent_time="$t_c$ [TCG]",
                    log_likelihood="$\\ln \\mathcal{L}$",
                    )

def corner_plot(posterior: dict | pd.DataFrame,
                parameter_names: list[str],
                truths: dict = {},
                color:str = "blue",
                legend_label:str = None,
                fig: matplotlib.figure.Figure = None,
                **kwargs):
    """
    Make a nice corner plot from the posterior with automated parameter labels.

    Args:
       posterior (dict | pd.DataFrame): posterior samples for which to do the corner plot.
       parameter_names (list[str]): parameters from posterior that should be included in the corner plot.
       truths (dict[str, float]): True (injected values) for some of the parameters. Defaults to {}.
       color (str): color for the corner plot contours. Defaults to blue.
       legend_label (str): Label for the legend. If not set, no legend will be shown. Defaults to None.
       fig (matplotlib.figure.Figure): Figure over which to do the corner plot. If set, ax must also be provided. Defaults to None.

    Returns:
        fig (matplotlib.figure.Figure): Figure with the corner plot.
        ax (matplotlib.axes.Axes): array of axes
    """
    
  
    posterior = pd.DataFrame(posterior)
    
    labels= []
    truths_list = []
    for p in parameter_names:
        labels.append(latex_labels.get(p, p))
        truths_list.append(truths.get(p, None))
   
    if fig is None:
        n = len(parameter_names)
        fig, ax = plt.subplots(n, n, figsize = (n*1.5, n*1.5))
    
    else:
        ax = fig.axes

    corner_args = default_corner_kwargs.copy()
    corner_args.update(kwargs)

    corner.corner(posterior[parameter_names], 
                  fig=fig,
                  color=color,
                  labels=labels,
                  truths=truths_list,
                  **corner_args,
                  hist_kwargs=dict(density=True, color=color))
    
    if legend_label is not None:
        
        if len(parameter_names) < 4:
            lx, ly = 0, -1
        else:
            lx, ly = 1, 4

        handle = plt.plot([],[], color=color)[0]
        ax[lx, ly].legend(handles=[handle], labels=[legend_label], fontsize=15, fancybox=False, framealpha=1)
    
    #fig.tight_layout()
    return fig, ax


def trace_plot(posterior: dict,
               parameter_names: list[str],
               truths: dict = {},
               color:str = "brown",
               fig: matplotlib.figure.Figure = None):
    
    n = len(parameter_names)

    if fig is None:
        fig, ax = plt.subplots(n, figsize=(6, 3*n), sharex=True)
    
    else:
        ax = fig.axes
        if len(ax) != len(parameter_names):
            raise ValueError(f"Provided Figure does not have enough axes for parameter names {parameter_names}.")
        
    if n==1:
        ax = [ax]
    
    for cax, p in zip(ax, parameter_names):
        samples = posterior[p]
        xlim = (0, samples.shape[0])


        cax.plot(samples, color=color)
        if p in truths:
            cax.hlines([truths[p]], *xlim, color="red")

        cax.set_ylabel(latex_labels.get(p, p))
        cax.set_xlim(xlim)
    
    ax[-1].set_xlabel("iteration")

    return fig, ax

def likelihood_plot(logl: np.ndarray,
                    inds: np.ndarray,
                    fig: matplotlib.figure.Figure = None):
    

    if fig is None:
        fig, ax = plt.subplots(1, figsize=(6, 3), sharex=True)
    
    else:
        ax = fig.axes
        if len(ax) != 1:
            raise ValueError(f"Provided Figure needs to have one axis for likelihood plot.")
    
    ax.plot(logl)
    ax.set_xlim((0, logl.shape[0]))

    ax.set_xlabel("iteration")
    ax.set_ylabel(latex_labels["log_likelihood"])

    return fig, ax