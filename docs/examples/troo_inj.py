import numpy as np
import bilby
import matplotlib.pyplot as plt
from bilby.gw.waveform_generator import WaveformGenerator

import lalsimulation as lalsim

from t_roo.ensemble import EnsembleSampler
from t_roo.moves import GWBinariesRJ, WithinModelStretchMove
from t_roo.backends import HDFBackend
from t_roo.prior import ProbDistContainer, check_parameter_order
from t_roo.prior.prior_gw import Uniform, DeltaFunction, AlignedSpin, UniformComovingVolume, Cosine, Sine, Categorical
from t_roo.proposal_block_sampling import ProposalCycle
from t_roo.state import State, initial_state_from_prior
from t_roo.bilby_utils import BilbyGWLikelihood, bilby_inject_gw_signal

from gwpy.timeseries import TimeSeries
logger = bilby.core.utils.logger

import multiprocessing as mp

#################################################
# Define basic quantities (freqs, duration, seed)
#################################################

### Note: this seed is used for waveform generator,
### since sampling seed is defined by model.random
### drawn later in the ensemble

np.random.seed(13599876)
bilby.core.utils.random.seed(13599876)

duration = 7100.0
sampling_frequency = 4096.0
minimum_frequency = 5.
reference_frequency = 20.
maximum_frequency = 2048.
trigger_time = 1126259642.413
end_time = trigger_time + 2.
start_time = end_time - duration

#########################
# Eryn general settings
#########################

models = ["bbh", "nsbh", "bns"]
binary_types = ["bbh", "nsbh", "bns"]
waveforms = ["IMRPhenomD", "IMRPhenomNSBH", "IMRPhenomD_NRTidalv2"]
inj_waveform = "IMRPhenomD_NRTidalv2"
distance_marginalization=True
phase_marginalization=True
fix_phenomnsbh = False
ncores = 8

prior_files = ["prior_bbh.prior", "prior_nsbh.prior", "prior_bns.prior"]

nwalkers = 500
ntemps = 16
burn = 0
thin_by = 1
tempering_kwargs = {"ntemps" : ntemps}

stopping_kwargs = {
'stop_crit': 'steps',
'n_steps': 20000,
#'n_iters': 10,
#'start_iteration': 200,
'verbose': True,
#'check_every': 5}
}

stopping_prel_kwargs ={
'stop_crit': 'steps',
'n_steps': 2000,
'verbose': True,
#'start_iteration': 
}

###############
# t-roo moves #
###############

move = WithinModelStretchMove(branch_names=models,
                              binary_types=binary_types,
                              live_dangerously=True)

rj_move = GWBinariesRJ(branch_names=models,
                       binary_types=binary_types,
                       steps_jump_params=200,
                       epsilon = 0.2)

########################
# Injection parameters #
########################

m1 = 1.4
m2 = 1.4
mchirp = bilby.gw.conversion.component_masses_to_chirp_mass(
    m1, m2
)
mratio = m2 / m1

injection_parameters = dict(
    chirp_mass=mchirp,
    mass_ratio=mratio,
    geocent_time=trigger_time,
    phase=1.3,
    chi_1=0.01,
    chi_2=0.02,
    luminosity_distance=150.0,
    theta_jn=1.57,
    psi=2.659,
    dec=-0.5,
    ra=3.44,
    lambda_2 = 600,
    lambda_1 = 600.,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
)


fixed_params = {"bbh": {},
                "nsbh": {'lambda_1':0}, "bns": {}}


####################
# Main function #
####################

ef main(distance_marginalization,phase_marginalization,fix_phenomnsbh):

    nleaves_max = {model: 1 for model in models}
    nleaves_min = {model: 0 for model in models}

    ##########
    # Priors #
    ##########

    priors = dict()
    bilby_priors = dict()

    for prior_file, model in zip(prior_files, models):
        priors[model] = ProbDistContainer(filename=prior_file)

        bilby_priors[model] = bilby.gw.prior.CBCPriorDict(filename=prior_file)

        if fix_phenomnsbh and phase_marginalization:

            phase_marginalization = False
            print(f"WARNING: Cannot use phase marginalization if the PhenomNSBH phase was not adjusted in LALSimulation, the new PhenomNSBH phase will be overwritten anyway if phase_marginalization in True. Setting phase_marginalization=False.")

        if phase_marginalization:
            fixed_params[model]["phase"] = priors[model].priors_in["phase"].minimum
            priors[model].remove_parameter("phase")

        if distance_marginalization:
            fixed_params[model]["luminosity_distance"] = priors[model].priors_in["luminosity_distance"].minimum
            priors[model].remove_parameter("luminosity_distance")

    priors, parameter_names = check_parameter_order(priors)

    for model in fixed_params:
        if set(fixed_params[model]) & set(parameter_names):
            print(f"WARNING: Fixed parameters will overwrite prior for parameters {set(fixed_params[model]) & set(parameter_names)} in model {model}.")

    #################
    # initial state #
    #################

    state = initial_state_from_prior(priors,
                                nwalkers,
                                ntemps)

    ndims = {model: len(parameter_names) for model in models}

    #################
    # inject signal #
    #################

    ifos = bilby_inject_gw_signal(injection_parameters=injection_parameters,
                          waveform_approximant=inj_waveform,
                          duration=duration,
                          minimum_frequency=minimum_frequency,
                          maximum_frequency=maximum_frequency,
                          interferometers=["S1", "XM1"],
                          sampling_frequency=sampling_frequency,
                          fix_phenomnsbh=fix_phenomnsbh,
                   )


    ##############
    # likelihood #
    ##############

    bilby_likelihood = BilbyGWLikelihood(models=models,
                                         parameter_names=parameter_names,
                                         waveforms=waveforms,
                                         binary_types=binary_types,
                                         interferometers=ifos,
                                         duration=duration,
                                         sampling_frequency=sampling_frequency,
                                         #bilby_priors=bilby_priors,
                                         reference_chirp_mass = 1.15,
                                         reference_frequency = 20.,
                                         minimum_frequency=minimum_frequency,
                                         maximum_frequency=maximum_frequency,
                                         reference_frame = ["S1", "XM1"],
                                         distance_marginalization=distance_marginalization,
                                         phase_marginalization=phase_marginalization,
                                         fix_phenomnsbh=fix_phenomnsbh,
                                         fixed_params=fixed_params,
                                         bilby_priors=bilby_priors)
                                         
    ##############
    # sample #
    ##############

    with mp.Pool(ncores) as pool:

        ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            bilby_likelihood.bilbylik_wrap_rj, # pass the wrapper as the likelihood
            priors,
            parameter_names = parameter_names,
            tempering_kwargs=tempering_kwargs,
            nbranches=len(models),
            branch_names=models,
            outdir='outdir',
            stopping_kwargs=stopping_kwargs,
            stopping_prel_kwargs = stopping_prel_kwargs,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            #provide_groups=False,
            moves=move,
            rj_moves= rj_move,
            niter_check_proposal=None,
            pool=pool,
            )

        out = ensemble.run_mcmc(state, progress=True, thin_by=thin_by)


if __name__=="__main__":
    main(distance_marginalization,phase_marginalization,fix_phenomnsbh)
                                                                                                                                                                                                                             



