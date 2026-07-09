### Run on injected data

In the following we show how to create a simulated GW signal and analyze it with t-roo. The full `*.py` file for this run can be found in the [examples](https://github.com/AnnaPuecher/t_roo/blob/main/docs/examples/troo_inj.py).

Define the features of the signal, such as duratio, frequency range, etc. This is the same as what is usually set in bilby runs.

```
#################################################
# Define basic quantities (freqs, duration, seed)
#################################################

### Note: this seed is used for waveform generator,
### since sampling seed is defined by model.random
### drawn later in the ensemble

np.random.seed(13599876)
bilby.core.utils.random.seed(13599876)

duration = 128.
sampling_frequency = 4096.0
minimum_frequency = 5.
reference_frequency = 20.
maximum_frequency = 2048.
trigger_time = 1126259642.413
end_time = trigger_time + 2.
start_time = end_time - duration
```

Define the settings of the `t-roo` run:

- `models`: name to use internally for the different models to use and compare in the analysis. Example `models = ["mod_bbh", "mod_nsbh", "mod_bns"].`
- `binary_types`: to which binary class each model belongs to (only accepted classes are "bbh", "nsbh", or "bns"). Example `binary_types = ["bbh", "nsbh", "bns"]`.
- `waveforms`: name of the approximant to use in each model. Example: `waveforms = ["IMRPhenomD", "IMRPhenomNSBH", "IMRPhenomD_NRTidalv2"]`.
- `inj_waveform`: approximant to use for the injection, i.e., to simulate the GW signal. Example: `inj_waveform = "IMRPhenomD_NRTidalv2"`.
- `ncores`: to run in parallel on more cores. Example: `ncores = 8`. **Note**: t-roo scales well only up to 8 cores.
- `prior_files`: list of prior files, one for each model. Example: `prior_files = ["prior_bbh.prior", "prior_nsbh.prior", "prior_bns.prior"]`
- `nwalkers`: number of walkers in the ensemble sampler. Example: `nwalkers = 500`.
- `ntemps`: number of temperatures to use for parallel tempering. Example: `ntemps = 16`.


Passing the temperature and stopping arguments:

```
tempering_kwargs = {"ntemps" : ntemps}
```

This passes the number of temperature specified for parallel tempering. Here one can also decide which maximum temperature to employ, and whether to use adapting tempering or no, and for how many iteratios.

```
stopping_kwargs = {
'stop_crit': 'steps',
'n_steps': 20000,
'verbose': True,
}

stopping_prel_kwargs ={
'stop_crit': 'steps',
'n_steps': 2000,
'verbose': True,
}
```
Defines the stopping criterion. By default we use a fixed number of steps, that is here specified via the argument `n_steps`. With `verbose:True` the sampler progress is periodically printed in the .out file.   


Next we need to define the moves to use in t-roo, both the within-model and between-model one:
```
move = WithinModelStretchMove(branch_names=models,
                              binary_types=binary_types,
                              live_dangerously=True)

rj_move = GWBinariesRJ(branch_names=models,
                       binary_types=binary_types,
                       steps_jump_params=200,
                       epsilon = 0.2)
```
In the RJ move, `steps_jump_params` decides how many last samples of the preliminary sampling phase to use to compute the likelihood center-of-mass for the mass parameters proposals, while `epsilon` defines the range in which the auxiliary variables for the tidal parameters stretch moves are drawn, i.e., $[1-\epsilon, 1+\epsilon]$.

Next we define the injection parameters to create the mock signal, as we would do in a bilby run
```
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
```

In t-roo, we can then specify if there are parameters that we want to keep fixed. For example, the IMRPhenomNSBH model always wants $\Lambda_1 = 0$, therefore we define
```
fixed_params = {"bbh": {},
                "nsbh": {'lambda_1':0}, "bns": {}}
```

In the `main` function, we read-in the prior from the file and make sure that all the settings are consistent, for example when using distance or phase marginalization in the likelihood.

We create the detector objects and inject the signal with the wrapper function `bilby_inject_gw_signal` as 
```
ifos = bilby_inject_gw_signal(injection_parameters=injection_parameters,
                          waveform_approximant=inj_waveform,
                          duration=duration,
                          minimum_frequency=minimum_frequency,
                          maximum_frequency=maximum_frequency,
                          interferometers=["H1", "L1", "V1"],
                          sampling_frequency=sampling_frequency,
                          fix_phenomnsbh=fix_phenomnsbh,
                   )
```
where `fix_phenomnsbh=True` takes care of the minus sign in the amplitude of IMRPhenomNSBH.

With the function `check_parameter_order` we create the priors containers and the list of names corresponding to each parameters, that will be used later in the likelihood wrapper and in the sampler routines:
```
priors, parameter_names = check_parameter_order(priors)
```


We need to create the initial state of the chains as
```
state = initial_state_from_prior(priors,
	                        nwalkers,
                                ntemps)
```
where the initial values of the parameters are simply drawn from the prior, 
and the dictionary with the dimensionality of each model
```
ndims = {model: len(parameter_names) for model in models}
```

We define the likelihood through a wrapper function that calls the bilby likelihood as
```
bilby_likelihood = BilbyGWLikelihood(models=models,
                                         parameter_names=parameter_names,
                                         waveforms=waveforms,
                                         binary_types=binary_types,
                                         interferometers=ifos,
                                         duration=duration,
                                         sampling_frequency=sampling_frequency,
                                         reference_chirp_mass = 1.15,
                                         reference_frequency = 20.,
                                         minimum_frequency=minimum_frequency,
                                         maximum_frequency=maximum_frequency,
                                         reference_frame = ["H1", "L1", "V1"],
                                         distance_marginalization=distance_marginalization,
                                         phase_marginalization=phase_marginalization,
                                         fix_phenomnsbh=fix_phenomnsbh,
                                         fixed_params=fixed_params,
                                         bilby_priors=bilby_priors)
```
Here specifically we are passing the argument `reference_chirp_mass` because the wrapper employs by default the mutlibanding likelihood `MBGravitationalWaveTransient` in bilby.

Finally, we create the `EnsembleSampler` object and run it
```
ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            bilby_likelihood.bilbylik_wrap_rj,
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
            moves=move,
            rj_moves= rj_move,
            niter_check_proposal=None,
            )

        out = ensemble.run_mcmc(state, progress=True, thin_by=thin_by)
```

To run in parallel on `ncores`
```
with mp.Pool(ncores) as pool:
	ensemble = EnsembleSampler(
		 nwalkers,
		...
		pool=pool)

	out = ensemble.run_mcmc(state, progress=True, thin_by=thin_by)
```
