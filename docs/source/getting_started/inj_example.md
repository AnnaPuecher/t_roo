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


