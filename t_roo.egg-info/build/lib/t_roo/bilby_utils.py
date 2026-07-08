import bilby
from bilby.gw.waveform_generator import WaveformGenerator
import numpy as np

from .logging import logger


def bilby_inject_gw_signal(injection_parameters:dict,
                        waveform_approximant: str,
                        duration: float,
                        minimum_frequency: float=20,
                        maximum_frequency: float=2048,
                        reference_frequency: float=20,
                        interferometers: list=["H1", "L1", "V1"],
                        sampling_frequency: float=4096,
                        zeronoise: bool=False,
                        fix_phenomnsbh: bool=False,
                        psd_files: dict={},
                        asd_files: dict={}):

    """
    Function to deal with signal injections like bilby does.
    TODO(COMMENTED FOR NOW BECAUSE WE HAVE MODIFIED LAL): We add a +pi to the phase if the waveform approximant is IMRPhenomNSBH because of the minus sign
    in front of the amplitude in its LALSuite implementation.
    """

    if "NSBH" in waveform_approximant or "NRTidal" in waveform_approximant:
        frequency_domain_source_model = bilby.gw.source.lal_binary_neutron_star
        parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters

    else:
        frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole
        parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters


    waveform_generator_inj = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=frequency_domain_source_model,
        parameter_conversion=parameter_conversion,
        waveform_arguments=dict(
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency)
    )

    ifos = bilby.gw.detector.InterferometerList(interferometers)

    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = maximum_frequency

    if psd_files:
        for ifo_name in interferometers:
            ifo_idx = np.where(interferometers==ifo_name)
            ifos[ifo_idx].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_files[ifo_name])

    if asd_files:
        for ifo_name in interferometers:
            ifo_idx = np.where(interferometers==ifo_name)
            ifos[ifo_idx].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=asd_files[ifo_name])

    if zeronoise:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=injection_parameters["geocent_time"] - duration + 2,
        )

    else:
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=injection_parameters["geocent_time"] - duration + 2,
        )

    if fix_phenomnsbh:
        if waveform_approximant == 'IMRPhenomNSBH' and 'phase' in injection_parameters.keys():
            injection_parameters['phase'] = (injection_parameters['phase'] + np.pi) % (2.*np.pi)

    ifos.inject_signal(waveform_generator=waveform_generator_inj, parameters=injection_parameters)
    return ifos



class BilbyLikelihood_singlesource:

    """                                                                                                                                                                                       
    TODO: CHANGE THIS TO BE SIMILAR TO THE MULTIODEL CLASS, OR ADAPT THE MULTIMODEL CLASS TO WORK ALSO WITH ONE MODEL ONLY
    Class to pass args to likelihood in the case of                                                                                                                                              
    a single source (no rjmcmc) run                                                                                                                                                               
    """

    def __init__(self, bilbylik,pnames):

        self.bilbylik = bilbylik
        self.pnames = pnames

    def bilbyllhoodwrap(self, parameters):

        """                                                                                                                                                                                         
        Simple wrapper to pass eryn parameters                                                                                                                                                     
        to bilby likelihood as a dictionary                                                                                                                                                           
        """
        # create params dict
        pdict = {pname:p for pname,p in zip(self.pnames,parameters) }

        # pass param dict to the likelihood
        self.bilbylik.parameters.update(pdict)
        return self.bilbylik.log_likelihood()
    
class BilbyGWLikelihood:

    """
    Class that handles the likelihood calls from the bilby classes.
    In this way we avoid passing arguments to the bilbyllhoodwrap, 
    making the (parallelized) code much faster. 
    """

    def __init__(self, 
                models: list,
                parameter_names: list[str],
                waveforms: list[str],
                binary_types: list[str] | dict[str,str],
                interferometers: list,
                duration: float,
                reference_chirp_mass: float,
                fixed_params: dict[str,dict] = {},
                reference_frequency: float=20,
                minimum_frequency: float=20,
                maximum_frequency: float=2048,
                reference_frame: str="sky",
                sampling_frequency: float=4096,
                distance_marginalization: bool=True,
                phase_marginalization: bool=True,
                fix_phenomnsbh: bool=False,
                bilby_priors: bilby.gw.prior.PriorDict = None):

        """
        Initialize BilbyGWLikelihood.
        
        We use Multibanding likelihood to reduce the computational cost.
        (Potential future developments also with ROQs, RelativeBinning, and 'regular' likelihood)

        Args:
            models (list[str]): names of branches used in the ensemble sampler.
            parameter_names (list[str]): list of parameter names. The order in this list determines which column in the state array is interpreted as which likelihood parameter!
            waveforms (list[str]): list of waveforms to use for the models.
            binary_types (list[str] | dict[str, str]): which binary type the models are. Binary types can be 'bbh', 'nsbh', 'bns'.
            interferometers (list): Interferometers carrrying the GW data.
            duration (float): Duration of the signal.
            reference chirp mass (float): reference chirp mass for multibanding likelihood.
            fixed_params (dict): A dictionary containing certain parameters that are kept fixed in the likelihood.
            reference_frequency: reference_frequency for waveform generator (default 20.0 Hz).
            reference_frame: determines whether the sky location is sampled over ra and dec (default) or in azimuth and zenith (if list of interferometers is provided).
            sampling_frequency (float): Sampling frequency for the frequency array. Defaults to 4096 Hz.
            distance_marginalization (bool): Whether to apply distance marginalization in the likelihoods. Defaults to True.
            phase_marginalization (bool): Whether to apply phase marginalization in the likelihoods. Defaults to True. 
            bilby_priors: bilby prior dict that is needed when marginalizing the likelihood over certaint parameters (phase, distance). Otherwise it will be ignored.

        """

        self.models = models
        self.parameter_names = parameter_names
        self.fixed_params = fixed_params
        self.waveforms = waveforms
        self.fix_phenomnsbh = fix_phenomnsbh

        if isinstance(binary_types, list):
            if len(binary_types) != len(models):
                raise ValueError(f"Provided binary types are {binary_types} which does not match branch names {models}")
            self.binary_types = dict(zip(models, binary_types))
        
        elif isinstance(binary_types, dict):
            assert models == list(self.binary_types.keys()), f"binary types dict needs to have the branch names as keys."
            self.binary_types = binary_types
        else:
            raise ValueError(f"binary_types must either be provided as list or dict with branch_names keys.")

        invalid = set(self.binary_types.values()) - {"bbh", "nsbh", "bns"}
        if invalid:
            raise ValueError(f"Binary types {invalid} not implemented.")

        logger.info(f"Initializing BilbyGWLikelihood with:")
        logger.info(f"\t models: {self.models}")
        logger.info(f"\t waveforms: {self.waveforms}")
        logger.info(f"\t fixed params: {self.fixed_params}")

        self.bilby_likelihoods = []
        for model, waveform in zip(models, self.waveforms):

            if self.binary_types[model] in ["nsbh", "bns"]:
                frequency_domain_source_model = bilby.gw.source.binary_neutron_star_frequency_sequence
                parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters

            else:
                frequency_domain_source_model = bilby.gw.source.binary_black_hole_frequency_sequence
                parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters


            waveform_generator = bilby.gw.WaveformGenerator(
                duration=duration,
                sampling_frequency=sampling_frequency,
                frequency_domain_source_model=frequency_domain_source_model,
                parameter_conversion=parameter_conversion,
                waveform_arguments=dict(
                waveform_approximant=waveform,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency
                reference_frequency=reference_frequency)
            )

            likelihood = bilby.gw.likelihood.MBGravitationalWaveTransient(
                interferometers=interferometers,
                waveform_generator=waveform_generator,
                reference_chirp_mass = reference_chirp_mass,
                priors=bilby_priors[model],
                reference_frame = reference_frame,
                distance_marginalization=distance_marginalization,
                phase_marginalization=phase_marginalization,
                #distance_marginalization_lookup_table=f"dL_marg_table_{model}.npz"
                )

            self.bilby_likelihoods.append(likelihood)


    def bilbylik_wrap_rj(self, parameters):

        """
        Wrapper to use bilby likelihood.
        Creates a dictionary with the parameters and the parameters names to update the likelihood parameters.
        not_nan_counter makes sure that only one model at a time as not None parameters and is therefore used.

        TODO(COMMENTED FOR NOW BECAUSE WE ARE USING MODIFIED LAL): We add a +pi to the phase if the waveform approximant is IMRPhenomNSBH because of the minus sign
        in front of the amplitude in its LALSuite implementation.
        """

        not_nan_counter = 0

        for model, waveform_approximant, bilby_likelihood, params in zip(self.models, self.waveforms, self.bilby_likelihoods, parameters):

            if params is None:
                continue

            parameter_dict = dict(zip(self.parameter_names, params[0]))
            parameter_dict.update(self.fixed_params[model])

            if self.fix_phenomnsbh:
                if waveform_approximant == 'IMRPhenomNSBH' and 'phase' in parameter_dict.keys():
                    parameter_dict['phase'] = (parameter_dict['phase'] + np.pi) % (2.*np.pi)        


            bilby_likelihood.parameters = parameter_dict
            logl = bilby_likelihood.log_likelihood()
            not_nan_counter += 1

        if not_nan_counter != 1:
            raise ValueError(f"Parameters passed to likelihood had more than non-nan entry (which is unintended): {parameters}")

        return logl


