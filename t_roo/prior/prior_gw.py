import numpy as np
from scipy import stats
from copy import deepcopy

from scipy.special import erfinv
from scipy.special._ufuncs import xlogy
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from ..cosmology import get_cosmology, z_at_value

from .prior_base import Prior
from ..logging import logger

"""
Prior classes specifically for GW data analysis, mostly inherited from bilby.
ProbDistContainer from eryn, to create and handle prior dictionaries; 
rvs methods to draw from priors
"""

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    pass

class DeltaFunction(Prior):

    def __init__(self, peak, name=None, latex_label=None, unit=None):

        """Dirac delta function, always returns peak.
        Parameters: 
        peak(float): value of the delta function
        name (str), latex_label (str), unit (str): see Prior class"""

        super(DeltaFunction, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                            minimum=peak, maximum=peak, check_range_nonzero=False)

        self.peak = peak
        self._is_fixed = True
        self.least_recently_sampled = peak

    def rescale(self, val):

        """Recsale prob to peak"""

        return self.peak * val ** 0

    def prob(self,val):

        """Return prior probability of val"""
        
        at_peak = (val == self.peak)
        return np.nan_to_num(np.multiply(at_peak,np.inf))


    def logpdf(self,val):

        """
        Log prob, we define it the same way as prob
        since otherwise it gets nan and eryn complains
        """

        at_peak = (val == self.peak)
        return np.nan_to_num(np.multiply(at_peak,np.inf))
    
class Uniform(Prior):

    """Replace also the uniform and log uniform distributons
       to have everything written in the same way"""

    def __init__(self, minimum, maximum, name=None,latex_label=None, unit=None):

        """Uniform prior withinin [minimum,maximum]"""

        super(Uniform,self).__init__(name=name, latex_label=latex_label,
                                     minimum=minimum, maximum=maximum, unit=unit)

    def rescale(self,val):

        """Rescale random samples in [0,1] to uniform values"""

        return self.minimum + val * (self.maximum - self.minimum)

    def prob(self,val):

        """Return probability of val for uniform distribution"""

        return ((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum)

    def logpdf(self,val):

        """Return the logarithm of probability for val"""

        return xlogy(1, (val >= self.minimum) & (val <= self.maximum)) - xlogy(1, self.maximum - self.minimum)

        
class PowerLaw(Prior):

    """For now we only need this for the log_uniform prior
    Parameters: 
    alpha (float): power law exponent parameter"""

    def __init__(self, alpha, minimum, maximum, name = None, latex_label = None, unit = None):

        super(PowerLaw,self).__init__(name=name, latex_label=latex_label, minimum=minimum, maximum=maximum, unit=unit)

        self.alpha = alpha

    def rescale(self,val):

        if self.alpha == -1:
            return self.minimum * np.exp(val * np.log(self.maximum / self.minimum))
        else:
            return (self.minimum ** (1 + self.alpha) + val *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def prob(self,val):

        if self.alpha == 1:
            return np.nan_to_num(1 / val / np.log(self.maximum / self.minimum)) * self.is_in_prior_range(val)
        else:
            return np.nan_to_num(val ** self.alpha * (1 + self.alpha) /
                                 (self.maximum ** (1 + self.alpha) -
                                  self.minimum ** (1 + self.alpha))) * self.is_in_prior_range(val)

    def logpdf(self,val):

        if self.alpha == 1.:
            normalising = 1. / np.log(self.maximum / self.minimum)
        else:
            normalising = (1 + self.alpha) / (self.maximum ** (1 + self.alpha) -
                                              self.minimum ** (1 + self.alpha))

        with np.errstate(divide='ignore', invalid='ignore'):
            ln_in_range = np.log(1. * self.is_in_prior_range(val))
            ln_p = self.alpha * np.nan_to_num(np.log(val)) + np.log(normalising)

        return ln_p + ln_in_range


class LogUniform(PowerLaw):

    def __init__(self, minimum, maximum, name = None, latex_label = None, unit = None):

        super(LogUniform, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                         minimum=minimum, maximum=maximum, alpha=-1)

        if self.minimum <= 0:
            raise Warning('You specified a uniform-in-log prior with minimum={}'.format(self.minimum))


class Cosine(Prior):

    """Cosine prior with bounds"""

    def __init__(self, minimum=-np.pi/2, maximum=np.pi/2, name = None,
                 latex_label = None, unit = None):

        super(Cosine,self).__init__(minimum=minimum, maximum=maximum, name=name,
                                    latex_label=latex_label, unit=unit)

    def rescale(self,val):

        norm = 1 / (np.sin(self.maximum) - np.sin(self.minimum))
        return np.arcsin(val / norm + np.sin(self.minimum))

    def prob(self,val):

        return np.cos(val) / 2 * self.is_in_prior_range(val)

    def logpdf(self,val):

        return np.log(self.prob(val))
    

class Sine(Prior):

    """Sine prior with bounds"""

    def __init__(self, minimum=0, maximum = np.pi, name=None, latex_label=None, unit=None):

        super(Sine,self).__init__(minimum=minimum, maximum=maximum, name=name, latex_label=latex_label, unit=unit)

    def rescale(self,val):

        norm = 1 / (np.cos(self.minimum) - np.cos(self.maximum))
        return np.arccos(np.cos(self.minimum) - val / norm)

    def prob(self,val):

        return np.sin(val) / 2 * self.is_in_prior_range(val)

    def logpdf(self,val):
        
        return np.log(self.prob(val))

class Gaussian(Prior):

    """Gaussian prior with parameters
    mu (float): mean of the Gaussian prior
    sigma (float): width/standard deviation of the Gaussian prior
    """
    
    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None):
        
        super(Gaussian, self).__init__(name=name, latex_label=latex_label, unit=unit)

        self.mu = mu
        self.sigma = sigma

    def rescale(self,val):
        
        return self.mu + erfinv(2 * val - 1) * 2 ** 0.5 * self.sigma

    def prob(self,val):

        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 / self.sigma

    def logpdf(self,val):

        return -0.5 * ((self.mu - val) ** 2 / self.sigma ** 2 + np.log(2 * np.pi * self.sigma ** 2))


class Categorical(Prior):

    """Equal-weighted categorical prior, for example 
    used for waveform models.
    Parameters:
    ncategories (int): number of available categories. The prior mass support
                       is then integers [0, ncategories - 1]."""

    def __init__(self, ncategories, name=None, latex_label=None, unit=None):

        minimum = 0
        maximum = ncategories - 1 + 1e-15    # Small delta added in bilby to help with MCMC walking, for now we keep it

        super(Categorical,self).__init__(name=name, latex_label=latex_label, minimum=minimum,
                                         maximum=maximum, unit=unit)

        self.ncategories = ncategories
        self.categories = np.arange(self.minimum, self.maximum)
        self.p = 1 / self.ncategories
        self.lnp = -np.log(self.ncategories)


    def rescale(self,val):

        return np.floor(val * (1 + self.maximum))

    def prob(self, val):

        if isinstance(val, (float, int)):
            if val in self.categories:
                return self.p
            else:
                return 0

        else:
            val = np.atleast_1d(val)
            probs = np.zeros_like(val, dtype=np.float64)
            idxs = np.isin(val, self.categories)
            probs[idxs] = self.p
            return probs

    def logpdf(self,val):

        if isinstance(val, (float, int)):
            if val in self.categories:
                return self.lnp
            else:
                return -np.inf

        else:
            val = np.atleast_1d(val)
            probs = -np.inf * np.ones_like(val, dtype=np.float64)
            idxs = np.isin(val, self.categories)
            probs[idxs] = self.lnp
            return probs


class Interped(Prior):

    """
    Create interpolated prior function from arrays of xx and yy=p(xx).
    Used for more 'complex' priors in gw, like cosmological and spins.
    Arguments:
    xx (array_like): x values for the to be interpolated prior function
    yy (array_like): p(xx) values for the to be interpolated prior function."""

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None,
                 latex_label=None, unit=None):

        self.xx = xx
        self.min_limit = min(xx)
        self.max_limit = max(xx)
        self._yy = yy
        self.YY = None
        self.probability_density = None
        self.inverse_cumulative_distribution = None
        self.__all_interpolated = interp1d(x=xx, y=yy, bounds_error=False, fill_value=0)
        minimum = float(np.nanmax(np.array((min(xx), minimum))))
        maximum = float(np.nanmin(np.array((max(xx), maximum))))

        super(Interped, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                       minimum=minimum, maximum=maximum)
        self._update_instance()
        

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if np.array_equal(self.xx, other.xx) and np.array_equal(self.yy, other.yy):
            return True
        return False

    def prob(self, val):
        return self.probability_density(val)

    def logpdf(self,val):
        return np.log(self.prob(val))
    
    def rescale(self,val):

        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled

    @property
    def minimum(self):

        """Return minimum of the prior distribution.
        Updates the prior distribution if minimum is set to a different value.
        Yields an error if value is set below instantiated x-array minimum. """
        
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        
        if minimum < self.min_limit:
            raise ValueError('Minimum cannot be set below {}.'.format(round(self.min_limit, 2)))
        self._minimum = minimum
        if '_maximum' in self.__dict__ and self._maximum < np.inf:
            self._update_instance()

    @property
    def maximum(self):

        """Return maximum of the prior distribution.
        Updates the prior distribution if maximum is set to a different value. 
        Yields an error if value is set above instantiated x-array maximum."""

        return self._maximum

    @maximum.setter
    def maximum(self, maximum):

        if maximum > self.max_limit:
            raise ValueError('Maximum cannot be set above {}.'.format(round(self.max_limit, 2)))
        self._maximum = maximum
        if '_minimum' in self.__dict__ and self._minimum < np.inf:
            self._update_instance()

    @property
    def yy(self):

        """Return p(xx) values of the interpolated prior function.
        Updates the prior distribution if it is changed."""

        return self._yy

    @yy.setter
    def yy(self,yy):
        self._yy = yy
        self.__all_interpolated = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self._update_instance()

    def _update_instance(self):
        self.xx = np.linspace(self.minimum, self.maximum, len(self.xx))
        self._yy = self.__all_interpolated(self.xx)
        self._initialize_attributes()

    def _initialize_attributes(self):
        from scipy.integrate import cumulative_trapezoid
        if trapezoid(self._yy, self.xx) != 1:
            print('Supplied PDF for {} is not normalised, normalising.'.format(self.name))
        self._yy /= trapezoid(self._yy, self.xx)
        self.YY = cumulative_trapezoid(self._yy, self.xx, initial=0)
        # Need last element of cumulative distribution to be exactly one.
        self.YY[-1] = 1
        self.probability_density = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=(0, 1))
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=True)


class Cosmological(Interped):

    """Base prior for distance parameters that make sense cosmologically
    Parameters:
    cosmology (str): the cosmology to use, see `bilby.gw.cosmology.set_cosmology` for details.
                     If nothing is specified, default cosmology is used."""
    

    @property
    def _default_args_dict(self):
        from astropy import units
        return dict(
            redshift=dict(name='redshift', latex_label='$z$', unit=None),
            luminosity_distance=dict(name='luminosity_distance', latex_label='$d_L$', unit=units.Mpc),
            comoving_distance=dict(name='comoving_distance', latex_label='$d_C$', unit=units.Mpc))

    def __init__(self, minimum, maximum, cosmology=None, name=None, latex_label=None, unit=None):
        
        from astropy import units
        self.cosmology = get_cosmology(cosmology)
        if name not in self._default_args_dict:
            raise ValueError("Name {} not recognised. Must be one of luminosity_distance, "
                             "comoving_distance, redshift".format(name))
        self.name = name
        label_args = self._default_args_dict[self.name]
        if latex_label is not None:
            label_args['latex_label'] = latex_label
        if unit is not None:
            if not isinstance(unit, units.Unit):
                unit = units.Unit(unit)
            label_args['unit'] = unit

        self.unit = label_args['unit']
        self._minimum = dict()
        self._maximum = dict()
        self.minimum = minimum
        self.maximum = maximum

        if name == 'redshift':
            xx, yy = self._get_redshift_arrays()
        elif name == 'comoving_distance':
            xx,yy = self._get_comoving_distance_arrays()
        elif name == 'luminosity_distance':
            xx, yy = self._get_luminosity_distance_arrays()
        else:
            raise ValueError('Name {} not recognized.'.format(name))

        super(Cosmological, self).__init__(xx=xx, yy=yy, minimum=minimum, maximum=maximum, **label_args)

    @property
    def minimum(self):
        return self._minimum[self.name]

    @minimum.setter
    def minimum(self, minimum):
        if (self.name in self._minimum) and (minimum < self.minimum):
            self._set_limit(value=minimum, limit_dict=self._minimum, recalculate_array=True)
        else:
            self._set_limit(value=minimum, limit_dict=self._minimum)

    @property
    def maximum(self):
        return self._maximum[self.name]

    @maximum.setter
    def maximum(self,maximum):
        if (self.name in self._maximum) and (maximum > self.maximum):
            self._set_limit(value=maximum, limit_dict=self._maximum, recalculate_array=True)
        else:
            self._set_limit(value=maximum, limit_dict=self._maximum)

    def _set_limit(self, value, limit_dict, recalculate_array=False):

        """Set either of the limits for redshift, luminosity, and comoving distances"""

        cosmology = get_cosmology(self.cosmology)
        limit_dict[self.name] = value
        if self.name == 'redshift':
            limit_dict['luminosity_distance'] =cosmology.luminosity_distance(value).value
            limit_dict['comoving_distance'] = cosmology.comoving_distance(value).value
        elif self.name == 'luminosity_distance':
            if value == 0:
                limit_dict['redshift'] = 0
            else:
                limit_dict['redshift'] = z_at_value(cosmology.luminosity_distance, value * self.unit)
            limit_dict['comoving_distance'] = (cosmology.comoving_distance(limit_dict['redshift']).value)
        elif self.name == 'comoving_distance':
            if value == 0:
                limit_dict['redshift']=0
            else:
                limit_dict['redshift'] = z_at_value(cosmology.comoving_distance, value * self.unit)
                limit_dict['luminosity_distance'] = (cosmology.luminosity_distance(limit_dict['redshift']).value)

        if recalculate_array:
            if self.name == 'redshift':
                self.xx, self.yy = self._get_redshift_arrays()
            elif self.name == 'comoving_distance':
                self.xx, self.yy = self._get_comoving_distance_arrays()
            elif self.name == 'luminosity_distance':
                self.xx, self.yy = self._get_luminosity_distance_arrays()
        try:
            self._update_instance()
        except (AttributeError, KeyError):
            pass

    def _convert_to(self, new, args_dict):
        args_dict.update(self._default_args_dict[new])
        args_dict['minimum'] = self._minimum[args_dict['name']]
        args_dict['maximum'] = self._maximum[args_dict['name']]

    def _get_comoving_distance_arrays(self):

        zs, p_dz = self._get_redshift_redshift()
        dc_of_z = self.cosmology.comoving_distance(zs).value
        ddc_dz = np.gradient(dc_of_z, zs)
        p_dc = p_dz / ddc_dz

        return dc_of_z, p_dc

    def _get_luminosity_distance_arrays(self):

        zs, p_dz = self._get_redshift_arrays()
        dl_of_z = self.cosmology.luminosity_distance(zs).value
        ddl_dz = np.gradient(dl_of_z, zs)
        p_dl = p_dz / ddl_dz
        return dl_of_z, p_dl
    
    def _get_redshift_arrays(self):
        raise NotImplementedError

    def get_instantiation_dict(self):
        from astropy import units
        from astropy.cosmology.realizations import available
        instantiation_dict = super().get_instantiation_dict()
        if self.cosmology.name in available:
            instantiation_dict['cosmology'] = self.cosmology.name
        if isinstance(self.unit, units.Unit):
            instantiation_dict['unit'] = self.unit.to_string()
        return instantiation_dict


class UniformComovingVolume(Cosmological):

    """Prior that is uniform in the comoving volume.
       p(z) \propto \frac{d_V_{c}}{dz}"""

    def _get_redshift_arrays(self):

        zs = np.linspace(self._minimum['redshift'] * 0.99, self._maximum['redshift'] * 1.01, 1000) 
        p_dz = self.cosmology.differential_comoving_volume(zs).value
        return zs, p_dz

class AlignedSpin(Interped):

    """Prior distribution for the aligned (z) component of the spin.
    This takes prior distributions for the magnitude and cosine of the tilt
    and forms a compound prior using a numerical convolution integral.
    
    \pi(\chi) = \int_{0}^{1} da \int_{-1}^{1} d\cos\theta
    \pi(a) \pi(\cos\theta) \delta(\chi - a \cos\theta)
    
    This is an extension of e.g., (A7) of https://arxiv.org/abs/1805.10457.
    Parameters:
    a_prior (Prior): prior distribution for spin magnitude
    z_prior (Prior): prior distribution for cosine spin tilt"""

    def __init__(self, a_prior=Uniform(0,1), z_prior=Uniform(-1,1), name=None,
                 latex_label=None, unit=None, minimum=np.nan, maximum=np.nan, num_interp=None):

        self.a_prior = a_prior
        self.z_prior = z_prior
        chi_min = min(a_prior.maximum * z_prior.minimum, a_prior.minimum * z_prior.maximum)
        chi_max = a_prior.maximum * z_prior.maximum

        if self._is_simple_aligned_prior:
            self.num_interp = 100_000 if num_interp is None else num_interp
            xx = np.linspace(chi_min, chi_max, self.num_interp)
            yy = - np.log(np.abs(xx) / a_prior.maximum) / (2 * a_prior.maximum)

        else:
            def integrand(aa, chi):
                """
                The integrand for the aligned spin (chi) probability density
                after performing the integral over spin orientation using a
                delta function identity.
                """
                return a_prior.prob(aa) * z_prior.prob(chi / aa) / aa

            self.num_interp = 10_000 if num_interp is None else num_interp
            xx = np.linspace(chi_min, chi_max, self.num_interp)
            yy = [
                quad(integrand, a_prior.minimum, a_prior.maximum, chi)[0]
                for chi in xx
            ]

        super(AlignedSpin, self).__init__(xx=xx, yy=yy, name=name, latex_label=latex_label,
                                          unit=unit, minimum=minimum, maximum=maximum)

    @property
    def _is_simple_aligned_prior(self):
        return (
            isinstance(self.a_prior, Uniform)
            and isinstance(self.z_prior, Uniform)
            and self.z_prior.minimum == -1
            and self.z_prior.maximum == 1
        )


class ProbDistContainer:
    """Container for holding and generating prior info.

    Args:
        priors_in (dict): Dictionary with keys as int or tuple of int
            describing which parameters the prior takes. Values are
            probability distributions with ``logpdf`` and ``rvs`` methods.

    Attributes:
        priors_in (dict): Dictionary with keys as int or tuple of int
            describing which parameters the prior takes. Values are
            probability distributions with ``logpdf`` and ``rvs`` methods.
        priors (list): list of indexes and their associated distributions arranged
            in a list.
        ndim (int): Full dimensionality.
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)

    Raises:
        ValueError: Missing parameters or incorrect index keys.

    """

    def __init__(self, 
                 priors_in: dict = None,
                 filename: str = None,
                 use_cupy=False):
        
        if filename is not None:
            self.from_file(filename)
        
        elif priors_in is not None:
            self.from_dict(priors_in, use_cupy)

    def from_file(self, filename):
        comments = ["#", "\n"]
        prior = dict()
        with open(filename, "r", encoding="unicode_escape") as f:
            for line in f:
                if line[0] in comments:
                    continue
                line.replace(" ", "")
                elements = line.split("=")
                key = elements[0].replace(" ", "")
                val = "=".join(elements[1:]).strip()

                if isinstance(val, Prior):
                    continue
                elif isinstance(val, (int, float)):
                    val = DeltaFunction(peak=val)
                elif isinstance(val, str):

                    cls_str = val.split("(")[0]
                    cls_str = cls_str.split(".")[-1]
                    args = "("+ "(".join(val.split("(")[1:])
                    try:
                        cls = globals()[cls_str]
                    except NameError:
                        logger.error(f"Cannot load prior class {cls_str} from entry {key}={val}.")
                    val = eval(cls_str + args, {cls_str: cls, "np": np, "Uniform": Uniform})

                prior[key] = val
        
        self.from_dict(prior)

    
    def from_dict(self, priors_in, use_cupy=False):

        # copy to have
        self.priors_in = priors_in.copy()

        # to separate out in list form
        self.priors = []

        # setup lists
        temp_inds = []
        for inds, entries in enumerate(priors_in.items()):
            dist = entries[1]
            # multiple index
            if isinstance(inds, tuple):
                inds_in = np.asarray(inds)
                self.priors.append([inds_in, dist])

            # single index
            elif isinstance(inds, int):
                inds_in = np.array([inds])
                self.priors.append([inds_in, dist])

            else:
                raise ValueError(
                    "Keys for prior dictionary must be an integer or tuple."
                )

            temp_inds.append(np.asarray([inds_in]))

        uni_inds = np.unique(np.concatenate(temp_inds, axis=1).flatten())
        if len(uni_inds) != len(np.arange(np.max(uni_inds) + 1)):
            raise ValueError(
                "Please ensure all sampled parameters are included in priors."
            )

        self.ndim = uni_inds.max() + 1

        self.use_cupy = use_cupy
    
    def remove_parameter(self, param: str) -> None:

        if param in self.priors_in:

            priors_in = self.priors_in.copy()
            del priors_in[param]
            self.from_dict(priors_in)

    def logpdf(self, x, keys=None):
        """Get logpdf by summing logpdf of individual distributions

        Args:
            x (double np.ndarray[..., ndim]):
                Input parameters to get prior values.
            keys (list, optional): List of keys related to which parameters to gather the logpdf for.
                They must exactly match the input keys for the ``priors_in`` dictionary for the ``__init__`` 
                function. Even when using this kwarg, must provide all ``ndim`` parameters as input. The prior will just not 
                be calculated if its associated key is not included. Default is ``None``.

        Returns:
            np.ndarray[...]: Prior values.

        """
        xp = np if not self.use_cupy else cp

        # make sure at least 2D
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True

        elif x.ndim != 2:
            raise ValueError("x needs to 1 or 2 dimensional array.")
        else:
            squeeze = False

        prior_vals = xp.zeros(x.shape[0])

        # sum the logs (assumes parameters are independent)
        for i, (inds, prior_i) in enumerate(self.priors):

            if keys is not None:
                if len(inds) > 1:
                    if tuple(inds) not in keys:
                        continue
                else:
                    if inds[0] not in keys:
                        continue

            vals_in = x[:, inds]
            if hasattr(prior_i, "logpdf"):
                temp = prior_i.logpdf(vals_in)
            else:
                temp = prior_i.logpmf(vals_in)

            prior_vals += temp.squeeze()

        # if only one walker was asked for, return a scalar value not an array
        if squeeze:
            prior_vals = prior_vals[0].item()

        return prior_vals

    def ppf(self, x, groups=None):
        """Get logpdf by summing logpdf of individual distributions

        Args:
            x (double np.ndarray[..., ndim]):
                Input parameters to get prior values.

        Returns:
            np.ndarray[...]: Prior values.

        """
        raise NotImplementedError
        if groups is not None:
            raise NotImplementedError

        xp = np if not self.use_cupy else cp

        # TODO: check if mutliple index prior will work
        is_1d = x.ndim == 1
        x = xp.atleast_2d(x)
        out_vals = xp.zeros_like(x)

        # sum the logs (assumes parameters are independent)
        for i, (inds, prior_i) in enumerate(self.priors):
            if len(inds) > 1:
                raise NotImplementedError

            vals_in = x[:, inds].squeeze()
            temp = prior_i.ppf(vals_in)

            out_vals[:, inds[0]] = temp

        if is_1d:
            return out_vals.squeeze()
        return out_vals

    def rvs(self, size=1, keys=None):
        """Generate random values according to prior distribution

        The user will have to be careful if there are prior functions that
        do not have an ``rvs`` method. This means that generated points may lay
        inside the prior of all input priors that have ``rvs`` methods, but
        outside the prior if priors without the ``rvs`` method are included.

        Args:
            size (int or tuple of ints, optional): Output size for number of generated
                sources from prior distributions.
            keys (list, optional): List of keys related to which parameters to generate.
                They must exactly match the input keys for the ``priors_in`` dictionary for the ``__init__`` 
                function. If used, it will produce and output array of ``tuple(size) + (len(keys),)``. 
                Default is ``None``.

        Returns:
            np.ndarray[``size + (self.ndim,)``]: Generated samples.

        Raises:
            ValueError: If size is not an int or tuple.


        """

        # adjust size if int
        if isinstance(size, int):
            size = (size,)

        elif not isinstance(size, tuple):
            raise ValueError("Size must be int or tuple of ints.")

        xp = np if not self.use_cupy else cp

        # setup the slicing to properly sample points
        out_inds = tuple([slice(None) for _ in range(len(size))])

        # setup output and loop through priors

        ndim = self.ndim

        out = xp.zeros(size + (ndim,))
        for i, (inds, prior_i) in enumerate(self.priors):
            # only generate desired parameters
            if keys is not None:
                if len(inds) > 1:
                    if tuple(inds) not in keys:
                        continue
                else:
                    if inds[0] not in keys:
                        continue

            # guard against extra prior functions without rvs methods
            if not hasattr(prior_i, "rvs"):
                continue
            # combines outer dimensions with indices of interest
            inds_in = out_inds + (inds,)

            # allows for proper adding of quantities to out array
            if len(inds) == 1:
                adjust_inds = out_inds + (None,)
                out[inds_in] = prior_i.rvs(size=size)[adjust_inds]
            else:
                out[inds_in] = prior_i.rvs(size=size)

        return out
