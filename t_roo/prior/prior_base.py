import numpy as np
from scipy import stats
from copy import deepcopy

"""
Module to define the prior class
"""

def infer_args_from_method(method):

    """ Infers all arguments of a method except for `self`                                                                                                                         
    Throws out `*args` and `**kwargs` type arguments.                                                                                                                              
    Basically reads all parameters except from 1st one"""

    parameters = inspect.getfullargspec(method).args
    del parameters[:1]
    return parameters

def get_dict_with_properties(obj):

    """Reads properties of an object and create dict"""

    property_names = [p for p in dir(obj.__class__)
                      if isinstance(getattr(obj.__class__, p), property)]
    dict_with_properties = obj.__dict__.copy()
    for key in property_names:
        dict_with_properties[key] = getattr(obj, key)
    return dict_with_properties

class Prior(object):
    
    _default_latex_labels = {}
    
                                                                                                                                                                       
    def __init__(self, name=None, latex_label=None, unit=None, minimum=-np.inf,
                 maximum=np.inf, check_range_nonzero=True, use_cupy=False):
        
        """                                                                                                                                                                       
        Try to define  a general prior class which can be inherited from all the other prior classes                                                                          
        Ideally similar to bilby in order to use the same GW priors                                                                                                          
        We also switch to a dictionary system where priors are identified by names                                                                                                             
        Implements a Prior object                                                                                                                                                
                                                                                                                                                                                  
        Parameters                                                                                                                                                                
        ==========                                                                                                                                                              
        name: str, optional                                                                                                                                                     
            Name associated with prior.                                                                                                                                       
        latex_label: str, optional                                                                                                                                              
            Latex label associated with prior, used for plotting.                                                                                                             
        unit: str, optional                                                                                                                                                  
            If given, a Latex string describing the units of the parameter.                                                                                                    
        minimum: float, optional                                                                                                                                              
            Minimum of the domain, default=-np.inf                                                                                                                                   
        maximum: float, optional                                                                                                                                                  
            Maximum of the domain, default=np.inf                                                                                                                                  
        check_range_nonzero: boolean, optional                                                                                                                                   
            If True, checks that the prior range is non-zero                                                                                                                         
        Note: we remove the boundary option since not sure it work with eryn                                                                                                        
        """
        
        if check_range_nonzero and maximum == minimum:
            raise ValueError("Min and max values are the same.")
        

        elif minimum > maximum:
            tmp = minimum
            minimum = maximum
            maximum = tmp
        
        self.name = name                                                                                                                                                          
        self.latex_label = latex_label
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.check_range_nonzero = check_range_nonzero
        self.least_recently_sampled = None
        self._is_fixed = False
        self.use_cupy = use_cupy


    #######################################################################
    #Define functions that will later be overwritten by each specific class
    #######################################################################

    def rvs(self,size=None):

        """ Generate random numbers between 0 and 1
        rvs function in eryn = sample + rescale

        Parameters (int or array of int): size, how many samples we want"""

        if not isinstance(size, int) and not isinstance(size, tuple):         
            raise ValueError("size must be an integer or tuple of ints.")
        

        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        self.least_recently_sampled = self.rescale(xp.random.rand(*size))

        return self.least_recently_sampled


    def rescale(self,val):

        """Rescale the distribution between 0 and 1 to the prior values
        Each prior class has a different rescale function

        Prameters (array of floats): val, random number between 0 and 1"""

        return None

    def prob(self, val):

        """ Return the probability of val
        Here just definition to inherit, must be overwritten by each class"""

        return np.nan

    def ln_prob(self, val):

        """Same as prob, but ln of probability"""

        with np.errstate(divide='ignore'):
            return np.nan

    ##########################################
    # For now ignoring cdf, if needed add later
    ###########################################


    def is_in_prior_range(self, val):

        """True if val in prior boundaries, zero otherwise
        Same as pdf_val in eryn UniformDistribution class"""

        return (val >= self.minimum) & (val <= self.maximum)

    @property
    def is_fixed(self):

        """ Return true if the prior is fixed and therefore the parameters
        should not be sampled over. Checks with delta function """

        return self._is_fixed

    @property
    def latex_label(self):

        """Latex label for plots. Not sure if needed in eryn,
        but for now we keep it"""

        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label=None):
        if latex_label is None:
            self.__latex_label = self.__default_latex_label
        else:
            self.__latex_label = latex_label

    @property
    def unit(self):
        return self.__unit
        
    @unit.setter
    def unit(self,unit):
        self.__unit = unit

    @property
    def latex_label_with_unit(self):

        """If unit is specifid, return latex string with label name and unit"""

        if self.unit is not None:
            return "{} [{}]".format(self.latex_label, self.unit)
        else:
            return self.latex_label

    @property
    def minimum(self):
        return self._minimum
    
    @minimum.setter
    def minimum(self, minimum):
        self._minimum=minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

    @property
    def width(self):
        return self.maximum - self.minimum

    def get_instantation_dict(self):

        """This should give us the priors as a name-value dictionary"""
        
        subclass_args = infer_args_from_method(self.__init__)
        dict_with_properties = get_dict_with_properties(self)
        return {key: dict_with_properties[key] for key in subclass_args}

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label

    ##################################################
    #For now ignoring json and argument string methods
    ##################################################


class Constraint(Prior):

    """Class to constrain parameter value between min and max, with extremes excluded"""
    
    def __init__(self, minimum, maximum, name=None, latex_label=None, unit=None):

        super(Comstraint, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                         latex_label=latex_label, unit=unit)
        self._is_fixed = True

    def prob(self,val):
        return (val > self.minimum) & (val < self.maximum)

        

    
    

     
        


    
