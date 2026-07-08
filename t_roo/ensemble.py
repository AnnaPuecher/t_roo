# -*- coding: utf-8 -*-

import warnings
import os

import numpy as np
from itertools import count
from copy import deepcopy

from .backends import Backend, HDFBackend
from .model import Model
from .moves import StretchMove, TemperatureControl, ReversibleJumpMove
from .pbar import get_progress_bar
from .logging import logger
from .state import State
from .prior import ProbDistContainer
from .utils.stopping import StoppingCriterionPrelSampling, StoppingCriterionRJMCMC

# from .utils import PlotContainer
from .utils import PeriodicContainer
from .utils.utility import groups_from_inds
from .prior import ProbDistContainer

__all__ = ["EnsembleSampler"]


try:
    from collections.abc import Iterable
except ImportError:
    # for py2.7, will be an Exception in 3.8
    from collections import Iterable


class EnsembleSampler(object):
    """An ensemble MCMC sampler

    The class controls the entire sampling run. Modified specifically for reversible jump mcmc runs for gravitational-wave
    signals from CBC sources. It can analyze the same signal with various waveform approximants (branches) describing BBH, NSBH, and BNS sources.
    Parallel tempering as inherited from eryn.
    It is also possible to run a 'standard' mcmc run with only one model.

    Parameters related to parallelization can be controlled via the ``pool`` argument.

    Args:
        nwalkers (int): The number of walkers in the ensemble per temperature.
        ndims (int, list of ints, or dict): The number of dimensions for each branch. If
                ``dict``, keys should be the branch names and values the associated dimensionality.
        log_like_fn (callable): A function that returns the natural logarithm of the
            likelihood for that position. To run on CBC GW signals, we use the wrapper function for the bilby likelihood "bilby_likelihood.bilbylik_wrap_rj"
            (see bilby_utils.py). All the args and kwargs specifically for the likelihood are pass before to the Likelihood class to reduce the computtaional cost.
        priors (dict): A dictionary with keys that are ``branch_names`` and values :class:`eryn.prior.ProbDistContainer` objects.
            For standard mcmc runs with only one model, priors is just a :class:`eryn.prior.ProbDistContainer` object.
        outdir (str): str to the directory in which the output files are going to be saved.
        tempering_kwargs (dict, optional): Keyword arguments for initialization of the
            tempering class: :class:`eryn.moves.tempering.TemperatureControl`.  (default: ``{}``)
        branch_names (list, optional): List of branch names. If ``None``, models will be assigned
            names as ``f"model_{index}"`` based on ``nbranches``. (default: ``None``)
        nbranches (int, optional): Number of branches (models) tested.
            Only used if ``branch_names is None``.
            (default: ``1``)
        nleaves_max (int, list of ints, or dict, optional): Maximum allowable leaf count for each branch.
            It should have the same length as the number of branches.
            If ``dict``, keys should be the branch names and values the associated maximal leaf value.
            (default: ``1``)
            Not directly used in t-roo because we always have one model possibly appearing only one time (so nleaves_max=1),
            but we keep it ensure dimension compatibility throughout the code.
        nleaves_min (int, list of ints, or dict, optional): Minimum allowable leaf count for each branch.
            It should have the same length as the number of branches. Only used with Reversible Jump.
            If ``dict``, keys should be the branch names and values the associated maximal leaf value.
            If ``None`` and using Reversible Jump, will fill all branches with zero.
            (default: ``0``)
            Not directly used in t-roo because we always have a model either be used (one time) or not (so nleaves_min=0),
            but we keep it ensure dimension compatibility throughout the code.
        pool (object, optional): An object with a ``map`` method that follows the same
            calling sequence as the built-in ``map`` function. This is
            generally used to compute the log-probabilities for the ensemble
            in parallel.
        moves (list or object, optional): This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(eryn.moves.StretchMove(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            If ``None``, the default will be :class:`StretchMove`.
            (default: ``None``)
        rj_moves (list or object, optional): If ``None`` or ``False``, reversible jump will not be included in the run.
            This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(eryn.moves.DistributionGenerateRJ(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            If ``True``, it defaults to :class:`DistributionGenerateRJ`.
            (default: ``None``)
        proposal_cycle: cycle of indices for block sampling. It mimics the structure of bilby mcmc proposal, but here
            we have always the same move (for now), we just change the parameters (indices) that are actually updated
        periodic (dict, optional): Keys are ``branch_names``. Values are dictionaries
            that have (key: value) pairs as (index to parameter: period). Periodic
            parameters are treated as having periodic boundary conditions in proposals.
        fill_zero_leaves_val (double, optional): When there are zero leaves in a
            given walker (across all branches), fill the likelihood value with
            ``fill_zero_leaves_val``. If wanting to keep zero leaves as a possible
            model, this should be set to the value of the contribution to the Likelihood
            from the data. (Default: ``-1e300``).
        num_repeats_in_model (int, optional): Number of times to repeat the in-model step
            within in one sampler iteration. When analyzing the acceptance fraction, you must
            include the value of ``num_repeats_in_model`` to get the proper denominator.
        num_repeats_rj (int, optional): Number of time to repeat the reversible jump step
            within in one sampler iteration. When analyzing the acceptance fraction, you must
            include the value of ``num_repeats_rj`` to get the proper denominator.
        track_moves (bool, optional): If ``True``, track acceptance fraction of each move
            in the backend. If ``False``, no tracking is done. If ``True`` and run is interrupted, it will check
            that the move configuration has not changed. It will not allow the run to go on
            if it is changed. In this case, the user should declare a new backend and use the last
            state from the previous backend. **Warning**: If the order of moves of the same move class
            is changed, the check may not catch it, so the tracking may mix move acceptance fractions together.

    Raises:
        ValueError: Any startup issues.

    """

    def __init__(
        self,
        nwalkers,
        ndims,
        log_like_fn,
        priors,
        parameter_names: list[str],
        outdir: str = "./outdir",
        tempering_kwargs={},
        branch_names=None,
        nbranches=1,
        nleaves_max=1,
        nleaves_min=0,
        pool=None,
        moves=None,
        rj_moves=None,
        proposal_cycle=None,
        stopping_kwargs={},
        stopping_prel_kwargs={},
        niter_check_proposal=None,
        args=None,
        kwargs=None,
        blobs_dtype=None,  # TODO check this
        periodic=None,
        fill_zero_leaves_val=-1e300,
        num_repeats_in_model=1,
        num_repeats_rj=1,
        track_moves=True,
    ):
        
        logger.info("Initializing t-roo ensemble sampler...")

        ###############################
        # Set up likelihood and prior #
        ###############################

        self.log_like_fn = _FunctionWrapper(log_like_fn, args, kwargs)
        
        # store priors
        self.priors = priors

        # store parameter names

        self.inds_to_params = dict(enumerate(parameter_names))
        self.params_to_inds = {p: j for j, p in enumerate(parameter_names)}
        self.check_masses = parameter_names

        # store proposal cycle
        self.proposal_cycle = proposal_cycle

        ###############################
        # Set up branches, dimensions #
        ###############################

        # set basic variables for sampling settings
        self.nbranches, self.branch_names, self.ndims, self.nleaves_max = self._setup_branches(branch_names, nbranches, ndims, nleaves_max)
        self.nwalkers = nwalkers

        ################################
        # Set up tempering information #
        ################################

        # setup tempering information
        # default is no temperatures
        if tempering_kwargs == {}:
            self.ntemps = 1
            self.temperature_control = None
        else:
            # get effective total dimension
            total_ndim = 0
            for key in self.branch_names:
                total_ndim += self.nleaves_max[key] * self.ndims[key]
            self.temperature_control = TemperatureControl(
                total_ndim, nwalkers, **tempering_kwargs
            )
            self.ntemps = self.temperature_control.ntemps

        #########################
        # Stopping criterion info
        #########################

        self.stopping_kwargs = stopping_kwargs
        self.stopping_prel_kwargs = stopping_prel_kwargs
        
        ##########################
        # Periodic parameters info
        ##########################
        
        if periodic is not None:
            raise ValueError("Periodic boundary conditions on priors not implemented yet.")
       
        ################
        # Set up moves # 
        ################

        self.niter_check_proposal = niter_check_proposal

        # Parse the move schedule
        if moves is None:
            if rj_moves is not None:
                raise ValueError("If providing rj_moves, must provide moves kwarg as well.")

            # defaults to stretch move
            self.moves = [
                StretchMove(
                    temperature_control=self.temperature_control,
                    periodic=periodic,
                    a=2.0,
                )
            ]
            self.weights = [1.0]

        elif isinstance(moves, Iterable):
            try:
                self.moves, self.weights = [list(tmp) for tmp in zip(*moves)]

            except TypeError:
                self.moves = moves
                self.weights = np.ones(len(moves))
        else:
            self.moves = [moves]
            self.weights = [1.0]

        self.weights = np.atleast_1d(self.weights).astype(float)
        self.weights /= np.sum(self.weights)

        for move in self.moves: 
            if hasattr(move, "synchronize_with_sampler"):
                move.synchronize_with_sampler(self)

        ###################
        # Set up rj moves #
        ###################

        if rj_moves is None:
            self.has_reversible_jump = False
        
        elif isinstance(rj_moves, bool):
            self.has_reversible_jump = rj_moves

        else:
            self.has_reversible_jump = True

        if self.has_reversible_jump:  
            self.rj_moves, self.rj_weights = self._setup_rj_moves(rj_moves, nleaves_min)

        logger.info(f"Has Reversible Jump: {self.has_reversible_jump}")


        #################################################
        # Sync moves on accepted, temperature, periodic #
        #################################################
        
        # make sure moves have temperature module and
        # prepare the per proposal accepted values that are held as attributes in the specific classes
        for move in self.moves:
            if move.temperature_control is None and self.temperature_control is not None:
                    move.temperature_control = self.temperature_control
            
            if move.periodic is None and periodic is not None:
                    move.periodic = periodic
            move.accepted = np.zeros((self.ntemps, self.nwalkers))
        
        if self.has_reversible_jump:
            for move in self.rj_moves:
                if move.temperature_control is None and self.temperature_control is not None:
                    move.temperature_control = self.temperature_control

                #if move.periodic is None and periodic is not None:
                #    move.periodic = periodic
                
                move.accepted = np.zeros((self.ntemps, self.nwalkers))

        #####################
        # Store some kwargs #
        #####################

        self.fill_zero_leaves_val = fill_zero_leaves_val
        self.num_repeats_in_model = num_repeats_in_model
        self.num_repeats_rj = num_repeats_rj
        self.track_moves = track_moves

        self.all_moves = self._get_all_move_dict()
        self.move_keys = list(self.all_moves.keys()) if self.track_moves else None

        # setup emcee-like basics
        self.pool = pool
        self.blobs_dtype = blobs_dtype

        ###################
        # Set up backends #
        ###################
        
        ### If we run a rjmcmc analysis, two separate backends are created for the preliminary sampling 
        ### and for the actual rjmcmc sampling. If we simply run an eryn-like analysis with only one model,
        ### only the main sampling backend in created.
 

        assert isinstance(outdir, str), 'outdir needs to be passed as a string.'
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        logger.info(f"Output will be provided in {outdir}.")
        
        random_state = None

        self.do_prel_sampling = False

        if self.has_reversible_jump:
        
            # set up backend for preliminary within model sampling
            self.backend_prel = HDFBackend(os.path.join(self.outdir, "chains_prel.hdf5"))
            self.prel_state = None
        
            if not self.backend_prel.initialized:
                self.backend_prel.reset( # this simply creates new file
                    nwalkers=self.nwalkers,
                    ndims=self.ndims,
                    branch_names=self.branch_names,
                    parameter_names=list(self.params_to_inds.keys()),
                    ntemps=self.ntemps,
                    nleaves_max=self.nleaves_max,
                    rj=self.has_reversible_jump,
                    moves=self.move_keys,
                    converged=False,
                    )
                self.do_prel_sampling = True
                self.it_prel = 0
                logger.info(f"Will do preliminary sampling from provided initial state.")
        
            else:
                # check whether the existing file is consistent with the current ensemble sampler
                self.check_backend_compatability(self.backend_prel)

                if not self.backend_prel.converged:
                    self.do_prel_sampling = True
                    random_state = self.backend_prel.random_state  # Get the last random state

                    # Grab the last step so that we can restart
                    self.it_prel = self.backend_prel.iteration
                    if self.it_prel > 0:
                        self.prel_state = self.backend_prel.get_last_sample()
                        logger.info(f"Will do preliminary sampling from checkpoint in {self.backend_prel.filename}.")
                
                else:
                    self.prel_state = self.backend_prel.get_last_sample()
        
        # set up backend for (rjmcmc) main sampling
        self.backend = HDFBackend(os.path.join(self.outdir, "chains.hdf5"))
        self.state = None

        if not self.backend.initialized:
            self.backend.reset( # this simply creates new file
                nwalkers=self.nwalkers,
                ndims=self.ndims,
                branch_names=self.branch_names,
                parameter_names=list(self.params_to_inds.keys()),
                ntemps=self.ntemps,
                nleaves_max=self.nleaves_max,
                rj=self.has_reversible_jump,
                moves=self.move_keys,
                )
            self.it = 0
            logger.info(f"Will do main sampling from provided initial state.")

        else:
            # check whether the existing file is consistent with the current ensemble sampler
            self.check_backend_compatability(self.backend)

            try:
                self.it = self.backend.iteration
            except:
                self.it = 0

            if not self.do_prel_sampling:
                random_state = self.backend.random_state

                # Grab the last step so that we can restart
                self.it = self.backend.iteration
                if self.it > 0:
                    self.state = self.backend.get_last_sample()
                    logger.info(f"Will do main sampling from checkpoint provided in {self.backend.filename}.")

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        if random_state is None:
                random_state = np.random.get_state()
        self._random = np.random.mtrand.RandomState()
        self._random.set_state(random_state)

        logger.info("Initializing t-roo ensemble sampler.... DONE")


    def _setup_branches(self, branch_names, nbranches, ndims, nleaves_max):
        # turn things into lists/dicts if needed
        if branch_names is None:
            branch_names = ["model_{}".format(i) for i in range(nbranches)]
        
        elif isinstance(branch_names, str):
            branch_names = [branch_names]
        
        elif not isinstance(branch_names, list ):
            raise ValueError("branch_names must be string or list of strings.")

        nbranches = len(branch_names)

        if isinstance(ndims, int):
            assert len(branch_names) == 1
            ndims = {branch_names[0]: ndims}

        elif isinstance(ndims, list) or isinstance(ndims, np.ndarray):
            assert len(branch_names) == len(ndims)
            ndims = {bn: nd for bn, nd in zip(branch_names, ndims)}

        elif isinstance(ndims, dict):
            assert len(list(ndims.keys())) == len(branch_names)
            for key in ndims:
                if key not in branch_names:
                    raise ValueError(
                        f"{key} is in ndims but does not appear in branch_names: {branch_names}."
                    )
        else:
            raise ValueError("ndims is to be a scalar int, list or dict.")

        if isinstance(nleaves_max, int):
            assert len(branch_names) == 1
            nleaves_max = {branch_names[0]: nleaves_max}

        elif isinstance(nleaves_max, list) or isinstance(nleaves_max, np.ndarray):
            assert len(branch_names) == len(nleaves_max)
            nleaves_max = {bn: nl for bn, nl in zip(branch_names, nleaves_max)}

        elif isinstance(nleaves_max, dict):
            assert len(list(nleaves_max.keys())) == len(branch_names)
            for key in nleaves_max:
                if key not in branch_names:
                    raise ValueError(
                        f"{key} is in nleaves_max but does not appear in branch_names: {branch_names}."
                    )
        else:
            raise ValueError("nleaves_max is to be a scalar int, list, or dict.")
        
        return nbranches, branch_names, ndims, nleaves_max

    def _setup_rj_moves(self, rj_moves, nleaves_min):
        
        # check nleaves_min

        if nleaves_min is None:
            nleaves_min = {bn: 0 for bn in self.branch_names}
        elif isinstance(nleaves_min, int):
            assert self.nbranches == 1
            nleaves_min = {self.branch_names[0]: nleaves_min}
        elif isinstance(nleaves_min, (list, np.ndarray)):
            assert self.nbranches == len(nleaves_min)
            nleaves_min = {bn: nl for bn, nl in zip(self.branch_names, nleaves_min)}
        elif isinstance(nleaves_min, dict):
            assert len(list(nleaves_min.keys())) == self.nbranches
            for key in nleaves_min:
                if key not in self.branch_names:
                    raise ValueError(
                        f"{key} is in nleaves_min but does not appear in branch_names: {self.branch_names}."
                    )
        else:
            raise ValueError(
                "If providing nleaves_min, nleaves_min is to be a scalar int, list, or dict."
            )
        
        self.nleaves_min = nleaves_min
        
        # here we check rj_moves
       
        #if isinstance(rj_moves, ReversibleJumpMove):    
        rj_moves = [rj_moves]
        rj_weights = [1.0]

        #else: 
        #    raise ValueError(
        #        "rj_moves must be an instance of ReversibleJumpMove. List of moves not supported at the moment."
        #    )

        # adjust rj weights properly
        rj_weights = np.atleast_1d(rj_weights).astype(float)
        rj_weights /= np.sum(rj_weights)

        # warn if base stretch is used
        for move in self.moves:
            if type(move) == StretchMove:
                logger.warning(
                    "If using revisible jump, using the Stretch Move for in-model proposals is not advised. It will run and work, but it will not be using the correct complimentary group of parameters meaning it will most likely be very inefficient."
                )
        
        return rj_moves, rj_weights
    
    def _get_all_move_dict(self,):
        all_moves_tmp = self.moves.copy()

        if self.has_reversible_jump:
            all_moves_tmp.extend(self.rj_moves.copy())

        self.all_moves = {}
        counts = {}
        for move in all_moves_tmp:

            move = move[0] if isinstance(move, tuple) else move # get out of tuple if weights are given
            move_name = move.__class__.__name__ # get the name of the class instance as a string

            # need to keep track how many times each type of move class has been used
            move_number = counts.get(move_name, 0)
            counts[move_name] = move_number + 1

            # get the full name including the index
            full_move_name = move_name + f"_{counts[move_name]}"
            self.all_moves[full_move_name] = move
        
        return self.all_moves

    def check_backend_compatability(self, backend):
        # check moves
        if self.track_moves:
            # compare as sets, because the hdf5 file stores the move_keys 
            # as group keys, which are always alphabetically ordered
            if set(self.move_keys) != set(backend.move_keys):
                raise ValueError(
                        "Configuration of moves has changed. Cannot use the same backend. "
                        "Declare a new backend and start from the previous state. "
                        "If you would prefer not to track move acceptance fraction, "
                        "set track_moves to False in the EnsembleSampler."
                    )

        # Check the backend shape
        for name, shape in backend.shape.items():
            expected_shape = (self.ntemps, self.nwalkers, self.nleaves_max[name], self.ndims[name])
        
            if shape != expected_shape:
                raise ValueError(
                    (f"For the model {name} the shape of the backend {backend.filename} is {shape}," 
                     f" which is incompatible with the current sampler where the shape is {expected_shape}."
                    )
                )
    
    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.

        """
        return self._random.get_state()

    @random_state.setter  # NOQA
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work. Don't say I didn't warn you...

        """
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def priors(self):
        """
        Return the priors in the sampler.

        """
        return self._priors

    @priors.setter
    def priors(self, priors):
        """Set priors information.

        This performs checks to make sure the inputs are okay.

        """
        if isinstance(priors, dict):
            self._priors = {}

            for key in priors.keys():
                test = priors[key]
                if isinstance(test, dict):
                    # check all dists
                    for ind, dist in test.items():
                        if not hasattr(dist, "logpdf"):
                            raise ValueError(
                                "Distribution for model {0} and index {1} does not have logpdf method.".format(
                                    key, ind
                                )
                            )

                    self._priors[key] = ProbDistContainer(test)

                elif isinstance(test, ProbDistContainer):
                    self._priors[key] = test

                elif hasattr(test, "logpdf"):
                    self._priors[key] = {"model_0": test}

                else:
                    raise ValueError(
                        "priors dictionary items must be dictionaries with prior information or instances of the ProbDistContainer class."
                    )

        elif isinstance(priors, ProbDistContainer):
            self._priors = {"model_0": priors}

        else:
            raise ValueError("Priors must be a dictionary.")

        return

    @property
    def iteration(self):
        return self.backend.iteration

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d["pool"] = None
        return d

    def get_model(self):
        """Get ``Model`` object from sampler

        The model object is used to pass necessary information to the
        proposals. This method can be used to retrieve the ``model`` used
        in the sampler from outside the sampler.

        Returns:
            :class:`Model`: ``Model`` object used by sampler.

        """
        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        # setup model framework for passing necessary items
        model = Model(
            self.log_like_fn,
            self.compute_log_like,
            self.compute_log_prior,
            self.temperature_control,
            map_fn,
            self._random,
        )
        return model

    def sample(
        self,
        initial_state,
        store=True,
    ):
        """Main sampling function, with within- and between-model moves.
           If running standard MCMC with only one model (no RJ) only the within-model steps are performed.

        Args:
            initial_state (State or ndarray[ntemps, nwalkers, nleaves_max, ndim] or dict): The initial
                :class:`State` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions. If ``betas`` are
                provided in the state object, they will be loaded into the
                ``temperature_control``.
            store (bool, optional): By default, the sampler stores in the backend
                the positions (and other information) of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``. (default: ``True``)
        Returns:
            State: This generator yields the :class:`State` of the ensemble.

        Raises:
            ValueError: Improper initialization.

        """

        # Interpret the input as a walker state and check the dimensions.
        state = State(initial_state, copy=True)
        state = self._check_input_state(state)

        # get the model object
        model = self.get_model()

        # main sampling loop
        j=0
        while True:
            # within model moves
            accepted = np.zeros((self.ntemps, self.nwalkers))
 
            for repeat in range(self.num_repeats_in_model):
                # Choose a random move
                move = self._random.choice(self.moves, p=self.weights)

                if self.proposal_cycle is None:
                    state, accepted_out = move.propose(model, state, 'all')
                else:
                    self.proposal_cycle.synchronize_with_sampler(self)
                    move_inds = self.proposal_cycle.get_cycle_proposal(model.random)
                    state, accepted_out = move.propose(model, state, move_inds)
                        
                accepted += accepted_out
                if self.ntemps > 1:
                    in_model_swaps = move.temperature_control.swaps_accepted
                else:
                    in_model_swaps = None

                state.random_state = self.random_state


            if self.has_reversible_jump:
                # between-model moves
                rj_accepted = np.zeros((self.ntemps, self.nwalkers))

                for repeat in range(self.num_repeats_rj):
                    rj_move = self._random.choice(self.rj_moves, p=self.rj_weights)

                    if self.niter_check_proposal is not None:
                        if j!=0 and (j % self.niter_check_proposal) == 0.:
                            logger.info("Checking and updating proposal parameters")
                            rj_move.update_proposal(self)
                
                    # Propose (Between models)
                    state, rj_accepted_out = rj_move.propose(model, state)
                    rj_accepted += rj_accepted_out
                    state.random_state = self.random_state

                    # Save the new step
                    if store:

                        self.backend.grow(1, state.blobs)
                    
                        if self.track_moves:
                            moves_accepted_fraction = {
                                key: move_tmp.acceptance_fraction
                                for key, move_tmp in self.all_moves.items()
                            }
                        else:
                            moves_accepted_fraction = None

                        self.backend.save_step(
                            state,
                            accepted,
                            pre_steps = False,
                            rj_accepted=rj_accepted,
                            swaps_accepted=in_model_swaps,
                            moves_accepted_fraction=moves_accepted_fraction,
                        )

                    # Yield the result as an iterator so that the user can do all
                    # sorts of fun stuff with the results so far.
                    j +=1
                    yield state

    def prel_sample(self, 
                    initial_state,
                    thin_by=1, 
                    store=True):
        
        # Interpret the input as a walker state and check the dimensions.
        state = State(initial_state, copy=True)
        state = self._check_input_state(state)

        # Check that the thin keyword is reasonable.
        thin_by = int(thin_by)
        if thin_by <= 0:
            raise ValueError("Invalid thinning argument")
        
        model = self.get_model()

        j = 0
        while True:
            # in model moves
            accepted = np.zeros((self.ntemps, self.nwalkers))

            # Choose a random move
            move = self._random.choice(self.moves, p=self.weights)

            # Propose (within model)
            if self.proposal_cycle is None:
                state, accepted_out = move.propose(model, state, "all")
            else:
                self.proposal_cycle.synchronize_with_sampler(self)
                move_inds = self.proposal_cycle.get_cycle_proposal(model.random)
                state, accepted_out = move.propose(model, state, move_inds)
            accepted += accepted_out

            if self.ntemps > 1:
                in_model_swaps = move.temperature_control.swaps_accepted
            else:
                in_model_swaps = None
            state.random_state = self.random_state

            # Save the new step
            if store and (j + 1) % thin_by == 0:

                self.backend_prel.grow(1, state.blobs)

                if self.track_moves:
                    moves_accepted_fraction = {
                        key: move_tmp.acceptance_fraction
                        for key, move_tmp in self.all_moves.items()
                    }
                else:
                    moves_accepted_fraction = None

                self.backend_prel.save_step(
                    state,
                    accepted,
                    pre_steps = True,
                    swaps_accepted=in_model_swaps,
                    moves_accepted_fraction=moves_accepted_fraction,
                )

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            j += 1
            yield state
  

    def run_mcmc(self, 
                 initial_state: State, 
                 verbose: bool = True,
                 **kwargs
    ):
        """
        Main function to run the (rj)mcmc sampler. If reversible jump, it starts with the preliminary within-model sampling.
        Stops both the preliminary sampling and the reversible jump when the chosen stopping criteria are met.

        Args:
            initial_state (State or ndarray[ntemps, nwalkers, nleaves_max, ndim] or dict): The initial
                :class:`State` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions. If ``betas`` are
                provided in the state object, they will be loaded into the
                ``temperature_control``.
            nsteps (int): The number of steps to generate. The total number of proposals is ``nsteps * thin_by``.
            **kwargs : Parameters are directly passed to :func:`sample`.

        Returns:
            State: This method returns the most recent result from the (pre)sampling.

        Raises:
            ValueError: ``If initial_state`` is None and ``run_mcmc`` has never been called.

        """
        if self.do_prel_sampling and self.prel_state is None:
            self.prel_state = initial_state

        elif not self.do_prel_sampling and self.state is None:
            self.state = initial_state
         
        #####################################
        # PRELIMINARY WITHIN MODEL SAMPLING #
        #####################################

        if self.do_prel_sampling:
            prel_sampling_stopper = StoppingCriterionPrelSampling(n_walkers=self.nwalkers,
                                                                  **self.stopping_prel_kwargs)
            logger.info("\n"); logger.info("\n");
            logger.info(50*"=")
            logger.info("Starting preliminary within model sampling.")
            logger.info(" ")
            
            # preliminary sampling loop
            for results in self.prel_sample(self.prel_state):
                stop = prel_sampling_stopper(self.it_prel, results, self) # based on a stopping criterion
                self.it_prel +=1
                if stop:
                    break
            
            self.state = results
            self.backend_prel.set_as_converged()
            logger.info(" ")
            logger.info("Preliminary within model sampling finished.")
            logger.info(50*"=")
            logger.info("\n"); logger.info("\n");
            
        
        # synchronize the rj_moves with the preliminary sampling results
        if self.has_reversible_jump:
            for rj_move in self.rj_moves:
                rj_move.synchronize_with_sampler(self)
            logger.info("Synchronized reversible jump moves with sampler state.")

        ########################
        # MAIN SAMPLING RJMCMC #
        ########################

        sampling_stopper = StoppingCriterionRJMCMC(n_walkers=self.nwalkers,  
                                                    models = self.branch_names,
                                                    **self.stopping_kwargs)
        
        logger.info("\n"); logger.info("\n");
        logger.info(50*"=")
        logger.info("Starting t-roo RJMCMC sampling.")
        logger.info(" ")
        for results in self.sample(self.state):
            stop = sampling_stopper(self.it, results, self)
            self.it += 1
            if stop:
                break

        logger.info(" ")
        logger.info("t-roo rjmcmc sampling finished.")
        logger.info(50*"=")
        logger.info("\n"); logger.info("\n");

        # Store so that the ``initial_state=None`` case will work
        self.state = results
        return results

    def compute_log_prior(self, coords, inds=None):
        """Calculate the vector of log-prior for the walkers

        Args:
            coords (dict): Keys are ``branch_names`` and values are
                the position np.arrays[ntemps, nwalkers, nleaves_max, ndim].
                This dictionary is created with the ``branches_coords`` attribute
                from :class:`State`.
            inds (dict, optional): Keys are ``branch_names`` and values are
                the ``inds`` np.arrays[ntemps, nwalkers, nleaves_max] that indicates
                which leaves are being used. This dictionary is created with the
                ``branches_inds`` attribute from :class:`State`.
                (default: ``None``)

        Returns:
            np.ndarray[ntemps, nwalkers]: Prior Values

        """

        # get number of temperature and walkers
        ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape

        if inds is None:
            # default use all sources
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        # take information out of dict and spread to x1..xn
        x_in = {}

        # flatten coordinate arrays
        for i, (name, coords_i) in enumerate(coords.items()):
            ntemps, nwalkers, nleaves_max, ndim = coords_i.shape

            x_in[name] = coords_i.reshape(-1, ndim)

        prior_out = np.zeros((ntemps, nwalkers))
        for name in x_in:
            ntemps, nwalkers, nleaves_max, ndim = coords[name].shape
            prior_out_temp = (
                self.priors[name]
                .logpdf(x_in[name])
                .reshape(ntemps, nwalkers, nleaves_max)
            )

            # fix any infs / nans from binaries that are not being used (inds == False)
            prior_out_temp[~inds[name]] = 0.0

            # vectorized because everything is rectangular (no groups to indicate model difference)
            prior_out += prior_out_temp.sum(axis=-1)

        return prior_out

      

    def compute_log_like(
        self, coords, inds=None, logp=None, supps=None, branch_supps=None
    ):
        """Calculate the vector of log-likelihood for the walkers

        Args:
            coords (dict): Keys are ``branch_names`` and values are
                the position np.arrays[ntemps, nwalkers, nleaves_max, ndim].
                This dictionary is created with the ``branches_coords`` attribute
                from :class:`State`.
            inds (dict, optional): Keys are ``branch_names`` and values are
                the inds np.arrays[ntemps, nwalkers, nleaves_max] that indicates
                which leaves are being used. This dictionary is created with the
                ``branches_inds`` attribute from :class:`State`.
                (default: ``None``)
            logp (np.ndarray[ntemps, nwalkers], optional): Log prior values associated
                with all walkers. If not provided, it will be calculated because
                if a walker has logp = -inf, its likelihood is not calculated.
                This prevents evaluting likelihood outside the prior.
                (default: ``None``)

        Returns:
            tuple: Carries log-likelihood and blob information.
                First entry is np.ndarray[ntemps, nwalkers] with values corresponding
                to the log likelihood of each walker. Second entry is ``blobs``.

         Raises:
            ValueError: Infinite or NaN values in parameters.

        """

        # TODO: this is a giant mess that needs to be redone (H.)

        # if inds not provided, use all
        if inds is None:
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        # Check that the parameters are in physical ranges.
        for name, ptemp in coords.items():
            if np.any(np.isinf(ptemp[inds[name]])):
                raise ValueError("At least one parameter value was infinite")
            if np.any(np.isnan(ptemp[inds[name]])):
                raise ValueError("At least one parameter value was NaN")

        # if no prior values are added, compute_prior
        # this is necessary to ensure Likelihood is not evaluated outside of the prior
        if logp is None:
            logp = self.compute_log_prior(coords, inds=inds)

        # if all points are outside the prior
        if np.all(np.isinf(logp)):
            warnings.warn(
                "All points input for the Likelihood have a log prior of -inf."
            )
            return np.full_like(logp, -1e300), None

        # do not run log likelihood where logp = -inf
        inds_copy = deepcopy(inds)
        inds_bad = np.where(np.isinf(logp))
        for key in inds_copy:
            inds_copy[key][inds_bad] = False

            # if inds_keep in branch supps, indicate which to not keep
            if (
                branch_supps is not None
                and key in branch_supps
                and branch_supps[key] is not None
                and "inds_keep" in branch_supps[key]
            ):
                # TODO: indicate specialty of inds_keep in branch_supp
                branch_supps[key][inds_bad] = {"inds_keep": False}

        # take information out of dict and spread to x1..xn
        x_in = {}
        
        # determine groupings from inds
        groups = groups_from_inds(inds_copy)

        # need to map group inds properly
        # this is the unique group indexes
        unique_groups = np.unique(
            np.concatenate([groups_i for groups_i in groups.values()])
        )

        # this is the map to those indexes that are used in the likelihood
        groups_map = np.arange(len(unique_groups))

        # get the indices with groups_map for the Likelihood
        ll_groups = {}
        for key, group in groups.items():
            # get unique groups in this sub-group (or branch)
            temp_unique_groups, inverse = np.unique(group, return_inverse=True)

            # use groups_map by finding where temp_unique_groups overlaps with unique_groups
            keep_groups = groups_map[np.isin(unique_groups, temp_unique_groups)]

            # fill group information for Likelihood
            ll_groups[key] = keep_groups[inverse]

        for i, (name, coords_i) in enumerate(coords.items()):
            ntemps, nwalkers, nleaves_max, ndim = coords_i.shape
            nwalkers_all = ntemps * nwalkers

            # fill x_values properly into dictionary
            x_in[name] = coords_i[inds_copy[name]]

        # prepare group information
        # this gets the group_map indexing into a list
        groups_in = list(ll_groups.values())

        # if only one branch, take the group array out of the list
        if len(groups_in) == 1:
            groups_in = groups_in[0]

        # list of paramter arrays
        params_in = list(x_in.values())

        # each Likelihood is computed individually, for now we removed the vectorized possibility for simplicity
        
        # if groups in is an array, need to put it in a list.
        if isinstance(groups_in, np.ndarray):
            groups_in = [groups_in]

        # prepare input args for all Likelihood calls
        # to be spread out with map functions below
        args_in = []

        # each individual group in the groups_map
        for group_i in groups_map:
            # args and kwargs for the individual Likelihood
            arg_i = [None for _ in self.branch_names]
            kwarg_i = {}

            # iterate over the group information from the branches
            for branch_i, groups_in_set in enumerate(groups_in):
                # which entries in this branch are in the overall group tested
                # this accounts for multiple leaves (or model counts)
                inds_keep = np.where(groups_in_set == group_i)[0]

                branch_name_i = self.branch_names[branch_i]

                if inds_keep.shape[0] > 0:
                    # get parameters

                    params = params_in[branch_i][inds_keep]

                    # if leaf count is constant and leaf count is 1
                    # just give 1D parameters
                    if not self.has_reversible_jump and params.shape[0] == 1:
                        params = params[0]

                    # add them to the specific args for this Likelihood
                    arg_i[branch_i] = params
                        
            # if only one model type, will take out of groups
            add_term = arg_i[0] if len(groups_in) == 1 else arg_i

            # based on how this is dealth with in the _FunctionWrapper
            # add_term is wrapped in a list
            args_in.append([[add_term], kwarg_i])

        # If the `pool` property of the sampler has been set (i.e. we want
        # to use `multiprocessing`), use the `pool`'s map method.
        # Otherwise, just use the built-in `map` function.
        if self.pool is not None:
            map_func = self.pool.map

        else:
            map_func = map

        # get results and turn into an array
        results = np.asarray(list(map_func(self.log_like_fn, args_in)))

        assert isinstance(results, np.ndarray)

        # -1e300 because -np.inf screws up state acceptance transfer in proposals
        ll = np.full(nwalkers_all, -1e300)
        inds_fix_zeros = np.delete(np.arange(nwalkers_all), unique_groups)

        # make sure second dimension is not 1
        if results.ndim == 2 and results.shape[1] == 1:
            results = np.squeeze(results)

        # parse the results if it has blobs
        if results.ndim == 2:
            # get the results and put into groups that were analyzed
            ll[unique_groups] = results[:, 0]

            # fix groups that were not analyzed
            ll[inds_fix_zeros] = self.fill_zero_leaves_val

            # deal with blobs
            blobs_out = np.zeros((nwalkers_all, results.shape[1] - 1))
            blobs_out[unique_groups] = results[:, 1:]

        elif results.dtype == "object":
            # TODO: check blobs and add this capability
            raise NotImplementedError

        else:
            # no blobs
            ll[unique_groups] = results
            ll[inds_fix_zeros] = self.fill_zero_leaves_val

            blobs_out = None

        # return Likelihood and blobs
        return ll.reshape(ntemps, nwalkers), blobs_out
    
    def _check_input_state(self, state: State):
        # Check the backend shape
        for i, (name, branch) in enumerate(state.branches.items()):
            ntemps_, nwalkers_, nleaves_, ndim_ = branch.shape
            if (ntemps_, nwalkers_, nleaves_, ndim_) != (
                self.ntemps,
                self.nwalkers,
                self.nleaves_max[name],
                self.ndims[name],
            ):
                raise ValueError("incompatible input dimensions")

        # get log prior and likelihood if not provided in the initial state
        if state.log_prior is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_prior = self.compute_log_prior(coords, inds=inds)

        if state.log_like is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_like, state.blobs = self.compute_log_like(
                coords,
                inds=inds,
                logp=state.log_prior,
                supps=state.supplimental,  # only used if self.provide_supplimental is True
                branch_supps=state.branches_supplimental,  # only used if self.provide_supplimental is True
            )

        # get betas out of state object if they are there
        if state.betas is not None:
            if state.betas.shape[0] != self.ntemps:
                raise ValueError(
                    "Input state has inverse temperatures (betas), but not the correct number of temperatures according to sampler inputs."
                )

            self.temperature_control.betas = state.betas.copy()

        if np.shape(state.log_like) != (self.ntemps, self.nwalkers):
            raise ValueError("incompatible input dimensions")
        if np.shape(state.log_prior) != (self.ntemps, self.nwalkers):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(state.log_like)):
            raise ValueError("The initial log_like was NaN")

        if np.any(np.isinf(state.log_like)):
            raise ValueError("The initial log_like was +/- infinite")

        if np.any(np.isnan(state.log_prior)):
            raise ValueError("The initial log_prior was NaN")

        if np.any(np.isinf(state.log_prior)):
            raise ValueError("The initial log_prior was +/- infinite")

        return state

    @property
    def acceptance_fraction(self):
        """The fraction of proposed steps that were accepted"""
        return self.backend.accepted / float(self.backend.iteration)

    @property
    def rj_acceptance_fraction(self):
        """The fraction of proposed reversible jump steps that were accepted"""
        if self.has_reversible_jump:
            return self.backend.rj_accepted / float(self.backend.iteration)
        else:
            return None

    @property
    def swap_acceptance_fraction(self):
        """The fraction of proposed steps that were accepted"""
        return self.backend.swaps_accepted / float(self.backend.iteration)

    @property
    def rj_swap_acceptance_fraction(self):
        """The fraction of proposed reversible jump steps that were accepted"""
        if self.has_reversible_jump:
            return self.backend.rj_swaps_accepted / float(self.backend.iteration)
        else:
            return None

    def get_chain(self, **kwargs):
        return self.get_value("chain", **kwargs)

    get_chain.__doc__ = Backend.get_chain.__doc__

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)

    get_blobs.__doc__ = Backend.get_blobs.__doc__

    def get_log_like(self, **kwargs):
        return self.backend.get_log_like(**kwargs)

    get_log_like.__doc__ = Backend.get_log_prior.__doc__

    def get_log_prior(self, **kwargs):
        return self.backend.get_log_prior(**kwargs)

    get_log_prior.__doc__ = Backend.get_log_prior.__doc__

    def get_log_posterior(self, **kwargs):
        return self.backend.get_log_posterior(**kwargs)

    get_log_posterior.__doc__ = Backend.get_log_posterior.__doc__

    def get_inds(self, **kwargs):
        return self.get_value("inds", **kwargs)

    get_inds.__doc__ = Backend.get_inds.__doc__

    def get_nleaves(self, **kwargs):
        return self.backend.get_nleaves(**kwargs)

    get_nleaves.__doc__ = Backend.get_nleaves.__doc__

    def get_last_sample(self, **kwargs):
        return self.backend.get_last_sample()

    get_last_sample.__doc__ = Backend.get_last_sample.__doc__

    def get_betas(self, **kwargs):
        return self.backend.get_betas(**kwargs)

    get_betas.__doc__ = Backend.get_betas.__doc__

    def get_value(self, name, **kwargs):
        """Get a specific value"""
        return self.backend.get_value(name, **kwargs)

    def get_autocorr_time(self, **kwargs):
        """Compute autocorrelation time through backend."""
        return self.backend.get_autocorr_time(**kwargs)

    get_autocorr_time.__doc__ = Backend.get_autocorr_time.__doc__


class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """

    def __init__(
        self,
        f,
        args,
        kwargs,
    ):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, args_and_kwargs):
        """
        Internal function that takes a tuple (args, kwargs) for entrance into the Likelihood.

        ``self.args`` and ``self.kwargs`` are added to these inputs.

        """

        args_in_add, kwargs_in_add = args_and_kwargs

        try:
            args_in = args_in_add + type(args_in_add)(self.args)
            kwargs_in = {**kwargs_in_add, **self.kwargs}

            out = self.f(*args_in, **kwargs_in)
            return out

        except:  # pragma: no cover
            import traceback

            print("eryn: Exception while calling your likelihood function:")
            print("  args added:", args_in_add)
            print("  args:", self.args)
            print("  kwargs added:", kwargs_in_add)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


