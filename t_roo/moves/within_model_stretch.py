# -*- coding: utf-8 -*-

import numpy as np
import copy

from .red_blue_for_rj import RedBlueMove

__all__ = ["StretchMove"]


class WithinModelStretchMove(RedBlueMove):
    """Affine-Invariant Proposal

    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    This class was originally implemented in ``emcee`` and later in ``eryn``.

    Adapted to our rjmcmc case, including a proposal cycle and the random walk for the pseudo-Lambda(s).

    Args:
        branch_names (list[str]): names of the branches (i.e. models) that are used in the ensemble sampler
        binary_types (list[str] | dict[str, str]): binary types of the corresponding branches. If a dict, the keys have to be branch names with values in 'bbh', 'nsbh', 'bns'.
        a (double, optional): The stretch scale parameter. (default: ``2.0``)
        step_rw (float, optional): The random walk step size for lambda parameters in the bbh model. Defaults to 10.
        rw_lower_limit (float, optional): Lower limit for random walk parameters (for lambdas default is 0.)
        rw_upper_limit (float, optional): Upper limit for random walk parameters (for lambdas default to 5000).
        kwargs (dict, optional): Additional keyword arguments passed down through :class:`RedRedBlueMove`_.

    """

    def __init__(self, 
                 branch_names: list[str],
                 binary_types: list[str],
                 a: float =2.0, 
                 step_rw: float =10.,
                 rw_lower_limit: float = 0.,
                 rw_upper_limit: float = 5000.,
                 random_seed=None, 
                 **kwargs) -> None:
        
        self.branch_names = branch_names
        self.a = a
        self.step_rw = step_rw
        self.rw_lower_limit = rw_lower_limit
        self.rw_upper_limit = rw_upper_limit


        if isinstance(binary_types, list):
            if len(binary_types) != len(branch_names):
                raise ValueError(f"Provided binary types are {binary_types} which does not match branch names {branch_names}")
            self.binary_types = dict(zip(branch_names, binary_types))

        
        elif isinstance(binary_types, dict):
            assert branch_names == list(self.binary_types.keys()), f"binary types dict needs to have the branch names as keys."
            self.binary_types = binary_types
        
        else:
            raise ValueError(f"binary_types must either be provided as list or dict with branch_names keys.")
        
        invalid = set(self.binary_types.values()) - {"bbh", "nsbh", "bns"}
        if invalid:
            raise ValueError(f"Binary types {invalid} not implemented.")

        # pass kwargs up
        super().__init__(**kwargs)
    
    def synchronize_with_sampler(self, ensemble_sampler):

        """
        Prepares data needed by sampler: parameters names and indeces,
        temperature control, and for which indeces a random walk is needed.
        """
        self.temperature_control = ensemble_sampler.temperature_control
        if len(self.binary_types) != len(ensemble_sampler.branch_names):
            raise ValueError(f"WithinModelStretchMove has binaries {self.binary_types} which does not match ensemble sampler {ensemble_sampler.branch_names}")
                
        self.binary_types = dict(zip(ensemble_sampler.branch_names, self.binary_types.values()))
        self.params_to_inds = ensemble_sampler.params_to_inds
        self.inds_to_params = ensemble_sampler.inds_to_params

        self._determine_random_walk_indices()
    
    def _determine_random_walk_indices(self,):

        """
        Function to find which for which indices for each model the random walk will be performed.
        """
        self.random_walk_inds = {}

        for branch_name in self.branch_names:

            idx_random_walk = []

            if self.binary_types[branch_name]=="bbh":
                idx_random_walk.extend(self.params_to_inds[p] for p in ["lambda_1", "lambda_2"] if p in self.params_to_inds)
            elif self.binary_types[branch_name]=="nsbh":
                idx_random_walk.extend(self.params_to_inds[p] for p in ["lambda_1"] if p in self.params_to_inds)
            
            self.random_walk_inds[branch_name] = idx_random_walk

    def choose_c_vals(self, c, Nc, Ns, ntemps, random_number_generator, **kwargs):
        """Get the compliment array

        The compliment represents the points that are used to move the actual points whose position is
        changing.

        Args:
            c (np.ndarray): Possible compliment values with shape ``(ntemps, Nc, nleaves_max, ndim)``.
            Nc (int): Length of the ``...``: the subset of walkers proposed to move now (usually nwalkers/2).
            Ns (int): Number of generation points.
            ntemps (int): Number of temperatures.
            random_number_generator (object): Random state object.
            **kwargs (ignored): Ignored here. For modularity.

        Returns:
            np.ndarray: Compliment values to use with shape ``(ntemps, Ns, nleaves_max, ndim)``.

        """
        rint = random_number_generator.randint(
            Nc,
            size=(
                ntemps,
                Ns,
            ),
        )
        c_temp = self.xp.take_along_axis(c, rint[:, :, None, None], axis=1)
        return c_temp


    
    def random_walk(self, params, random, lower_limit, upper_limit):

        """
        Random walk for the pseudo-parameters (lambda_1_bbh, lambda_2_bbh, lambda_1_nsbh)
        Same probability to move to smaller or larger values.
        Args:
            params(np.ndarray): parameters to which apply the random walk
            random (object): Random state object.
            lower_limit(float): lower allowed limit for the range of the parameters updated with random walk. 
                If the value proposed by random walk goes below we just keep the current values.
            upper_limit(float): upper allowed limit for the range of the parameters updated with random walk.
                If the value proposed by random walk goes above we just keep the current values.
        Returns:
            parameters updated with random walk
        """

        walk_up = random.choice([False, True], size=params.shape)

        params[walk_up] += self.step_rw 
        params[~walk_up] -= self.step_rw
        

        negative_inds = params<lower_limit
        params[negative_inds] += self.step_rw
        over_inds = params>upper_limit
        params[over_inds] -= self.step_rw

        return params
    
    def get_stretch_move(self, coords, compl, num_to_change, random, inds_stretch):

        """
        Stretch move to update parameters in the within-model moves.
        Args:
            coords (np.ndarray): coordinates of the states that we are updating in a specific model, shape(ntemps, num_change, nleaves_max, ndim).
            compl (np.ndarray): coordinates of the complementary set of walkers, shape((ntemps, num_change, nleaves_max, ndim)).
            num_to_change (int): number of points to change.
            random (object): Random state object.
            inds_stretch (list): list of indeces with the positions of parameters to be updated with the stretch move.

        Returns:
            tuple: new coordinates, log factor for symmetry condition
        """
        zz = (
            (self.a - 1.0) * random.rand(num_to_change) + 1
        ) ** 2.0 / self.a

        # get proper distance
        diff = compl[:, inds_stretch] - coords[:, inds_stretch]
        # Factors must be computed ony considering the dimension of the parameters changed by the stretch move
        ndim_here = len(inds_stretch)

        new_coords = coords.copy()
            
        if ndim_here != 0 :
            new_coords[:, inds_stretch]= compl[:, inds_stretch] - diff * zz[:, None]
            log_factor = (ndim_here - 1) * np.log(zz)
       
        else:
            log_factor = np.zeros(shape=(num_to_change))

        return new_coords, log_factor
            
    def get_new_points(
        self, name, s, c_temp, num_change, branch_shape, random_number_generator, inds_block_prop, lower_rw, upper_rw
    ):
        """Get mew points with a stretch move, and/or random walk for pseudo-lambdas, for the parameters in the current block of the proposal cycle.

        Takes compliment and uses it to get new points for those being proposed.

        Args:
            name (str): Branch name.
            s (np.ndarray): Points to be moved with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c_temp (np.ndarray): Compliment to move points with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            num_change (int): Number to generate.
            branch_shape (tuple): Full branch shape.
            random_number_generator (object): Random state object.
            inds_block_prop: list of indeces for parameters to update in block proposal.
            lower_rw: lower limit for the range of the parameters updated with a random walk.
            upper_rw: upper limit for the range of the parameters updated with a random walk.

        Returns:
            np.ndarray: New proposed points with shape ``(ntemps, num_change, nleaves_max, ndim)``.


        """

        # handle the block proposal
        if inds_block_prop=="all":
            inds_block_prop = list(range(0, s.shape[1]))
        

        idx_random = self.random_walk_inds[name] 
        idx_random = list(set(idx_random).intersection(inds_block_prop)) # if the random walk inds are in inds_block_prop use them
        inds_stretch = list(set(inds_block_prop) - set(idx_random)) # the other inds go into the stretch move

        params_random_walk = s[:, idx_random].copy()
        params_random_walk = self.random_walk(params_random_walk, random_number_generator, lower_rw, upper_rw)

        proposed_coords, factors = self.get_stretch_move(s, c_temp, num_change, random_number_generator, inds_stretch)
        proposed_coords[:, idx_random] = params_random_walk


        return proposed_coords, factors

    def get_proposal(self, s_all, c_all, inds_block_proposal, inds_all, random, gibbs_ndim=None, **kwargs):
        """Generate stretch proposal

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values.
            inds_block_proposal(str or list): list of indices corresponding to the parameters that need to be updated
                in the block sampling.If 'all', all parameters are updated
            inds_all(dict): Keys are ``branch_names`` and values are lists or True/False
                to show which model is used
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """
        
        random_number_generator = random
        new_pos = copy.deepcopy(s_all)

        branch_shape = list(s_all.values())[0].shape
        ntemps, nwalkers, _, _ = branch_shape
        factors = np.zeros((ntemps,nwalkers))
            
        # iterate over branches
        for i, name in enumerate(s_all):

            # get points to move
            s = self.xp.asarray(s_all[name])

            if not isinstance(c_all[name], list):
                raise ValueError("c_all for each branch needs to be a list.")

            # get compliment possibilities
            c = [self.xp.asarray(c_tmp) for c_tmp in c_all[name]]
            c = self.xp.concatenate(c, axis=1)

            Ns, Nc = s.shape[1], c.shape[1]

            # need to properly handle ndim
            if i == 0:
                Ns_check = Ns

            else:
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            # get actual compliment values
            c_temp = self.choose_c_vals(c, Nc, Ns, ntemps, random_number_generator)

            ### We change only the points where now inds are True
            ### Here we can set it in the same for loop since we are
            ### not changing indices at the end
            mask_inds_change = inds_all[name] == True

            s_for_change = s_all[name][mask_inds_change]
            c_for_change = c_temp[mask_inds_change]

            num_to_change = s_for_change.shape[0]
            # use stretch to get new proposals
            coords_to_change, factors_move = self.get_new_points(
                name, s_for_change, c_for_change, num_to_change, branch_shape, random_number_generator, inds_block_proposal, self.rw_lower_limit, self.rw_upper_limit
            )

            new_pos[name][mask_inds_change] = coords_to_change
            mask_factors = np.reshape(mask_inds_change,(ntemps,nwalkers))
            factors[mask_factors] = factors_move

        return new_pos, factors
