# -*- coding: utf-8 -*-

"""
Contains the actual proposals for the between-model moves
to compare different models describing BNS, BBH, and NSBH systems, 
including jumps between models for the same source class.
"""

from ssl import ALERT_DESCRIPTION_NO_RENEGOTIATION
import numpy as np
import copy

from .rj_setup_gwbinaries import ReversibleJumpMove
from ..prior import ProbDistContainer
from ..utils.conversion import lambdaTilde_from_lambda1_lambda2_eta, symmetric_mass_ratio_from_coords, symmetric_mass_ratio_from_mass_ratio 

class GWBinariesRJ(ReversibleJumpMove):

    def __init__(self,
                 branch_names: list[str],
                 binary_types: list[str] | dict[str, str],
                 model_priors: list[str] = None,
                 epsilon: float =0.1,
                 steps_jump_params: int = 100,
                 *args,
                 **kwargs) -> None:
        '''
        Main reversible jump move class that handles the jumps between different waveform models. 

        Args:
            branch_names (list[str]): names of the branches (i.e. models) that are used in the ensemble sampler
            binary_types (list[str] | dict[str, str]): binary types of the corresponding branches. If a dict, the keys have to be branch names with values in 'bbh', 'nsbh', 'bns'.
            model_priors (list[str]): priors for the models, for now we always assume models with the same probability (default: None).
            epsilon (float): Interval to draw the stretch parameters from. Defaults to 0.1.
            steps_jump_params (int): number of final steps of the preliminary sampling to consider to find the chirp mass-mass ratio - lambda tilde 
                likelihood-center of mass for the computation of the slopes for the chirp mass and mass ratio proposals. 
            *args: args to be passed to ReversibleJumpMove
            **kwargs: kwargs to be passed to ReversibleJumpMove
        '''
        self.branch_names = branch_names
        nleaves_min = {key: 0 for key in branch_names}
        nleaves_max = {key: 1 for key in branch_names}
        super().__init__(nleaves_max, nleaves_min, *args, **kwargs)
        self.epsilon = epsilon
        self.steps_jump_params = steps_jump_params
        
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
        
        if model_priors is None:
            model_priors = np.full(len(self.branch_names), 1/len(self.branch_names))
        self.model_priors = np.array(model_priors)
        self.model_priors /= np.sum(self.model_priors)

    def get_proposal(self, 
                     all_coords, 
                     all_inds, 
                     nleaves_min_all, # for now nleaves is only here for compatability purposes
                     nleaves_max_all, 
                     random):

        """Make a proposal for a different model with different parameters.

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            nleaves_min_all (dict): Minimum values of leaf ount for each model. Must have same order as ``all_cords``. 
            nleaves_max_all (dict): Maximum values of leaf ount for each model. Must have same order as ``all_cords``. 
            random (object): Current random state of the sampler.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance, including the probability of
                the auxiliary variables and the Jacobian.

        """

        # prepare the output data structures
        # it is important to copy the old inds here to make sure in the next step of the for loop we propose a jump for the walkers
        # that were in the model considered at teh beginning, and not because of a jump from the previous model considered
        
        q = copy.deepcopy(all_coords)
        new_inds = copy.deepcopy(all_inds)
        old_inds = copy.deepcopy(all_inds)

        ntemps, nwalkers, _, _ = list(all_coords.values())[0].shape
        factors = np.zeros((ntemps, nwalkers))
        
        ### Loop to propose new points: based on the model in which a walker is, a jump is proposed to one of the other models.

        for current_branch in all_coords.keys():
                                     
            # We select one alternative model
            proposed_branch, factor_from_branch = self.propose_branch(current_branch, random)

            ### We look where the current model is on (True inds)
            ### We want to propose a jump from this model to a different one
            ### The indices where the current model is True are set to False (we switch off this model)
            ### The same indices in the alternative model are set to True (we switch on this model)
            ### The other indices should be taken care of in the next step of the for loop
            inds_to_change = old_inds[current_branch]
            new_inds[current_branch][inds_to_change] = False
            new_inds[proposed_branch][inds_to_change] = True
           
            coords_to_change, factors_from_jump = self.change_coords(q[current_branch], 
                                                                     current_branch, 
                                                                     proposed_branch, 
                                                                     inds_to_change, 
                                                                     random)

            q[proposed_branch][inds_to_change] = coords_to_change
        
            mask_factors = np.reshape(old_inds[current_branch], (ntemps, nwalkers))
            factors[mask_factors] = factors_from_jump + factor_from_branch
            
        return q, new_inds, factors
    
  
    def change_coords(self, 
                      old_coords, 
                      current_branch, 
                      proposed_branch, 
                      inds_to_change, 
                      random):
        """
        Decides which coordinate map to use based on the source type models involved in the proposal
        and uses it to get the new coordinates and the move factors.
        Args:
            old_coords(np.ndarray): coordinates of the current states, i.e., coordinates that we want to change, 
                with shape [ntemps, nwalkers, nleaves_max, ndim].
            current_branch (str): name of the current branch (we update the position of all the walkers in a given branch at a time).
            proposed_branch (str): name of the branch (model) to which we want to propose a jump.
            inds_to_change (np.ndarray): inds of coordinates to change.
            random (object): Current random state of the sampler.
        Returns:
            new coordinates, move factors
        """

        coordinate_changer = self.get_coordinate_map(self.binary_types[current_branch], 
                                                     self.binary_types[proposed_branch])
        
        return coordinate_changer(old_coords, inds_to_change, random)

    def synchronize_with_sampler(self, ensemble_sampler):

        """
        Synchromize with ensemble attributes and prepare what is needed for the move,
        such as computing the slopes needed for the chirp mass and mass ratio proposals.
        """
        self.temperature_control = ensemble_sampler.temperature_control
        self.nleaves_min = ensemble_sampler.nleaves_min
        self.nleaves_max = ensemble_sampler.nleaves_max

        if len(self.binary_types) != len(ensemble_sampler.branch_names):
            raise ValueError(f"GWBinariesRJ has binaries {self.binary_types} which does not match ensemble sampler {ensemble_sampler.branch_names}")
        
        self.binary_types = dict(zip(ensemble_sampler.branch_names, self.binary_types.values()))
        self.params_to_inds = ensemble_sampler.params_to_inds
        self.inds_to_params = ensemble_sampler.inds_to_params
        self.check_masses = ensemble_sampler.check_masses

        # fetch parameter slopes etc. from preliminary sampling results
        self._get_additional_jump_parameters(ensemble_sampler.backend_prel, self.steps_jump_params)

    def update_proposal(self, ensemble_sampler):
        """
        Updates the slopes for the chirp mass and mass ratio proposals. This could be used after 
        a given number of iterations, when the sampler should have a better estimate of the parameters.
        However more factors are needed to ensure that detailed balance is not violates, so this
        is currently not used.
        """
        self._get_additional_jump_parameters(ensemble_sampler.backend, self.steps_jump_params)

    def propose_branch(self, current_branch: str, random) -> tuple[str, float]:
        """
        Handles the selection of which branch to jump to.
        Makes sure that the probability is equally split between the number of sources and not models.
        Currently in t-roo we use the same number of models for each source class, so all models
        have the same probability.
        
        Args:
            current_branch (str): The current branch from which to jump from
            random (np.random.Generate): random generator object
        Returns:
            proposed branch (str): name of the branch to which we propose the jump.
            log_factor(float): potential factor related to the branch choice to ensure detailes balance.
        """
        
        current_index = self.branch_names.index(current_branch)
        alternative_models = self.branch_names[:current_index] + self.branch_names[current_index+1:]
        q = np.delete(self.model_priors, current_index) # this is the proposal distribution q
        q /= np.sum(q)
        prior_current = self.model_priors[current_index] # this is the model prior value

        proposed_branch = random.choice(alternative_models, p=q)
        proposed_index = self.branch_names.index(proposed_branch)
        prior_proposed = self.model_priors[proposed_index] # this is the model prior value
        q_reverse = np.delete(self.model_priors, proposed_index)
        q_reverse /= np.sum(q_reverse)

        # proposal probabilites
        q_current_to_proposed = q[alternative_models.index(proposed_branch)]
        alternative_models_reverse = self.branch_names[:proposed_index] + self.branch_names[proposed_index+1:]
        q_proposed_to_current = q_reverse[alternative_models_reverse.index(current_branch)]
        
        # factors include the model prior as well as the proposal probability q
        factor_from_branch = prior_proposed / prior_current * q_proposed_to_current / q_current_to_proposed

        return proposed_branch, np.log(factor_from_branch)

    def get_coordinate_map(self, 
                           current_binary_type: str,
                           proposed_binary_type: str):
        """
        Decides which deterministic map h (or h') to use based on current and proposed model.
        Args:
        current_binary_type (str): source class of the current model.
        proposed_binary_type (str): source class of the proposed model.
        """

        dispatch = {
            "bbh": {
                "bbh": self.identity,
                "nsbh": self.bbh_to_nsbh,
                "bns": self.bbh_to_bns,
            },
            "nsbh": {
                "bbh": self.nsbh_to_bbh,
                "nsbh": self.identity,
                "bns": self.nsbh_to_bns,
            },
            "bns": {
                "bbh": self.bns_to_bbh,
                "nsbh": self.bns_to_nsbh,
                "bns": self.identity,
            },
        }

        try:
            return dispatch[current_binary_type][proposed_binary_type]
        except KeyError:
            raise ValueError(
                f"Unsupported conversion: {current_binary_type} → {proposed_binary_type}"
            )
    
    def identity(self, old_coords, inds_to_change, random):
        """
        Mapping that keeps all coordinates the same.
        Used for jumps between models describing the same type of source.
        Args:
            old_coords (np.ndarray): coordinates to change.
            inds_to_change (np.ndarray): inds of coordinates to change.
        Returns:
            new coords, log_factor
        """

        coords_change = old_coords[inds_to_change].copy()
        num_change = inds_to_change.sum()
        factors_tot = np.zeros(shape=(num_change))
        
        return coords_change, factors_tot

    
    def bbh_to_nsbh(self, old_coords, inds_to_change, random):
        
        """
        This is the case where we are going from the bbh to nsbh models (Eq.15 in arxiv:).
        The main parameter to be affected is Lambda_2, since it encodes the tidal information for the nsbh model.
        Lambda_2 is proposed with a stretch move, while Lambda_1 (which is a pseudo-parameter, i.e., a parameter that will not effectively enter the likelihood computation,
        for both bbh and nsbh models) is kept the same.
        The chirp mass is proposed taking into account the expected distribution from the preliminary sampling, to avoid that the sampler gets stuck in one model if the 
        chirp mass posteriors are quite seprated.
        All the other parameters are kept the same.
            
        The complementary is chosen by picking a random walker among the other walkers of the other model, which is not changed at this stage
        (the ones with False indeces). Althouhg they might be switched on when jumping from a different model,
        this happens in a different for-loop step so this is safe also when running in parallel.
        
        Args:
            old_coords (np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): coordinates to change.
            inds_to_change (np.ndarray[ntemps, nwalkers, nleaves_max]): positions of coordinate to change (where BBH is True).
            random (np.random.Generate): random generator object.
        Return:
            new coordinates, move factors
        """

        num_change = inds_to_change.sum()
        idx_lam2 = self.params_to_inds["lambda_2"]
        coords_change = old_coords[inds_to_change].copy()

        compl_pool_lam2 = old_coords[~inds_to_change][:,idx_lam2]
        compl_lambda2 = random.choice(compl_pool_lam2, size=num_change)

        old_lambda2= old_coords[inds_to_change][:,idx_lam2]
        
        # stretch move lambda
        u = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)
        new_lambda2 = compl_lambda2 + u*(old_lambda2 - compl_lambda2)
        mask_negative_lambdas = new_lambda2 < 0. # make sure we only have positive lambdas
        new_lambda2[mask_negative_lambdas] = old_lambda2[mask_negative_lambdas]
        
        coords_change[:, idx_lam2] = new_lambda2

        #if self.mchirp_proposal_nsbh_bbh or self.mratio_proposal_nsbh_bbh:
        eta = symmetric_mass_ratio_from_coords(old_coords[inds_to_change], self.params_to_inds)
        new_lambdaT = lambdaTilde_from_lambda1_lambda2_eta(np.zeros_like(new_lambda2), new_lambda2, eta)

        if self.mchirp_proposal_nsbh_bbh:
            # update mchirp
            idx_mchirp = self.params_to_inds["chirp_mass"]
            old_mchirp_bbh = old_coords[inds_to_change][:,idx_mchirp]
            new_mchirp_nsbh = old_mchirp_bbh + self.slope_mchirp_lambdaT_nsbh_bbh * new_lambdaT
            coords_change[:, idx_mchirp] = new_mchirp_nsbh            

        #if self.mratio_proposal_nsbh_bbh:
        # update mratio
        idx_mratio = self.params_to_inds["mass_ratio"]
        old_mratio_bbh = old_coords[inds_to_change][:, idx_mratio]
        new_mratio_nsbh = old_mratio_bbh + self.slope_mratio_lambdaT_nsbh_bbh * new_lambdaT
        coords_change[:, idx_mratio] = new_mratio_nsbh

        # factors
        fact_g = 1./np.sqrt(u)
        v_from_u = 1./u
        fact_inverse = 1./np.sqrt(v_from_u)
        jacobian = 1./u  
        factors_tot = fact_inverse*jacobian/fact_g

        return coords_change, np.log(factors_tot)

    
    def nsbh_to_bbh(self, old_coords, inds_to_change, random):

        """
        Proposal to go from nsbh to bbh models (Eq.B14 in arxiv:).
        It is the inverse of the mapping from bbh to nsbh.
        Args:
            old_coords (np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): coordinates to change.
            inds_to_change (np.ndarray[ntemps, nwalkers, nleaves_max]): positions of coordinate to change (where NSBH is True).
            random (np.random.Generate): random generator object.
        Return:
            new coordinates, move factors
        """

        num_change = inds_to_change.sum()
        idx_lam2 = self.params_to_inds["lambda_2"]
        #idx_mchirp = self.params_to_inds["chirp_mass"]
        coords_change = old_coords[inds_to_change].copy()

        compl_pool_lam2 = old_coords[~inds_to_change][:,idx_lam2]
        compl_lambda2 = random.choice(compl_pool_lam2, size=num_change)

        old_lambda2 = old_coords[inds_to_change][:,idx_lam2]
        #old_mchirp_nsbh = old_coords[inds_to_change][:, idx_mchirp]
        
        # stretch move lambda
        v = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)
        new_lambda2 = compl_lambda2 + v*(old_lambda2 - compl_lambda2)
        mask_negative_lambdas = new_lambda2 < 0. # Make sure we do not have negative values of lambdas
        new_lambda2[mask_negative_lambdas] = old_lambda2[mask_negative_lambdas]
    
        coords_change[:, idx_lam2] = new_lambda2

        #if self.mchirp_proposal_nsbh_bbh or self.mratio_proposal_nsbh_bbh:
        eta = symmetric_mass_ratio_from_coords(old_coords[inds_to_change], self.params_to_inds)
        old_lambdaT = lambdaTilde_from_lambda1_lambda2_eta(np.zeros_like(old_lambda2), old_lambda2, eta)

        if self.mchirp_proposal_nsbh_bbh:        
            # update mchirp 
            #eta = symmetric_mass_ratio_from_coords(old_coords[inds_to_change], self.params_to_inds)
            #old_lambdaT = lambdaTilde_from_lambda1_lambda2_eta(np.zeros_like(old_lambda2), old_lambda2, eta)
            idx_mchirp = self.params_to_inds["chirp_mass"]
            old_mchirp_nsbh = old_coords[inds_to_change][:, idx_mchirp]
            new_mchirp_bbh = old_mchirp_nsbh - self.slope_mchirp_lambdaT_nsbh_bbh * old_lambdaT
            coords_change[:, idx_mchirp] = new_mchirp_bbh

        #if self.mratio_proposal_nsbh_bbh:
        # update mratio
        idx_mratio = self.params_to_inds["mass_ratio"]
        old_mratio_nsbh = old_coords[inds_to_change][:, idx_mratio]
        new_mratio_bbh = old_mratio_nsbh - self.slope_mratio_lambdaT_nsbh_bbh * old_lambdaT
        coords_change[:, idx_mratio] = new_mratio_bbh            


        # factors
        fact_g = 1./np.sqrt(v)
        u_from_v = 1./v 
        fact_inverse = 1./np.sqrt(u_from_v)
        jacobian = 1./v
        factors_tot = fact_inverse*jacobian/fact_g
       
        return coords_change, np.log(factors_tot)

    
    def bbh_to_bns(self, old_coords, inds_to_change, random):
       
        """
        This is the case where we are going from BBH to BNS models (Eq.B15 in arxiv: ).
        Lambda_1 and Lambda_2 are proposed with a stretch move.
        We implement a specific chirp mass and mass ratio proposals to account for the differences induced by the presence or absence of tidal effects, based on lambda_tilde.
        The symmetric mass ratio to compute Lambda_tilde can be computed from the old coordinates since it does not change in the between-model moves.
        All the other parameters are kept the same.

        The complementary is chosen by picking a random walker among the other walkers of the other model, which is not changed at this stage
        (the ones with False indeces). Althouhg they might be switched on when jumping from a different model,
        this happens in a different for-loop step so this is safe also when running in parallel.
        
        Args:
            old_coords (np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): coordinates to change.
            inds_to_change (np.ndarray[ntemps, nwalkers, nleaves_max]): positions of coordinate to change (where BBH is True).
            random (np.random.Generate): random generator object.
        Return:
            new coordinates, move factors

        """

        num_change = inds_to_change.sum()
        
        idx_lam1 = self. params_to_inds["lambda_1"]
        idx_lam2 = self.params_to_inds["lambda_2"]

        coords_change = old_coords[inds_to_change].copy()

        compl_pool_lam1 = old_coords[~inds_to_change][:,idx_lam1]
        compl_lambda1 = random.choice(compl_pool_lam1, size=num_change)

        compl_pool_lam2 = old_coords[~inds_to_change][:,idx_lam2]
        compl_lambda2 = random.choice(compl_pool_lam2, size=num_change)

        old_lambda1= old_coords[inds_to_change][:,idx_lam1]
        old_lambda2= old_coords[inds_to_change][:,idx_lam2]

        # stretch move lambda
        u1 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)
        u2 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)

        new_lambda1 = compl_lambda1 + u1*(old_lambda1 - compl_lambda1)
        new_lambda2 = compl_lambda2 + u2*(old_lambda2 - compl_lambda2)
        mask_negative_lambdas_1 = new_lambda1 < 0. # make sure we only have positive lambdas
        new_lambda1[mask_negative_lambdas_1] = old_lambda1[mask_negative_lambdas_1]
        mask_negative_lambdas_2 = new_lambda2 < 0. # make sure we only have positive lambdas
        new_lambda2[mask_negative_lambdas_2] = old_lambda2[mask_negative_lambdas_2]

        coords_change[:, idx_lam1] = new_lambda1
        coords_change[:, idx_lam2] = new_lambda2

        #if self.mchirp_proposal_bbh_bns or self.mratio_proposal_bbh_bns:
        eta = symmetric_mass_ratio_from_coords(old_coords[inds_to_change], self.params_to_inds)
        new_lambdaT = lambdaTilde_from_lambda1_lambda2_eta(new_lambda1, new_lambda2, eta)

        if self.mchirp_proposal_bbh_bns:
            # update mchirp
            idx_mchirp = self.params_to_inds["chirp_mass"]
            old_mchirp_bbh = old_coords[inds_to_change][:,idx_mchirp]
            new_mchirp_bns = old_mchirp_bbh + self.slope_mchirp_lambdaT_bbh_bns * new_lambdaT 
            coords_change[:, idx_mchirp] = new_mchirp_bns
       
        #if self.mratio_proposal_bbh_bns:
        idx_mratio = self.params_to_inds["mass_ratio"]
        old_mratio_bbh = old_coords[inds_to_change][:,idx_mratio]            
        new_mratio_bns = old_mratio_bbh + self.slope_mratio_lambdaT_bbh_bns * new_lambdaT
        coords_change[:, idx_mratio] = new_mratio_bns
            

        # factors
        fact_g1 = 1./np.sqrt(u1) 
        w1_from_u1 = 1./u1
        fact_inverse_1 = 1./np.sqrt(w1_from_u1)
        fact_g2 = 1./np.sqrt(u2) 
        w2_from_u2 = 1./u2
        fact_inverse_2 = 1./np.sqrt(w2_from_u2)
        jacobian = (1./u1) * (1./u2)
        factors_tot = fact_inverse_1 * fact_inverse_2 *jacobian/(fact_g1 * fact_g2)

        return coords_change, np.log(factors_tot)


    def bns_to_bbh(self, old_coords, inds_to_change, random):

        """
        Proposal to go from BNS to BBH models (Eq.16 in arxiv:).
        Inverse of map from BBH to BNS.

        Args:
            old_coords (np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): coordinates to change.
            inds_to_change (np.ndarray[ntemps, nwalkers, nleaves_max]): positions of coordinate to change (where BNS is True).
            random (np.random.Generate): random generator object.
        Return:
            new coordinates, move factors

        """

        num_change = inds_to_change.sum()

        idx_lam1 = self. params_to_inds["lambda_1"]
        idx_lam2 = self.params_to_inds["lambda_2"]
        
        coords_change = old_coords[inds_to_change].copy()

        compl_pool_lam1 = old_coords[~inds_to_change][:,idx_lam1]
        compl_lambda1 = random.choice(compl_pool_lam1, size=num_change)

        compl_pool_lam2 = old_coords[~inds_to_change][:,idx_lam2]
        compl_lambda2 = random.choice(compl_pool_lam2, size=num_change)

        old_lambda1= old_coords[inds_to_change][:,idx_lam1]
        old_lambda2= old_coords[inds_to_change][:,idx_lam2]
        
        # stretch move lambda
        w1 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)
        w2 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)

        new_lambda1 = compl_lambda1 + w1*(old_lambda1 - compl_lambda1)
        new_lambda2 = compl_lambda2 + w2*(old_lambda2 - compl_lambda2)
        mask_negative_lambdas_1 = new_lambda1 < 0. # make sure we only have positive lambdas
        new_lambda1[mask_negative_lambdas_1] = old_lambda1[mask_negative_lambdas_1]
        mask_negative_lambdas_2 = new_lambda2 < 0. # make sure we only have positive lambdas
        new_lambda2[mask_negative_lambdas_2] = old_lambda2[mask_negative_lambdas_2]

        coords_change[:, idx_lam1] = new_lambda1
        coords_change[:, idx_lam2] = new_lambda2
            

        #if self.mchirp_proposal_bbh_bns or self.mratio_proposal_bbh_bns:
        eta = symmetric_mass_ratio_from_coords(old_coords[inds_to_change], self.params_to_inds)
        old_lambdaT = lambdaTilde_from_lambda1_lambda2_eta(old_lambda1, old_lambda2, eta)

        if self.mchirp_proposal_bbh_bns:
            # update mchirp
            idx_mchirp = self.params_to_inds["chirp_mass"]
            old_mchirp_bns = old_coords[inds_to_change][:,idx_mchirp]
            new_mchirp_bbh = old_mchirp_bns - self.slope_mchirp_lambdaT_bbh_bns * old_lambdaT     
            coords_change[:, idx_mchirp] = new_mchirp_bbh
 
        #if self.mratio_proposal_bbh_bns:
        # update mass ratio
        idx_mratio = self.params_to_inds["mass_ratio"]
        old_mratio_bns = old_coords[inds_to_change][:,idx_mratio]
        new_mratio_bbh = old_mratio_bns - self.slope_mratio_lambdaT_bbh_bns * old_lambdaT
        coords_change[:, idx_mratio] = new_mratio_bbh



        fact_g1 = 1./np.sqrt(w1) 
        u1_from_w1 = 1./w1
        fact_inverse_1 = 1./np.sqrt(u1_from_w1)
        fact_g2 = 1./np.sqrt(w2) 
        u2_from_w2 = 1./w2
        fact_inverse_2 = 1./np.sqrt(u2_from_w2)
        jacobian = (1./w1) * (1./w2)
        factors_tot = fact_inverse_1 * fact_inverse_2 *jacobian/(fact_g1 * fact_g2)

        return coords_change, np.log(factors_tot)

    
    def nsbh_to_bns(self, old_coords, inds_to_change, random):
       
        """
        Proposal to go from NSBH to BNS model (Eq.17 in arxiv:).
        Lambda1 and Lambda2 proposed with two separate stretch moves.
        Mchirp proposal and mass ratio now depends on lambda_tilde in both the BNS and NSBH model 
        (where the lambda_tilde in the NSBH model assumes lambda_1 = 0).
        Complementary chosen as in the other cases. 

        Args:
            old_coords (np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): coordinates to change.
            inds_to_change (np.ndarray[ntemps, nwalkers, nleaves_max]): positions of coordinate to change (where NSBH is True).
            random (np.random.Generate): random generator object.
        Return:
            new coordinates, move factors

        """

        num_change = inds_to_change.sum()
        idx_lambda1 = self.params_to_inds["lambda_1"]
        idx_lambda2 = self.params_to_inds["lambda_2"]
        
        coords_change = old_coords[inds_to_change].copy()

        compl_pool_lambda1 = old_coords[~inds_to_change][:, idx_lambda1]
        compl_lambda1 = random.choice(compl_pool_lambda1, size=num_change)
        compl_pool_lambda2 = old_coords[~inds_to_change][:, idx_lambda2]
        compl_lambda2 = random.choice(compl_pool_lambda2, size=num_change)

        old_lambda1= old_coords[inds_to_change][:,idx_lambda1]
        old_lambda2= old_coords[inds_to_change][:,idx_lambda2]

        v1 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)
        v2 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)
    
        new_lambda1 = compl_lambda1 + v1 * (old_lambda1 - compl_lambda1)
        new_lambda2 = compl_lambda2 + v2 * (old_lambda2 - compl_lambda2)
       
                        
        ### Make sure we do not have negative values of lambdas
        mask_negative_lambdas1 = new_lambda1 < 0.
        new_lambda1[mask_negative_lambdas1] = new_lambda1[mask_negative_lambdas1]
        mask_negative_lambdas2 = new_lambda2 < 0.
        new_lambda2[mask_negative_lambdas2] = new_lambda2[mask_negative_lambdas2]

        coords_change[:, idx_lambda2] = new_lambda2
        coords_change[:, idx_lambda1] = new_lambda1

        #if self.mchirp_proposal_bns_nsbh or self.mratio_proposal_bns_nsbh:
        eta = symmetric_mass_ratio_from_coords(old_coords[inds_to_change], self.params_to_inds)
        new_lambdaT_bns = lambdaTilde_from_lambda1_lambda2_eta(new_lambda1, new_lambda2, eta)
        old_lambdaT_nsbh = lambdaTilde_from_lambda1_lambda2_eta(np.zeros_like(old_lambda2), old_lambda2, eta)


        if self.mchirp_proposal_bns_nsbh:
            ### Updated chirp mass
            idx_mchirp = self.params_to_inds["chirp_mass"]
            old_mchirp_nsbh = old_coords[inds_to_change][:, idx_mchirp]
            new_mchirp_bns = old_mchirp_nsbh - self.slope_mchirp_lambdaT_bns_nsbh * (old_lambdaT_nsbh - new_lambdaT_bns)        
            coords_change[:, idx_mchirp] = new_mchirp_bns

        #if self.mratio_proposal_bns_nsbh:
        # update mass ratio
        idx_mratio = self.params_to_inds["mass_ratio"]
        old_mratio_nsbh = old_coords[inds_to_change][:,idx_mratio]
        new_mratio_bns = old_mratio_nsbh - self.slope_mratio_lambdaT_bbh_bns * (old_lambdaT_nsbh - new_lambdaT_bns)
        coords_change[:, idx_mratio] = new_mratio_bns


        ### Factors
    
        fact_g1 = 1./np.sqrt(v1) 
        w1_from_v1 = 1./v1
        fact_inverse_1 = 1./np.sqrt(w1_from_v1)
        fact_g2 = 1./np.sqrt(v2)
        w2_from_v2 = 1./v2
        fact_inverse_2 = 1./np.sqrt(w2_from_v2)
        jacobian = (1./v1) * (1./v2)
        factors_tot = fact_inverse_1 * fact_inverse_2 *jacobian/(fact_g1 * fact_g2)
         
        return coords_change, np.log(factors_tot)
       
    
    def bns_to_nsbh(self, old_coords, inds_to_change, random):

        """
        Proposal to go from BNS to NSBH models (Eq.B16 in arxiv:).
        Inverse of map from NSBH to BNS.

        Args:
            old_coords (np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): coordinates to change.
            inds_to_change (np.ndarray[ntemps, nwalkers, nleaves_max]): positions of coordinate to change (where BNS is True).
            random (np.random.Generate): random generator object.
        Return:
            new coordinates, move factors

        """
        num_change = inds_to_change.sum()
        idx_lambda1 = self.params_to_inds["lambda_1"]
        idx_lambda2 = self.params_to_inds["lambda_2"]
        
        coords_change = old_coords[inds_to_change].copy()

        compl_pool_lambda1 = old_coords[~inds_to_change][:, idx_lambda1]
        compl_lambda1 = random.choice(compl_pool_lambda1, size=num_change)
        compl_pool_lambda2 = old_coords[~inds_to_change][:, idx_lambda2]
        compl_lambda2 = random.choice(compl_pool_lambda2, size=num_change)

        old_lambda1= old_coords[inds_to_change][:,idx_lambda1]
        old_lambda2= old_coords[inds_to_change][:,idx_lambda2]
        
        w1 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)
        w2 = random.uniform(1.-self.epsilon, 1.+self.epsilon, size=num_change)

        ### Update lambdas
        new_lambda1 = compl_lambda1 + w1 * (old_lambda1 - compl_lambda1)
        new_lambda2 = compl_lambda2 + w2 * (old_lambda2 - compl_lambda2)

        mask_negative_lambdas1 = new_lambda1 < 0.
        new_lambda1[mask_negative_lambdas1] = new_lambda1[mask_negative_lambdas1]
        mask_negative_lambdas2 = new_lambda2 < 0.
        new_lambda2[mask_negative_lambdas2] = new_lambda2[mask_negative_lambdas2]

        coords_change[:, idx_lambda2] = new_lambda2
        coords_change[:, idx_lambda1] = new_lambda1

        #if self.mchirp_proposal_bns_nsbh or self.mratio_proposal_bns_nsbh:
        eta = symmetric_mass_ratio_from_coords(old_coords[inds_to_change], self.params_to_inds)
        new_lambdaT_nsbh = lambdaTilde_from_lambda1_lambda2_eta(np.zeros_like(new_lambda2), new_lambda2, eta)
        old_lambdaT_bns = lambdaTilde_from_lambda1_lambda2_eta(old_lambda1, old_lambda2, eta)

        if self.mchirp_proposal_bns_nsbh:
            ### Update chirp mass
            idx_mchirp = self.params_to_inds["chirp_mass"]
            old_mchirp_bns = old_coords[inds_to_change][:, idx_mchirp]
            new_mchirp_nsbh = old_mchirp_bns + self.slope_mchirp_lambdaT_bns_nsbh * (new_lambdaT_nsbh - old_lambdaT_bns)
            coords_change[:, idx_mchirp] = new_mchirp_nsbh
    
        #if self.mratio_proposal_bns_nsbh:
        # update mass ratio
        idx_mratio = self.params_to_inds["mass_ratio"]
        old_mratio_bns = old_coords[inds_to_change][:,idx_mratio]
        new_mratio_nsbh = old_mratio_bns + self.slope_mratio_lambdaT_bbh_bns * (new_lambdaT_nsbh - old_lambdaT_bns)
        coords_change[:, idx_mratio] = new_mratio_nsbh


        fact_g1 = 1./np.sqrt(w1) 
        v1_from_w1 = 1./w1
        fact_inverse_1 = 1./np.sqrt(v1_from_w1)
        fact_g2 = 1./np.sqrt(w2)
        v2_from_w2 = 1./w2
        fact_inverse_2 = 1./np.sqrt(v2_from_w2)
        jacobian = (1./w1) * (1./w2)
        factors_tot = fact_inverse_1 * fact_inverse_2 *jacobian/(fact_g1 * fact_g2)

        return coords_change, np.log(factors_tot)


    def _get_additional_jump_parameters(self, saved_back, last_iters):

        """
        Functon to find the slope for the linear interpolation in the chirp_mass and mass-ratio between-models proposal,
        and to decide whether the chirp-mass proposal is needed.

        Args:
            saved_back (HDF5Backend): backend containing the posterior samples from the preliminary sampling run.
            last_iters (int): number of last iterations to get samples from to compute quantities needed for the proposals.
        """

        ### Changed to take maximul likelihood among last 100 samples, can be fine-tuned or changed back to 30
        ### Define different slope parameters for each combination of sources

        if "nsbh" in self.binary_types.values() and "bbh" in self.binary_types.values():
            nsbh_branch = [key for key, val in self.binary_types.items() if val=="nsbh"][0]
            nsbh_samples, _ = saved_back.get_posterior_samples(branch=nsbh_branch,
                                                            flatten=True,
                                                            ind_start=-last_iters)
            logl_nsbh, _ = saved_back.get_log_likelihood(branch=nsbh_branch,
                                                     flatten=True,
                                                     ind_start=-last_iters)
            
            bbh_branch = [key for key, val in self.binary_types.items() if val=="bbh"][0]
            bbh_samples, _ = saved_back.get_posterior_samples(branch=bbh_branch,
                                                           flatten=True,
                                                           ind_start=-last_iters)
            logl_bbh, _ = saved_back.get_log_likelihood(branch=bbh_branch,
                                                     flatten=True,
                                                     ind_start=-last_iters)
       
            mc_cm_nsbh, mc_cm_bbh, lam2_cm_nsbh, _, _, _, q_cm_nsbh, q_cm_bbh = self._posterior_center_of_mass(nsbh_samples["chirp_mass"], bbh_samples["chirp_mass"], nsbh_samples["lambda_2"], None, None, None, nsbh_samples["mass_ratio"], bbh_samples["mass_ratio"], logl_nsbh, logl_bbh)
 
            eta_nsbh = symmetric_mass_ratio_from_mass_ratio(q_cm_nsbh)
            lambdaT_nsbh = lambdaTilde_from_lambda1_lambda2_eta(np.zeros_like(lam2_cm_nsbh), lam2_cm_nsbh, eta_nsbh)
            
            self.slope_mchirp_lambdaT_nsbh_bbh = (mc_cm_nsbh - mc_cm_bbh) / lambdaT_nsbh
            self.slope_mratio_lambdaT_nsbh_bbh = (q_cm_nsbh - q_cm_bbh) / lambdaT_nsbh
 
            self.mchirp_proposal_nsbh_bbh = self.is_param_proposal_needed(nsbh_samples["chirp_mass"],bbh_samples["chirp_mass"],mc_cm_nsbh,mc_cm_bbh) 
            #self.mratio_proposal_nsbh_bbh = self.is_param_proposal_needed(nsbh_samples["mass_ratio"],bbh_samples["mass_ratio"],q_cm_nsbh,q_cm_bbh) 

       
        if 'bbh' in self.binary_types.values() and "bns" in self.binary_types.values():

            bns_branch = [key for key, val in self.binary_types.items() if val=="bns"][0]
            bns_samples, _ = saved_back.get_posterior_samples(branch=bns_branch,
                                                            flatten=True,
                                                            ind_start=-last_iters)
            logl_bns, _ = saved_back.get_log_likelihood(branch=bns_branch,
                                                     flatten=True,
                                                     ind_start=-last_iters)

            bbh_branch = [key for key, val in self.binary_types.items() if val=="bbh"][0]
            bbh_samples, _ = saved_back.get_posterior_samples(branch=bbh_branch,
                                                           flatten=True,
                                                           ind_start=-last_iters)
            logl_bbh, _ = saved_back.get_log_likelihood(branch=bbh_branch,
                                                     flatten=True,
                                                     ind_start=-last_iters)

            mc_cm_bns, mc_cm_bbh, lam2_cm_bns, _, lam1_cm_bns, _, q_cm_bns, q_cm_bbh = self._posterior_center_of_mass(bns_samples["chirp_mass"], bbh_samples["chirp_mass"], bns_samples["lambda_2"], None, bns_samples["lambda_1"], None, bns_samples["mass_ratio"], bbh_samples["mass_ratio"], logl_bns, logl_bbh)

            eta_bns = symmetric_mass_ratio_from_mass_ratio(q_cm_bns)
            lambdaT_bns = lambdaTilde_from_lambda1_lambda2_eta(lam1_cm_bns, lam2_cm_bns, eta_bns)

            self.slope_mchirp_lambdaT_bbh_bns = (mc_cm_bns - mc_cm_bbh) / lambdaT_bns
            self.slope_mratio_lambdaT_bbh_bns = (q_cm_bns - q_cm_bbh) / lambdaT_bns

            self.mchirp_proposal_bbh_bns = self.is_param_proposal_needed(bbh_samples["chirp_mass"],bns_samples["chirp_mass"],mc_cm_bbh,mc_cm_bns)
            #self.mratio_proposal_bbh_bns = self.is_param_proposal_needed(bbh_samples["mass_ratio"],bns_samples["mass_ratio"],q_cm_bbh,q_cm_bns)

 
        if "bns" in self.binary_types.values() and "nsbh" in self.binary_types.values():
            nsbh_branch = [key for key, val in self.binary_types.items() if val=="nsbh"][0]
            nsbh_samples, _ = saved_back.get_posterior_samples(branch=nsbh_branch,
                                                               flatten=True,
                                                               ind_start=-last_iters)
            logl_nsbh, _ = saved_back.get_log_likelihood(branch=nsbh_branch,
                                                         flatten=True,
                                                         ind_start=-last_iters)
            
            bns_branch = [key for key, val in self.binary_types.items() if val=="bns"][0]
            bns_samples, _ = saved_back.get_posterior_samples(branch=bns_branch,
                                                              flatten=True,
                                                              ind_start=-last_iters)
            logl_bns, _ = saved_back.get_log_likelihood(branch=bns_branch,
                                                        flatten=True,
                                                        ind_start=-last_iters)
        
            mc_cm_bns, mc_cm_nsbh, lam2_cm_bns, lam2_cm_nsbh, lam1_cm_bns, _, q_cm_bns, q_cm_nsbh = self._posterior_center_of_mass(bns_samples["chirp_mass"], nsbh_samples["chirp_mass"], bns_samples["lambda_2"], nsbh_samples["lambda_2"], bns_samples["lambda_1"], None, bns_samples["mass_ratio"], nsbh_samples["mass_ratio"], logl_bns, logl_nsbh)

            eta_nsbh = symmetric_mass_ratio_from_mass_ratio(q_cm_nsbh)
            lambdaT_nsbh = lambdaTilde_from_lambda1_lambda2_eta(np.zeros_like(lam2_cm_nsbh), lam2_cm_nsbh, eta_nsbh)
            
            eta_bns = symmetric_mass_ratio_from_mass_ratio(q_cm_bns)
            lambdaT_bns = lambdaTilde_from_lambda1_lambda2_eta(lam1_cm_bns, lam2_cm_bns, eta_bns)
            
            self.slope_mchirp_lambdaT_bns_nsbh = (mc_cm_bns - mc_cm_nsbh) / (lambdaT_bns - lambdaT_nsbh)
            self.slope_mratio_lambdaT_bns_nsbh = (q_cm_bns - q_cm_nsbh) / (lambdaT_bns - lambdaT_nsbh)
            self.mchirp_proposal_bns_nsbh = self.is_param_proposal_needed(bns_samples["chirp_mass"],nsbh_samples["chirp_mass"],mc_cm_bns,mc_cm_nsbh)
            #self.mratio_proposal_bns_nsbh = self.is_param_proposal_needed(bns_samples["mass_ratio"],nsbh_samples["mass_ratio"],q_cm_bns,q_cm_nsbh)


    def _posterior_center_of_mass(self,mc_mod1, mc_mod2, lam2_mod1, lam2_mod2, lam1_mod1, lam1_mod2, q_mod1, q_mod2, loglik_mod1, loglik_mod2):

        """
        Computes the chirp mass, mass ratio, and mass-weighted tidal deformability corresponding
        to the likelihood center-of-mass for two models mod1 and mod2.
        TODO: not sure why I did it for two models at the same time

        Args:
            mc_mod1 (list): chirp mass samples for model1.
            mc_mod2 (list): chirp mass samples for model2.
            lam2_mod1 (list): lambda_2 samples for model1.
            lam2_mod2 (list): lambda_2 samples for model2.
            lam1_mod1 (list): lambda_1 samples for model1.
            lam1_mod2 (list): lambda_1 samples for model2.
            q_mod1 (list): mass ratio samples for model1.
            q_mod2 (list): mass ratio samples for model2.
            loglik_mod1 (list): samples log likelihood for model1.
            loglik_mod2 (list): samples log likelihood for model2.

        Returns:
            mc_com_mod1 (float): likelihood center-of-mass chirp mass for model1
            mc_com_mod2 (float): likelihood center-of-mass chirp mass for model2
            lambda2_com_mod1 (float): likelihood center-of-mass lambda_2 for model1
            lambda2_com_mod2 (float): likelihood center-of-mass lambda_2 for model2
            lambda1_com_mod1 (float): likelihood center-of-mass lambda_1 for model1
            lambda1_com_mod2 (float): likelihood center-of-mass lambda_1 for model2
            q_com_mod1 (float): likelihood center-of-mass mass ratio for model1
            q_com_mod2 (float): likelihood center-of-mass mass ratio for model2
        """

        mc_com_mod1 = 0.
        mc_com_mod2 = 0.
        q_com_mod1 = 0
        q_com_mod2 = 0
        lambda2_com_mod1 = 0.
        lambda2_com_mod2 = 0.
        lambda1_com_mod1 = 0.
        lambda1_com_mod2 = 0.

        lik_mod1_tot = 0.
        lik_mod2_tot = 0.

        for ii in range(0,len(loglik_mod1)):

            mc_com_mod1+=mc_mod1[ii] * loglik_mod1[ii]
            q_com_mod1 += q_mod1[ii] * loglik_mod1[ii]
            lik_mod1_tot+=loglik_mod1[ii]

            if lam2_mod1 is not None:
                lambda2_com_mod1+=lam2_mod1[ii]*loglik_mod1[ii]
            if lam1_mod1 is not None:
                lambda1_com_mod1+=lam1_mod1[ii]*loglik_mod1[ii]

        for kk in range(0,len(loglik_mod2)):

            mc_com_mod2+=mc_mod2[kk] * loglik_mod2[kk]
            q_com_mod2 += q_mod2[kk] * loglik_mod2[kk]
            lik_mod2_tot+=loglik_mod2[kk]
            
            if lam2_mod2 is not None:
                lambda2_com_mod2+=lam2_mod2[kk]*loglik_mod2[kk]
            if lam1_mod2 is not None:
                lambda1_com_mod2+=lam1_mod2[kk]*loglik_mod2[kk]


        mc_com_mod1/=lik_mod1_tot
        q_com_mod1/=lik_mod1_tot

        if lam2_mod1 is not None:
            lambda2_com_mod1/=lik_mod1_tot
        if lam1_mod1 is not None:
            lambda1_com_mod1/=lik_mod1_tot

        mc_com_mod2/=lik_mod2_tot
        q_com_mod2/=lik_mod2_tot

        if lam2_mod2 is not None:
            lambda2_com_mod2/=lik_mod2_tot
        if lam1_mod2 is not None:
            lambda1_com_mod2/=lik_mod2_tot

        return mc_com_mod1, mc_com_mod2, lambda2_com_mod1, lambda2_com_mod2, lambda1_com_mod1, lambda1_com_mod2, q_com_mod1, q_com_mod2


    
    def is_param_proposal_needed(self, param_mod1_samples,param_mod2_samples,param_com_mod1, param_com_mod2):

        """
        Decide whether we need the chirp mass proposal or no. For now we just check if the mchirp posterior CoM is within 1sigma of the other posterior.
        Args:
            param_mod1_samples (list): samples of the parameter for model1
            param_mod2_samples (list): samples of the parameter for model2
            param_com_mod1 (float): likelihood center-of-mass for the parameter in model1
            param_com_mod2 (float): likelihood center-of-mass for the parameter in model2
        Returns:
            param_proposal_flag (bool)
        """

        if param_com_mod1 > np.quantile(param_mod2_samples,0.16) and param_com_mod1 < np.quantile(param_mod2_samples,0.84) and param_com_mod2 > np.quantile(param_mod1_samples,0.16) and param_com_mod2 < np.quantile(param_mod1_samples,0.84):

            param_proposal_flag = False
        
        else:
            param_proposal_flag = True    

        return param_proposal_flag



