# -*- coding: utf-8 -*-

"""
Module to set up the class for the reversible jump move.
"""

from multiprocessing.sharedctypes import Value
import numpy as np
from copy import deepcopy
from ..state import State
from .move import Move

__all__ = ["ReversibleJumpMove"]


class ReversibleJumpMove(Move):
    """
    An abstract reversible jump move. Adapted to the specific case of GW CBC signals with different sources models.

    Args:
        nleaves_max (dict): Maximum number(s) of leaves for each model.
            Keys are ``branch_names`` and values are ``nleaves_max`` for each branch.
            This is a keyword argument, nut it is required.
        nleaves_min (dict): Minimum number(s) of leaves for each model.
            Keys are ``branch_names`` and values are ``nleaves_min`` for each branch.
            This is a keyword argument, nut it is required.
        tune (bool, optional): If True, tune proposal. (Default: ``False``)
        fix_change (int or None, optional): Fix the change in the number of leaves. Make them all
            add a leaf or remove a leaf. This can be useful for some search functions. Options
            are ``+1`` or ``-1``. (default: ``None``)

    """

    def __init__(
        self,
        nleaves_max=None,
        nleaves_min=None,
        tune=False,
        fix_change=None,
        **kwargs
    ):
        # super(ReversibleJumpMove, self).__init__(**kwargs)
        Move.__init__(self, is_rj=True, **kwargs)

        # store info
        self.nleaves_max = nleaves_max
        self.nleaves_min = nleaves_min
        self.tune = tune
        self.fix_change = fix_change
        if self.fix_change not in [None, +1, -1]:
            raise ValueError("fix_change must be None, +1, or -1.")        

    def setup(self, branches_coords):
        """Any setup for the proposal.

        Args:
            branches_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.

        """

    def get_proposal(
        self, all_coords, all_inds, nleaves_min_all, nleaves_max_all, random, **kwargs
    ):
        """Function to propose new points, needs to be overwritten by the specific move one.

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
            **kwargs (ignored): For modularity.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        Raises:
            NotImplementedError: If this proposal is not implemented by a subclass.

        """
        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def get_model_change_proposal(self, inds, random, nleaves_min, nleaves_max):
        """Helper function for changing the model count by 1.

        This helper function works with nested models where you want to add or remove
        one leaf at a time.

        This was implemented in eryn and is currently not used by t-roo.

        Args:
            inds (np.ndarray): ``inds`` values for this specific branch with shape
                ``(ntemps, nwalkers, nleaves_max)``.
            random (object): Current random state of the sampler.
            nleaves_min (int): Minimum allowable leaf count for this branch.
            nleaves_max (int): Maximum allowable leaf count for this branch.

        Returns:
            dict: Keys are ``"+1"`` and ``"-1"``. Values are indexing information.
                    ``"+1"`` and ``"-1"`` indicate if a source is being added or removed, respectively.
                    The indexing information is a 2D array with shape ``(number changing, 3)``.
                    The length 3 is the index into each of the ``(ntemps, nwalkers, nleaves_max)``.

        """

        raise NotImplementedError

    def propose(self, model, state):
        """Use the specific rj move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        
        # Run any move-specific setup.
        self.setup(state.branches)

        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        all_branch_names = list(state.branches.keys())
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # gibbs sampling is only over branches so pick out that info
            coords_propose_in = {
                key: state.branches_coords[key] for key in branch_names_run
            }
            
            inds_propose_in = {
                key: state.branches_inds[key] for key in branch_names_run
            }

            if len(list(coords_propose_in.keys())) == 0:
                raise ValueError(
                    "Right now, no models are getting a reversible jump proposal. Check nleaves_min and nleaves_max or do not use rj proposal."
                )

            # get min and max leaf information
            nleaves_max_all = {brn: self.nleaves_max[brn] for brn in branch_names_run}
            nleaves_min_all = {brn: self.nleaves_min[brn] for brn in branch_names_run}

            self.current_model = model
            self.current_state = state
            # propose new sources and coordinates
            q, new_inds, factors = self.get_proposal(
                coords_propose_in,            
                inds_propose_in,
                nleaves_min_all,
                nleaves_max_all,
                model.random,
            )

            ### Keep this, also if not used, for the ordering part
            branches_supps_new = None
           
            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # put back any branches that were left out from Gibbs split
            for name, branch in state.branches.items():
                if name not in q:
                    q[name] = state.branches[name].coords[:].copy()
                if name not in new_inds:
                    new_inds[name] = state.branches[name].inds[:].copy()

            # fix any ordering issues
            q, new_inds, branches_supps_new = self.ensure_ordering(
                list(state.branches.keys()), q, new_inds, branches_supps_new
            )

            # setup supplimental information

            if state.supplimental is not None:
                new_supps = deepcopy(state.supplimental)

            else:
                new_supps = None

            # for_transfer information can be taken directly from custom proposal

            # supp info

            if hasattr(self, "mt_supps"):
                # logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
                new_supps = self.mt_supps

            if hasattr(self, "mt_branch_supps"):
                # logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
                new_branch_supps = self.mt_branch_supps

            # logp and logl
           
            # Compute prior of the proposed position
            if hasattr(self, "mt_lp"):
                logp = self.mt_lp.reshape(ntemps, nwalkers)

            else:
                logp = model.compute_log_prior_fn(q, inds=new_inds)
          
            if hasattr(self, "mt_ll"):
                logl = self.mt_ll.reshape(ntemps, nwalkers)

            else:
                # Compute the ln like of the proposed position.
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=new_inds,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=branches_supps_new,
                )

            # posterior and previous info

            logP = self.compute_log_posterior(logl, logp)
         
            prev_logl = state.log_like

            prev_logp = state.log_prior
           
            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)
           
            # acceptance fraction
            lnpdiff = factors + logP - prev_logP

            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

                       
            # update with new state
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=None,
                inds=new_inds,
                )
            state = self.update(state, new_state, accepted)


        if self.temperature_control is not None and not self.prevent_swaps:
            state = self.temperature_control.temper_comps(state, adapt=False)

        # add to move-specific accepted information
        self.accepted += accepted
        self.num_proposals += 1

        return state, accepted
