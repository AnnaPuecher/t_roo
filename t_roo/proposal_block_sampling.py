import numpy as np

"""
Module for block sampling proposal. 
Proposal cycle similar to the one in bilby mcmc.
"""

##############################
# TODO: add random to arguments
###############################


class ProposalCycle(object):

    """
    Proposal cycle to allow block sampling for within model moves. 
    Follow the structure of bilby-mcmc cycle, although for now 
    we always use the strecth move (apart from the pseudo-lambdas that have random walk by default), 
    but just for different groups of parameters

    proposal list = list of group of parameters for block sampling
        (No dictionary here so we need to give the 'position' of the from the pseudo-lambdas that have random walk from the pseudo-lambdas that have random walk by defaultt)
    proposals_weights = weight of each proposal in the cycle; does not need
        to be normalized since it gets normalized after
    """

    def __init__(self, proposal_list_names, proposals_weights):

        self.proposal_list_names = proposal_list_names
        self.weights = proposals_weights
        self.normalized_weights = [w / sum(self.weights) for w in self.weights]
        self._position = 0 ####Initialize cycle
        self.weighted_proposal_list = None

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        ### To continue cycle after number of steps > number of proposals in the cycle
        self._position = np.mod(position, self.nproposals)
   
    def synchronize_with_sampler(self,ensemble_sampler):

        self.params_to_inds = ensemble_sampler.params_to_inds
        self.inds_to_params = ensemble_sampler.inds_to_params

        self._translate_proposal_params()

    def get_cycle_proposal(self, random):
        
        if self.weighted_proposal_list is None:
           
            prop_array = np.array(self.proposal_list, dtype=object) 
            ### Should do this only at the first step
            ### In this way we can use the random state defined in the ensenble
            ### and used everywhere

            self.weighted_proposal_list = [random.choice(prop_array, p=self.normalized_weights)
                                        for _ in range(10 * int(1 / min(self.normalized_weights)))
                                ]
            self.nproposals = len(self.weighted_proposal_list)

        prop = self.weighted_proposal_list[self._position]
        self.position += 1
       
        return prop


    def _translate_proposal_params(self):

        proposal_inds_list = []

        for block_idx in range(0,len(self.proposal_list_names)):
            
            if self.proposal_list_names[block_idx] == 'all':
                proposal_inds_list.append('all')
            else:
                block = []
                for pname in self.proposal_list_names[block_idx]:
                    block.append(self.params_to_inds[pname])
                proposal_inds_list.append(block)

        self.proposal_list = proposal_inds_list
        






