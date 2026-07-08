# -*- coding: utf-8 -*-

from abc import ABC

import numpy as np

from ..logging import logger


class Stopping(ABC, object):
    """Base class for stopping.
    
    Stopping checks are only performed every ``thin_by`` iterations.
    
    """

    @classmethod
    def __call__(self, iter, last_sample, sampler):
        """Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).

        Returns:
            bool: Value of ``stop``. If ``True``, stop sampling.
            
        """
        raise NotImplementedError


class StoppingCriterionPrelSampling(Stopping):
    """Stopping function based on a convergence to a maximunm Likelihood.

    Stopping checks are only performed every ``thin_by`` iterations.
    Therefore, the iterations of stopping checks are really every
    ``sampler iterations * thin_by``.  

    All arguments are stored as attributes.

    Args:
        n_walkers (int): Number of MCMC walkers in the sampler
        stop_crit (str): which stopping criterion to use, "steps" for a fixed number of steps, "lik" (default) to automatically check the likelihood convergence.
        n_steps (int, optionale): number of step for the preliminary sampling, to provide only if the stopping criterion is "steps".
        n_iters (int, optional): Number of iteratations a walkers' maximum log-likelihood has to stay the same. Defaults to 60.
        tolerance (float, optional): tolerance with which the maximum log-likelihood can be exceeded.
        start_iteration (int, optional): Iteration of sampler to start checking to stop. (default: 100)
        verbose (bool, optional): If ``True``, print information. (default: ``False``)
    
    """

    def __init__(self,
                 n_walkers: int,
                 stop_crit: str = 'lik',
                 n_iters: int = 60, 
                 tolerance: float = 0.1, 
                 start_iteration: int = 100,
                 n_steps: int = 2000,
                 verbose: bool = False):

        self.n_iters = n_iters
        self.n_walkers = n_walkers

        self.stop_crit = stop_crit
        self.n_steps = n_steps

        self.tolerance = tolerance
        self.verbose = verbose
        self.start_iteration = start_iteration

        self.iters_consecutive = np.zeros(shape=(n_walkers))
        self.past_best_likelihood = np.full(shape=(n_walkers), fill_value=-np.inf)

    def __call__(self, iter, sample, sampler) -> bool:
        """Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).

        Returns:
            bool: Value of ``stop``. If ``True``, stop sampling.
            
        """

       
        if self.stop_crit == 'steps':
               
            if self.start_iteration > self.n_steps:
                raise ValueError(f"The number of steps before starting to check the convergence criterion ({self.start_iteration}) is larger than the number of steps set for the preliminary sampling ({self.n_steps})")

            if iter == (self.n_steps - 1):
                return True
            else:
                if self.verbose and (iter % 10 ==0):
                    logger.info(f"Preliminary sampling | step {iter}/{self.n_steps}")
                return False

        if self.stop_crit == 'lik':

            if iter < self.start_iteration:
                if self.verbose and (iter % 10 ==0):
                    logger.info(f"Preliminary sampling | step {iter} | will start checking convergence at step {self.start_iteration}")
                return False
   
            # get best Likelihood so far
            likelihood = sampler.backend_prel.get_log_like(discard=self.start_iteration)
            best_likelihood= np.max(likelihood[:, 0, :], axis=0)

            mask = (best_likelihood - self.past_best_likelihood) > self.tolerance

            self.iters_consecutive[~mask] += 1
            self.iters_consecutive[mask] = 0
            self.past_best_likelihood[mask] = best_likelihood[mask]

            converged_walkers = np.sum(self.iters_consecutive > self.n_iters)

            # print information
            if self.verbose and (iter % 50 ==0):
                logger.info(f"Preliminary sampling | step {iter} | converged walkers: {converged_walkers}")

            if converged_walkers > 0.8*self.n_walkers:
                return True

            else:
                return False


class StoppingCriterionRJMCMC(Stopping):
    
    '''
    Different stopping criteria implemented
    - number of steps: the run is stoppes after the desired number of steps
    - walkers mean difference: stop when the mean across walkers of the cumulative probability for each model stays constant for 200 steps
    - standard deviation: stop when the standard deviation among walkers for each model's cumulative number of samples corresponds to the variance expected 
                          for a binomial distribution (multimodal in the case of more than two models) over x steps, with x arbitrary set to 500; the probbality 
                          is computed as the mean among walkers in the last step within a tolerance of 30% (quite large but otherwise it takes too long to converge)

    Args
        n_walkers (int): number of walkers
        models (list,optional): list of models involved in the rjmcmc sampling. Not needed if the stopping criterion is "steps".
        stop_crit (str,optional): determined the stopping criterion to use. "steps" for the number of steps, "mean_walkers" for the walkers mean difference, 
                                  "multinomial" (default) for the standard deviation method
        n_iters (int, optional): over how many iterations the cumulative model probabilities for each walker are computed (default 1000).
        start_iteration (int, optional): at which step start checking for convergence (default 1000).
        check_every (int, optional): check the convergence every {} steps (default 200).
        n_steps (int, optional): after how many steps the run is stopped (default 10000), must be provided only if criterion is "steps".
        verbose (bool): if True prints rjmcmc sampling progress (default False).
    '''


    def __init__(self,
                 n_walkers: int,
                 models: list,
                 stop_crit: str = 'multinomial',
                 n_iters: int = 1000,
                 start_iteration: int = 1000,
                 check_every: int = 200,
                 n_steps: int = 10000,
                 verbose: bool = False):
        
        self.stop_crit = stop_crit
        self.n_walkers = n_walkers
        self.n_iters = n_iters
        self.models = models
        self.start_iteration = start_iteration
        self.check_every = check_every
        self.n_steps = n_steps
        self.verbose = verbose

    def __call__(self, iter, sample, sampler) -> bool:
        if self.verbose and (iter % 50 ==0):
            logger.info(f"RJMCM sampling | step {iter}")


        if self.stop_crit == 'steps':

            if self.start_iteration > self.n_steps:
                raise ValueError(f"The number of steps before starting to check the convergence criterion ({self.start_iteration}) is larger than the number of steps set for the rjmcmc sampling ({self.n_steps})")
            if iter == (self.n_steps - 1):
                return True 
        

        if self.stop_crit == 'mean_walkers':
            
            if not iter >= self.start_iteration:
                return False
            else:
               
                if (iter % self.check_every == 0):
                   
                    converged_models_count = 0
                    first_it = iter - self.n_iters
                   
                    for model_name in self.models[:-1]:
                        model_prob_mean = []

                        for it in range(first_it, iter):
                            
                            _,inds_samples = sampler.backend.get_posterior_samples(branch=model_name, ind_start=first_it-1,ind_end=it, flatten=False)
                            model_prob_cumulative = []
    
                            for nw in range(self.n_walkers):          
                                inds_samples_w = inds_samples[:,nw]
                                model_prob_cumulative.append((np.sum(inds_samples_w, axis=0))/(it-first_it+1))                                
    
                            model_prob_mean.append(np.mean(model_prob_cumulative))
                            
                           
                        diff_last = []

                        for jj in range(0,len(model_prob_mean)-1):
                            diff_last.append(model_prob_mean[-1] - model_prob_mean[jj])
    
                        if self.consecutive_zeros(np.array(diff_last),nzeros=200):
                            converged_models_count+=1    
                            continue
                        else:
                            break
                    
                    if converged_models_count == len(self.models)-1:
                        return True
                    else:
                        return False

                else:
                    return False

        if self.stop_crit == 'multinomial':

            if not iter >= self.start_iteration:
                return False
            else:

                if (iter % self.check_every == 0):

                    converged_models_count = 0
                    first_it = iter-self.n_iters

                    for model_name in self.models[:-1]:
                        ### For loop over models, continue only if number of samples for the previous model is checked
                        ### We ignore the last model because it should be complementary to the other ones, save some computational time
                            
                        model_prob_mean = []
                        model_prob_std = []

                        for it in range(first_it,iter):

                            _, inds_samples = sampler.backend.get_posterior_samples(branch=model_name, ind_start=first_it-1,ind_end=it, flatten=False) 
                        
                            model_prob_cumulative = []

                            for nw in range(0, self.n_walkers):
                                
                                inds_samples_w = inds_samples[:,nw]
                                model_prob_cumulative.append(np.sum(inds_samples_w, axis=0)/(it-first_it+1))
                             

                            
                            model_prob_mean.append(np.mean(model_prob_cumulative))
                            model_prob_std.append(np.std(model_prob_cumulative))

                        sigma_binomial = np.sqrt(self.n_iters * model_prob_mean[-1]*(1. - model_prob_mean[-1]))/self.n_iters
                        convergence = model_prob_std <= sigma_binomial * 1.3

                        if any(convergence):
                            converged_models_count+=1
                            continue
                        else:
                            break
                    
                    if converged_models_count == len(self.models)-1:
                        return True
                    else:
                        return False

                else:
                    return False    
                        


    def consecutive_zeros(self,data,nzeros=100,tol=1e-6):
        
        is_zero = np.abs(data) <= tol
        run_lengths = np.convolve(is_zero, np.ones(nzeros, dtype=int), mode='valid')
        return np.any(run_lengths == nzeros)

