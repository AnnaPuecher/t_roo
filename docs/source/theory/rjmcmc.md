# Reversible Jump MCMC

Bayesian inference is commonly employed in gravitational-wave (GW) data analysis not only to estimate the posterior probability of the source parameters, but also to compare different models and hypotheses.
Reversible jump Markov chain Monte Carlo (RJMCMC) []{cite}`rjmcmc` is a particular instance of MCMC samplers that allows one to compare also models with different dimensionalities. In the following we summarize the key features of RJMCMC samplers; a more detailed description can be found in Appenix A of the apper:

 

to sample directly the joint posterior `$p(\vec{\theta}_{k}, k | d)$`, where $k$ is a lable denoting the model and $\vec{\theta}_{k}$ the parameters for that specific model.
In the following we summarize the working mechanisms of RJMCMC samplers; a more detailed description can be found in Appendix A of the paper:

```{bibliography}
