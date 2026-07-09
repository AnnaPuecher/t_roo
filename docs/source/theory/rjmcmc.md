# Reversible Jump MCMC

Bayesian inference is commonly employed in gravitational-wave (GW) data analysis not only to estimate the posterior probability of the source parameters, but also to compare different models and hypotheses.
Reversible jump Markov chain Monte Carlo (RJMCMC) {cite}`rjmcmc` is a particular instance of MCMC samplers that allows one to compare also models with different dimensionalities. In the following we summarize the key features of RJMCMC samplers; a more detailed description can be found in Appenix A of the paper:

When we analyze some data $d$ with a model $\mathcal{H}$ with parameters $\vec{\theta}_k$, the parameters' posterior probabilities are obtained via the Bayes theorem
$$
p(\vec{\theta}_k | d, \mathcal{H}) = \frac{\mathcal{L}(d|\vec{\theta}_k, \mathcal{H} p(\vec{\theta}_k | \mathcal{H}))}{p(d | \mathcal{H})},
$$
where $\mathcal{L} (d | \vec{\theta}_k, \mathcal{H})$ is the *likelihood*, $p(\vec{\theta}_k | \mathxal{H})$ the prior, and $p(d|\matchal{H})$ the evidence.

To compare two models $\mathca{H}_A$ and $\mathcal{H}_B$, to see which one decribes the data best, one usually computes the *odds ratio* 
$$
\mathcal{O}_A^B = \frac{p(d|\mathcal{H}_B) p(\mathcal{H}_B)}{p(d|\mathcal{H}_A p(\mathcal{H}_A))},
$$ 
where the second term denotes the ratio of models prior and is usually set to 1 if we do not have any a-priori preference or additional information about the models.

Therefore, model selection analyses require computing the evidences (usually with nested samplers) for each model and comparing them. With an alternative approach, instead, one can sample directly over the model itself. Suppose we have a set of models {$M_k$} that we want to compare. RJMCMC samples directly the joint posterior
$$
p(\vec{\theta}_k, k | d),
$$
where $k$ is a label identifying the model and $\vec{\theta}_k$ the parameters specific to that model.	



```{bibliography}
