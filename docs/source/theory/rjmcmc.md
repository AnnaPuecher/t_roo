# Reversible Jump MCMC

Bayesian inference is commonly employed in gravitational-wave (GW) data analysis not only to estimate the posterior probability of the source parameters, but also to compare different models and hypotheses.
Reversible jump Markov chain Monte Carlo (RJMCMC) {cite}`rjmcmc` is a particular instance of MCMC samplers that allows one to compare also models with different dimensionalities. In the following we summarize the key features of RJMCMC samplers; a more detailed description can be found in Appenix A of the paper:

When we analyze some data $d$ with a model $\mathcal{H}$ with parameters $\vec{\theta}_k$, the parameters' posterior probabilities are obtained via the Bayes theorem
```{math}
p(\vec{\theta}_k | d, \mathcal{H}) = \frac{\mathcal{L}(d|\vec{\theta}_k, \mathcal{H} p(\vec{\theta}_k | \mathcal{H}))}{p(d | \mathcal{H})},
```
where $\mathcal{L} (d | \vec{\theta}_k, \mathcal{H})$ is the *likelihood*, $p(\vec{\theta}_k | \mathcal{H})$ the prior, and $p(d|\mathcal{H})$ the evidence.

To compare two models $\mathcal{H}_A$ and $\mathcal{H}_B$, to see which one decribes the data best, one usually computes the *odds ratio* 
```{math}
\mathcal{O}_A^B = \frac{p(d|\mathcal{H}_B) p(\mathcal{H}_B)}{p(d|\mathcal{H}_A p(\mathcal{H}_A))},
```
where the second term denotes the ratio of models prior and is usually set to 1 if we do not have any a-priori preference or additional information about the models.

Therefore, model selection analyses require computing the evidences (usually with nested samplers) for each model and comparing them. With an alternative approach, instead, one can sample directly over the model itself. Suppose we have a set of models {$M_k$} that we want to compare. RJMCMC samples directly the joint posterior
```{math}
p(\vec{\theta}_k, k | d),
```
where $k$ is a label identifying the model and $\vec{\theta}_k$ the parameters specific to that model.	

The idea is that the more (less) favored a model is, the more (less) time the sampler spends in that model (similary to how it usually spends more time in regions of the parameter space with larger likelihood). Therefore, at the end of a RJMCMC analysis, the probability of each model $k$ is given by
```{math}
p(k|d) = \frac{n_k}{\sum_k n_k},
```
where $n_k$ is the number of sampels in that specific model. Consequently, the odds ratio between two models $\mathcal{H}_A$ with $n_A$ samples and $\mathcal{H}_B$ with $n_B$ samples reads
```{math}
\mathcal{O}_A^B = \frac{n_B}{n_A}.
```

### Metropolis-Hastings algorithm

The Metropolis-Hastings algorithm for MCMC samplers includes two steps. Assume that the chain with limiting distribution $\pi$ (the limiting distribution is what we sample) is in a point $x_i$ and wants to go to the next point $x_{i+1}$; we need to

- Propose the new point from the proposal distribution $q(x_{i+1}|x_i)$ (the new points depends only on the currrent one since in Markov chains each state depends on the previous one and only on that)
- Decide whether to accept the new point or not. Fot this purpose, we compute an **acceptance probability** $\alpha(x_i, x_{i+1})$, the draw a random number $r \in [0,1]$
	- if $\alpha \ge r$: the new point is accepted
	- i $\alpha(x_i, x_{i+1}) < r$: the new point is rejected and the chain stays in $x_i$


The generally employed expression of the acceptance probability is
```{math}
\alpha(x_i, x_{i+1}) = \min \left\{ 1, \frac{\pi(x_{i+1} q(x_i | x_{i+1}))}{\pi(x_{i+1}) q(x_{i}|x_{i+1}) } \right}
```



```{bibliography}
