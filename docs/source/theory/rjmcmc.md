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

This yields a potentially significant computational gain, especially when comparing many models or analyzing informative data, which make it easier for t-roo (or RJMCMC samplers) to understand which are the favored models.

In only one run, t-roo returns the probability for all the models considered and the posteriors for the parameters under the favored model. The posteriors for highly disfavored models appear very scattered, because, as expected, the walkers spend very little time in models that the sampler deems disfavored, i.e., not a good description of the data.

##### Metropolis-Hastings algorithm

The Metropolis-Hastings algorithm for MCMC samplers includes two steps. Assume that the chain with limiting distribution $\pi$ (the limiting distribution is the sampling target) is in a point $x_i$ and wants to go to the next point $x_{i+1}$; we need to

- Propose the new point from the proposal distribution $q(x_{i+1}|x_i)$ (the new points depends only on the currrent one since in Markov chains each state depends on the previous one and only on that)
- Decide whether to accept the new point or not. For this purpose, we compute an **acceptance probability** $\alpha(x_i, x_{i+1})$, then draw a random number $r \in [0,1]$
	- if $\alpha(x_i, x_{i+1}) \ge r$: the new point is accepted
	- if $\alpha(x_i, x_{i+1}) < r$: the new point is rejected and the chain stays in $x_i$


The generally employed expression of the acceptance probability is
```{math}
:label: eq:acc_mh
\alpha(x_i, x_{i+1}) = \min \left\{ 1, \frac{\pi(x_{i+1} q(x_i | x_{i+1}))}{\pi(x_{i+1}) q(x_{i}|x_{i+1}) } \right\},
```
and is derived to ensure that the **detailed balance** condition is fulfilled. Detailed balance basically means that the probability of being in one state $x$ and going to a state $x'$ must be the same as being in $x'$ and going back to $x$. This condition is the baisis of MCMC algorithms because it ensures that the limiting distribution of the Markov chains is stationary.


### The trans-dimensional case


Assume we want to move from a state $x$, described by a model $k$ with parameters $\vec{\theta}_k$ with dimension $n_k$, to a state $x'$ with a model $k'$ and $n_{k'}$-dimensional parameters $\vec{\theta}_{k'}$, where we could have $n_k \neq n_{k'}$.

The idea the algorithm proposed by Green is to include some so-called **auxiliary variables**, the random variables $\vec{u}$ in the space of set $x$ and $\vec{u}'$ in $x'$. $\vec{u}$ is drawn from a ditribution $g(\vec{u})$ and has dimension $r$, while $\vec{u}\$ is drawn from $g'(\vec{u}')$ and has dimension $r'$.
We then define a deterministic mappring between the set of a state and its auxilary variables and the other one, i.e.
```{math}
(x', \vec{u}') = h(x, \vec{u}) \hspace{1cm} (x, \vec{u}) = h'(x', \vec{u'}).
```

The auxiliary variables and their distributions can be chosen arbitrarily, with the only condition that
```{math}
n_k + r = n_{k'} + r'.
```
This so-called  **dimension matching** condition ensures that the mapping $h$ is a diffeomorphism, which in turn is needed to ensure detailed balance.
Looking at a generalized version of the Metropolis-Hastings algorithm for moves between these two sets, the idea is that now every new point proposal probability is simply given by the probability of the drawn auxiliary variable, since then the mapping is deterministic.

Therefore, one can re-derive the acceptance probability for such trans-dimensional moves as
```{math}
:label: eq:acc_rjmcmc
\alpha(x,x') = \min \left\{ 1, \frac{\pi(x')g'(\vec{u}')}{\pi(x)g(\vec{u})} \left| \frac{\partial h(x,\vec{u})}{\partial(x,\vec{u})} \right|  \right\},
```
where the last term is the Jacobian corresponding to the mapping between states and auxiliary variables.


### RJMCMC algorithm

In a RJMCMC algorithm, at each sampler iterations for each chain we have

1. **Within-model move**: update the parameters of the chain staying in the same model. This can be done with a "standard" Metroplois-Hastings algorithm. One needs to define a proposal function for the move, and then decide whether to accept the new point or not with the acceptance probability in Eq.{eq}`eq:acc_mh`

```{image} ../images/within_models.png
:alt: within
:width: 600px
:align: center
```

2. **Between-model move**: propose a jump to a different model, updating $k$ and $\vec{\theta}_k$ at the same time. For this step one needs to define the mapping $h$ and its inverse $h'$. The jump is then accepted or rejected based on the acceptance probability in Eq.{eq}`acc_rjmcmc`, with the Jacobian term of the mapping $h$ (or $h'$ for the inverse move).

```{image} ../images/between_models.png
:alt: betw
:width: 600px
:align: center
```












```{bibliography}
