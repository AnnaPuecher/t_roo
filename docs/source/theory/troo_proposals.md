# Proposals in t-roo

t-roo is based on the RJMCMC framework, including parallel tempering, implemented in eryn, an affine invariant ensemble sampler that is in turn based on emcee. On the otehr hand, to enable the analyses of CBC signals, it inherits also elements of bilby, such as the utilities to computed the likelihood and manage the detectors data.

Currently, t-roo is specifically deisgned to compare models describing different classes of CBC sources, i.e., binary black hole (BBH), binary neutron star (BNS) or neutron star-black hole (NSBH) systems. The main difference between these models is the presence of tidal information (and consequently parameters) for the NS components, therefore also the between-model proposal is focused on these parameters. Moreover, we added specific proposal also for the chirp mass $\mathcal{M}_c$ and mass ratio $q$, to account for the different valued recovered by models with and without tidal content. In the following we summarize the moves implemented in t-roo, which are also schematically shown in the figure below. All the details can be found in the paper.

```{image} ../images/troo.png
:alt: troo_moves
:width: 800px
:align: center
```

### Stretch move

Both in the within-model and between-model moves t-roo uses **stretch-moves**, one of the standard moves employed in eryn and emcee.

In a stretch move, the new position of a walker $X_i$ at time $t$ is proposed by moving by a fraction of the distance with respect to another random walker $X_j$ as 
```{math}
    X_{i,{\rm new}} = X_j(t) - z (X_i(t) - X_j(t)),
```
with $z$ a random variable drawn from a distribution $g(z)$.
This probability distribution is
```{math}
    g(z) = \begin{cases}   
    \frac{1}{\sqrt{z}}, & \text{if $z \in [1/a,a]$}\\
    0, & \text{otherwise}, 
    \end{cases}
```
with $a$ arbitrarily set to 2.
The new proposed point is then accepted with probability
```{math}
    \alpha (X_{i, new}, X_i(t)) = \min \left\{1,  z^{d-1} \frac{p(X_{i, new})}{p(X_i(t))}\right\},
```
where $p(X)$ denotes the probability of a point $X$, $d$ is the dimension of the parameters involved in the stretch move, and $z^{d-1}$ a factor needed to preserve the symmetry condition.


### Preliminary sampling

For chirp mass and mass ratio we employ a tidal-dependent, data-informed proposal.  
A t-roo run starts with a **preliminary sampling** phase in which only within-model moves are performed (although the walkers can change model due to temperature swaps). At the end of this stage, the sampler looks at the last samples of chirp mass, mass ratio, and mass-weighted tidal deformability, computes the likelihood center-of-mass for each model, and finds the slope of the line conneting the center-of-mass likelihood in the $\mathcal{M}_c-\tilde{\Lambda}$ or $q-\tilde{Lambda}$ space. In the between-model moves, the new values for chirp-mass and mass-ratio are then proposed following a line going trough the current samples and with the slope computed this way.

The preliminaty sampling stage also improves the overall analysis convergenvce, because when the RJMCMC starts, the walkers have already identifies the parameters regions in which each model has support.



### Between model moves

t-roo is specifically design to compare models describing different kinds of binary systems. In particular, t-roo can also compare multiple models describing different or the same source at the same time, and it can directly jump between BBH and BNS models without the need to go through a NSBH model. Currently, all the models are always assumed to have the same probability, so at each step the jump is proposed to any other model with the same probability.

We implement specific moves to go from BBH to NSBH, from NSBH to BNS, and from BNS to BBH, and viceversa. In order to avoid losing information about the tidal parameters when going from a model that includes such parameters to a model that does not, we add the *pseudo*-parameters $Lambda_{1, BBH}$, $\Lambda_{2, BBH}$, and $\Lambda_{1, NSBH}$ in the BBH and NSBH models: these parameters serve just to store the information about the tidal parameters, and are not included in the likelihood computation. 
\vec{\Theta} denotes all the models parameters for which no specific proposal is implemeneted, while $u_1, u_2, v_1, v_2, w_1, w_2$ are the auxiliary variables. The reverse proposals are given by the inverse of the mappings below.


** BBH to NSBH **
```{math}
\begin{cases}
        \vec{\Theta}_{NSBH, i+1} = \vec{\Theta}_{BBH, i} \\
        \Lambda_{1, NSBH, i+1} = \Lambda_{1, BBH, i} \\
        \Lambda_{2, NSBH, i+1} = \Lambda_{2,j, i} + u_2 \left( \Lambda_{2, BBH, i} - \Lambda_{2,j, i} \right) \\
        \mathcal{M}_{c, NSBH, i+1} = \mathcal{M}_{c, BBH, i} + s_{\mathcal{M}_c (NSBH, BBH)} \cdot \bar{\tilde{\Lambda}}_{NSBH, i+1} \\
        q_{NSBH, i+1} = q_{BBH, i} + s_{q (NSBH, BBH)} \cdot \bar{\tilde{\Lambda}}_{NSBH, i+1} \\
        v_2 = 1/u_2 
        \end{cases}  \label{eq:bbh_to_nsbh}
```

**BNS to BBH**
```{math}
\begin{cases}
        \vec{\Theta}_{BBH, i+1} = \vec{\Theta}_{BNS, i} \\
        \Lambda_{1, BBH, i+1} = \Lambda_{1,j, i} + w_{1} \left( \Lambda_{1, BNS, i} - \Lambda_{1,j, i} \right) \\
        \Lambda_{2, BBH, i+1} = \Lambda_{2,j, i} + w_2 \left( \Lambda_{1, BNS, i} - \Lambda_{2,j, i} \right) \\
        \mathcal{M}_{c, BBH, i+1} = \mathcal{M}_{c, BNS, i} - s_{\mathcal{M}_c (BBH, BNS)} \tilde{\Lambda}_{BNS, i} \\
        q_{BBH, i+1} = q_{BNS, i} - s_{q (BBH, BNS)} \tilde{\Lambda}_{BNS, i} \\
        u_1 = 1 / w_1 \\
        u_2 = 1/w_2 
        \end{cases}
```

**NSBH to BNS**
```{math}
\begin{cases}
        \vec{\Theta}_{BNS, i+1} = \vec{\Theta}_{NSBH,i} \\
        \Lambda_{1, BNS, i+1} = \Lambda_{1,j,i} + v_1 \left( \Lambda_{1, NSBH, i} - \Lambda_{1,j, i} \right) \\
        \Lambda_{2, BNS, i+1} = \Lambda_{2,j,i} + v_2 \left( \Lambda_{1, NSBH, i} - \Lambda_{2,j, i} \right) \\
        \begin{aligned}
        \mathcal{M}_{c, BNS, i+1} = &\mathcal{M}_{c, NSBH, i} + \\ &- s_{\mathcal{M}_c (BNS, NSBH)}  \left( \bar{\tilde{\Lambda}}_{NSBH, i} - {\tilde{\Lambda}}_{BNS, i+1} \right) \end{aligned} \\
        q_{\textsc{bns}, i+1} = q_{NSBH, i} - s_{q (BNS, NSBH)}  \left( \bar{\tilde{\Lambda}}_{NSBH, i} - {\tilde{\Lambda}}_{BNS, i+1} \right) \\
        w_1 = 1 / v_1 \\
        w_2 = 1/v_2
        \end{cases}
```

In the case in which we want to jump between models belonging to the same source class, instead, all the parameters are kept the same. This is possible because in the current implementation we only consider models with the same physics content, for example all aligned-spin models and not precessing ones.

### Within-model moves

In the within-model moves, we need to update also the *pseudo*-parameters $\Lambda_{1, BBH}$, $\Lambda_{2, BBH}$, $\Lambda_{1, NSBH}$, since they are effectively part of the parameter space we consider for each state. Since these parameters do not enter the likelihood calculations, and thus we do not expect to recover informationa about them, we expect the walkers to return the prior for these parameters. For consistency with the actual tidal parameters, we choose a uniform prior for the *pseudo*-parameters. The fastest way to recover the uniform prior when the walkers visit often these model is with a random walk. Therefore $\Lambda_{1, BBH}$, $\Lambda_{2, BBH}$, $\Lambda_{1, NSBH}$ are updated with a random walk, with default step 10. 

We update all the other parameters of a model together with a strecth move. Also block sampling is implemented in t-roo, but not used by default to improve convergence.

