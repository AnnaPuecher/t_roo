# Proposals in t-roo

t-roo is based on the RJMCMC framework implemented in eryn, an affine invariant ensemble sampler that is in turn based on emcee. On the otehr hand, to enable the analyses of CBC signals, it inherits also elements of bilby, such as the utilities to computed the likelihood and manage the detectors data.

Currently, t-roo is specifically deisgned to compare models describing different classes of CBC sources, i.e., binary black hole (BBH), binary neutron star (BNS) or neutron star-black hole (NSBH) systems. The main difference between these models is the presence of tidal information (and consequently parameters) for the NS components, therefore also the between-model proposal is focused on these parameters. Moreover, we added specific proposal also for the chirp mass $\mathcal{M}_c$ and mass ratio $q$, to account for the different valued recovered by models with and without tidal content. In the following we summarize the moves implemented in t-roo, which are also schematically shown in the figure below. All the details can be found in the paper.

```{image} ../images/troo.pdf
:alt: troo_moves
:width: 800px
:align: center
```

### Between model moves

We implement specific moves to go from BBH to NSBH, from NSBH to BNS, and from BNS to BBH, and viceversa.

```{math}
\begin{cases}
        \vec{\Theta}_{\textsc{nsbh}, i+1} = \vec{\Theta}_{\textsc{bbh}, i} \\
        \Lambda_{1, \textsc{nsbh}, i+1} = \Lambda_{1, \textsc{bbh}, i} \\
        \Lambda_{2, \textsc{nsbh}, i+1} = \Lambda_{2,j, i} + u_2 \left( \Lambda_{2, \textsc{bbh}, i} - \Lambda_{2,j, i} \right) \\
        \mathcal{M}_{c, \textsc{nsbh}, i+1} = \mathcal{M}_{c, \textsc{bbh}, i} + s_{\mathcal{M}_c (\textsc{nsbh}, \textsc{bbh})} \cdot \bar{\tilde{\Lambda}}_{\textsc{nsbh}, i+1} \\
        q_{\textsc{nsbh}, i+1} = q_{\textsc{bbh}, i} + s_{q (\textsc{nsbh}, \textsc{bbh})} \cdot \bar{\tilde{\Lambda}}_{\textsc{nsbh}, i+1} \\
        v_2 = 1/u_2 
        \end{cases}  \label{eq:bbh_to_nsbh}
```
