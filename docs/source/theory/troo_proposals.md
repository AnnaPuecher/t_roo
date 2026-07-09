# Proposals in t-roo

t-roo is based on the RJMCMC framework implemented in eryn, an affine invariant ensemble sampler that is in turn based on emcee. On the otehr hand, to enable the analyses of CBC signals, it inherits also elements of bilby, such as the utilities to computed the likelihood and manage the detectors data.

Currently, t-roo is specifically deisgned to compare models describing different classes of CBC sources, i.e., binary black hole (BBH), binary neutron star (BNS) or neutron star-black hole (NSBH) systems. The main difference between these models is the presence of tidal information (and consequently parameters) for the NS components, therefore also the between-model proposal is focused on these parameters. Moreover, we added specific proposal also for the chirp mass $\mathcal{M}_c$ and mass ratio $q$, to account for the different valued recovered by models with and without tidal content. In the following we summarize the moves implemented in t-roo, which are also schematically shown in the figure below. All the details can be found in the paper.

```{image} ../images/troo.pdf
:alt: troo_moves
:width: 600px
:align: center
```

### Between model moves

We implement specific moves to go from BBH to NSBH, from NSBH to BNS, and from BNS to BBH, and viceversa.

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
        q_{BBH, i+1} = q_{BNS, i} - s_{q (BBH, BNS)}} \tilde{\Lambda}_{BNS, i} \\
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
