# t-roo #

**t**ipsy kanga**roo**[^1]: a Reversible Jump Markov chain Monte Carlo (RJMCMC) sampler for compact-binary-coalescence gravitational-wave signals. Based on [eryn](https://github.com/mikekatz04/Eryn) and [bilby](https://github.com/bilby-dev/bilby).

<img width="300" height="300" alt="t-roo_name" class="center-image" src="https://github.com/user-attachments/assets/a1f8db76-2b17-4fd4-a3e7-3905c55ba21f" />

***A story about a tipsy kangaroo: Reversible jump MCMC for model selection in the analysis of gravitational-wave signals from the coalescence of compact objects***

Once upon a time, there was a cute kangaroo who was studying gravitational waves signals. Some signals were very complicated, so the kangaroo could not understand what source they were coming from... was it a BBH, a NSBH, a BNS? Walking through each model and then look at the different results was taking too much time. A bit discouraged, the kangaroo decided to go out for drinks with his friends. After a few drinks, the kangaroo decided to jump between the different models instead of just walking through them, it would have been much more fun. Since it was a bit tipsy, the kangaroo started jumping back and forth, with back flips and reverse jumps. And this is how the tipsy kangaroo found the best way to jump between models to understand the source of the GWs, and it finally became a happy t-roo.

[^1]: *kangaroo* because it recalls the *jump* in ``reversible jump", and for consistency with Australian names for samplers such as bilby and dingo. The kangaroo is *tipsy* because it not only jumps, but it also *reverse* jumps.

### Installation

```
conda create -n t-roo-env python=3.11
conda activate t-roo-env
conda install numpy scipy pandas matplotlib seaborn
conda install -c conda-forge bilby=2.7
conda install -c conda-forge lalsimulation
cd ${CONDA_PREFIX}
mkdir src
cd src
git clone https://github.com/AnnaPuecher/t_roo.git
cd t_roo/t_roo
pip install .
```

### Documentation

Complete documentation can be found here:

Example scripts to run t-roo on injections and real data are provided in

### Citation

### Acknowledgements

The project logo was created with the assistance of ChatGPT (OpenAI).

We acknowledge funding from the EU Horizon under ERC Starting Grant, no.SMArt-101076369. Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<img width="485" height="209" alt="LOGO_ERC-FLAG_EU" src="https://github.com/user-attachments/assets/6bc996d7-1674-41c2-bae1-c6ee51676c64" />




