# T-ROO #

Tipsy kangaROO: a Reversible Jump Markov chain Monte Carlo (RJMCMC) sampler for compact-binary-coalescence gravitational-wave signals. Based on [eryn](https://github.com/mikekatz04/Eryn) and [bilby](https://github.com/bilby-dev/bilby).

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

Issues:
- Currently conda-installed bilby can give the following error `ImportError: cannot import name 'btdtri' from 'scipy.special._ufuncs'`
  Can be solved by installing bilby versions > 2.5
- to avoid conflicts with numpy deprecations bilby >= 2.7
- bilby is developed and tested for python 3.10-3.12 (for example pythonv3.14 gives issues with importing astropy)

### Citation
