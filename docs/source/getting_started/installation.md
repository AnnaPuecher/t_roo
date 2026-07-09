## Installation instructions

The code can be cloned from the repo: [https://github.com/AnnaPuecher/t_roo](https://github.com/AnnaPuecher/t_roo)
and can easily installed by running

```
pip install .
```

It is advisible to create a specific `conda` enviroment to run t-roo. An example of complete installation workflow is the following
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

### Possible issues

- Currently conda-installed bilby can give the following error `ImportError: cannot import name 'btdtri' from 'scipy.special._ufuncs'`
  Can be solved by installing bilby versions > 2.5
- to avoid conflicts with numpy deprecations bilby >= 2.7
- bilby is developed and tested for python 3.10-3.12 (for example pythonv3.14 gives issues with importing astropy)

### Waveform models

t-roo can directly run analyses with the models implemented and available in `lalsimualtion`. Any other model should be installed separately and might require specific wrappers to be used.

**Note**: differently from the other models, in the lalsimulation implementation the IMRPhenomNSBH model has a `-` sign in front of the amplitude. Although it is mainly a matter of conventions and this does not affect usual analyses with just this model, when comparing it with other approximants in a RJMCMC approach it creates issues because it effectively corresponds to a phase shift of $\pi$. The option `fix_phenomnsbh` in the wrapper for the likelihood wrapper is implemented specifically to take care of this feature. Alternatively, one can manually install lalsimulation and change the amplitude sign in the waveform. 
