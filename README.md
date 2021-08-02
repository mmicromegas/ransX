# rans(eXtreme) Analysis framework for compressible multi-fluid hydrodynamic simulations

# Overview

ransX framework implements mean field Reynolds-Averaged Navier-Stokes (a.k.a RANS) transport/flux/variance equations for mass, momenta, kinetic/internal/total energies, temperature, enthalpy, pressure and composition densities. For a quick but limited exposure to some of these equations, check out the  [ransX Web Studio](http://ransx.pythonanywhere.com) (all browsers supported) or [ransX code-comparison-project Web Studio](http://codecomparisonproject.pythonanywhere.com/) (do not use Firefox). Theoretical background on RANS analysis can be found here [Compressible Hydrodynamic Mean-Field Equations in Spherical Geometry and their Application to Turbulent Stellar Convection Data](https://ui.adsabs.harvard.edu/abs/2014arXiv1401.5176M/abstract).

# Prerequisities

ransX requires python 3 and the following modules to work:

1. numpy
2. scipy
3. matplotlib
4. ast
5. sys

# Installation

Download the repository:

```
git clone https://github.com/mmicromegas/ransX
```

Change to ransX directory:

```
cd ransX
```

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and create a Conda environment:

```
conda create --name ransX python=3.8
```

Activate the environment:

```
conda activate ransX
```

Install Python libraries listed in `requirements.txt` file:

```
pip install -r requirements.txt
```




# Usage

1. cd ransX
2. start ipython
3. run ransX.py
4. run ransX_res.py
5. run ransX_evol.py
6. run ransX_reaclib.py

# Available Data (research-grade)

Mean fields and their evolution based on 2D and 3D simulations of Two-Layer setup specified for the code comparison project https://sites.google.com/view/stellarhydrodays/home

<img src="https://user-images.githubusercontent.com/34376626/103656159-8ca8d500-4f68-11eb-92dd-a1cb7d41f2b1.png" width="25%"></img> 

Mean fields based on 3D Oxygen-Neon Burning Shell in a massive star

<img src="https://user-images.githubusercontent.com/34376626/103656869-6e8fa480-4f69-11eb-909d-335d12398a9a.png" width="25%"></img> 

# Available Documentation (a bit outdated)

Available documentation encompassing ransX theory/user/implementation and developer's guide can be found in the DOCS folder.

RansXtheoryGuide.pdf

RansXinstallationGuide.pdf

RansXimplementationGuide.pdf

RansXdevelopersGuide.pdf

RansXuserGuide.pdf
