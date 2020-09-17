# rans(eXtreme) Analysis framework for compressible multi-fluid hydrodynamic simulations

# Overview

ransX framework implements mean field Reynolds-Averaged Navier-Stokes (a.k.a RANS) transport/flux/variance equations for mass, momenta, kinetic/internal/total energies, temperature, enthalpy, pressure and composition densities.

# Prerequisities

ransX requires python 2.7 and the following modules to work:

1. numpy (version <= 1.16.2)
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
conda create --name ransX python=2.7
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

# Available Documentation

Available documentation encompassing ransX theory/user/implementation and developer's guide can be found in the DOCS folder.

RansXtheoryGuide.pdf

RansXinstallationGuide.pdf

RansXimplementationGuide.pdf

RansXdevelopersGuide.pdf

RansXuserGuide.pdf
