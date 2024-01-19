dyeScreen
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/mariacm12/dyeScreen/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/dyeScreen/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/mariacm12/dyeScreen/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/dyeScreen/branch/main)


A code for the High-Throughput Screening of Molecules in DNA Scaffolding and studying the Electronic Structure properties of chromophores in dissordered systems. Provides functions for studying the molecules based on Molecular Dynamics (MD) simulations and quantum mehanical calculations (QM).

### Copyright

Copyright (c) 2024, Maria A. Castellanos

### Requirements
------------
- Python >= 3.8
- [PySCF](https://pyscf.org/install.html)
- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [AmberTools](https://ambermd.org/GetAmber.php)
- [MDAnalysis](https://www.mdanalysis.org/pages/installation_quick_start/)
- Access to Q-Chem

### Instructions
------------
The parent directory of `dyeScreen` must be in `$PYTHONPATH`, enabling
    `import dyeScreen`.

To add the package to the path, add the following lines to the ~/.bashrc or ~/.bash\_profile file:  
```
{
  export DYESCREEN=/pathto/dye-screening  
  export PYTHONPATH=$DYESCREEN:$PYTHONPATH
}
```

Create an environment variable with the location of AmberTools executables:  
`export AMBERPATH="/pathtoAmberTools/bin/"`

### Contents
--------

The following is a diagram showing the flow of information in our screening software package.
![A flow diagram for the code](docs/code_diagram.pdf)

- README.md
- commons/  
    This module contains two submodules with functions shared by the other three modules.
    - `commons/couplingutils.py`: Functions for carrying out Quantum Mechanical (QM) calculations from MD trajectory coordinates. All QM is performed in PySCF, MD trajectories are process with MDAnalaysis.
    - `commons/geom_utils.py`: Functions for geometry manipulation/processing from MD trajectories or PDB files data. Integration is done using MDAnalyisis.
- FF/  
    This module contains functions for building the force field for the dye and the linker using a fragment-based approach.
    - `FF/gen_dye_lnk.py`
    - `FF/file_process.py`
- MD/
    This module carries out the High-Throughput sampling of a dye+linker molecule within a DNA fragment and initializes MD simulations for each one of the generated samples.
    - `MD/sampling.py`: Performs the sampling starting with a PDB for the dye+linker structure and the DNA, and the force-field mol2/frcmod files.
    - `MD/md_samples.py`: Takes the output from sampling to initialize the MD simulations.
- QM/  
    - `QM/cluster_trajs.py`
    This module contains function for importing MD trajectories (as parameter prmtop and trajectory nc files) and statistical analysis of the data.  
    
### To-do
-----
- [ ] Fix and test function for joining dye and linker frcmod files `FF/gen_dye_lnk.py/join_dye_lnk_frcmod`. 
- [ ] Enhanced sampling of parallel dimers trapped in local minima.
- [ ] PySCF implementation of electron/hole transfer integrals t\_e/t\_h.
- [ ] Tests 

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
