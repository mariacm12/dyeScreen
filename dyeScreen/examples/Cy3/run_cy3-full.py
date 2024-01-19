'''
Cy3 calculation but using a dye+linker optimized geometry
'''

import os
import numpy as np
from dyeScreen.commons.geom_utils import join_pdb
from dyeScreen import FF

path = "dyeScreen/examples/Cy3/cy3_full/"

# Amber paths (change if different)
# If saved as an environment variable
amber_path = os.getenv("AMBERPATH")
# If installed in miniconda
# amber_path = "/Users/username/opt/miniconda3/envs/AmberTools23"
# If installed in Nersc 
#amber_path = "/global/common/software/nersc/pm-2021q4/sw/amber/20" 

path_achamber = amber_path + "/bin/antechamber" # change if different
path_parmchk2 = amber_path + "/bin/parmchk2"

#- 1) We start with the pre-optimized geometries of the dye and linker
# Cy3 is small, so there's no need to optimize the linker separately

dye_file = path+"geom_Cy3-link.pdb"

lnk_pdb = None
dye_pdb = path+"cy3-link.pdb"

#- 2) We clean the initial pdbs so Amber can read them
FF.clean_pdb(dye_file, dye_pdb, res_code='CY3', mol_title='Cy3')

#- 3) We run antechamber to get the mol2 files with am1-bcc charges
FF.gen_antech_mol2(dye_pdb, lnk_pdb, path_achamber, ch_dye=-1)

#- 5) We generate the frcmod files
FF.gen_frcmod(dye_pdb[:-3]+"mol2", None, path_parm=path_parmchk2)







