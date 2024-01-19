'''
Cy3 calculation using a fragment-base approach
'''

import os
import numpy as np
from dyeScreen.commons.geom_utils import join_pdb
from dyeScreen import FF

path = "dyeScreen/examples/Cy3/cy3_dyelink"

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
# dye and linker are processed separately for testing purposes

dye_file = path + "/geom_Cy3-nolink.pdb"
lnk_file = path + "/geom_linker.pdb"

lnk_pdb = path + "/linker_final.pdb"
dye_pdb = path + "/cy3_final.pdb"

#- 2) We clean the initial pdbs so Amber can read them
FF.clean_pdb(dye_file, dye_pdb, res_code='CY3', mol_title='Cy3')
FF.clean_pdb(lnk_file, lnk_pdb, res_code='LNK', mol_title='Linker')

#- 3) We run antechamber to get the mol2 files with am1-bcc charges
#     Depending on the size of the molecule it can take a while to run 
FF.gen_antech_mol2(dye_pdb, lnk_pdb, path_achamber, ch_dye=1, ch_link=-1)

#- 5) We generate the frcmod files
# Pending join frcmod of dye and linker
FF.gen_frcmod(dye_pdb[:-3]+"mol2", lnk_pdb[:-3]+"mol2", path_parm=path_parmchk2)

#- 4) We clean the mol2 files to reflect dye+linker bond
lnk_in = f"{lnk_pdb[:-3]}mol2"
dye_in = f"{dye_pdb[:-3]}mol2"

lnk_mol2 = path + "/linker.mol2"
dye_mol2 = path + "/cy3.mol2"


dye_del = [38,41,42,56] + [33,39,40,55]
link_del = [1,2,3,4,5] + [17,18,19,20] 
link_po3 = [21,22,23,24] # Also delete the atoms bonding to the DNA (leave P)
dye_del.sort()
link_del.sort()
chg_fix_atoms = [7,17,6] # format: [dye1,dye2,...,dyen,link], n is num_links

link_del_data = FF.process_mol2(dye_in, lnk_in, dye_mol2, lnk_mol2,
                                dye_del, link_del, chg_fix_atoms, dye_code='PNT', lnk_code='LNK',
                                num_links=2, extra_del=link_po3)


# -6) We generate a pdb for the dye and linker
mol_pdb = "cy3_wLinker.pdb"
dye_bond = [[7,38],[17,33]] #[keep, replace]*num_links
link_bond = 6
#bond_atoms = join_dye_lnk_pdb(dye_pdb, lnk_pdb, f"{path}/{mol_pdb}", 
 #                             dye_del, link_del, dye_bond, link_bond,
 #                             path_bond_info = f"{path}/dye-link-bond.txt")

# The following implementation aligns the dye linker as far as possible
bond_atoms = join_pdb(dye_pdb, lnk_pdb, dye_del, link_del, 
                      dye_bond, link_bond, path_bond_info=f"{path}/dye-link-bond.txt", 
                      path_save=f"{path}/{mol_pdb}")

# 7) Write leap input file
leap_out = "{path}/tleap_noDNA.in"
frcmod = "cy3_linker.frcmod" #Joined manually
FF.write_leap(dye_mol2, lnk_mol2, mol_pdb, frcmod, leap_out, 
              dye_res='CY3', link_res='LNK', add_ions=['NA', '0'],
              make_bond=bond_atoms, water_box=20.0)

