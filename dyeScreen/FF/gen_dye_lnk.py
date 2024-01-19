#%%
import numpy as np
import subprocess
import dyeScreen.FF.file_process as fp

def clean_pdb(in_path, out_path, res_code='DYE', mol_title='Dye molecule'):
    """Prepare PDB file for Amber Antechamber

    Args:
        in_path (str): Path to input pdb file
        out_path (str): Path for output pdb file
        res_code (str, optional): Residue name for final pdb. Defaults to 'DYE'.
        mol_title (str, optional): Molecule name in final pdb. Defaults to 'Dye molecule'.
    """
    lpdb = fp.PDB_DF()
    lpdb.read_file(in_path)
    # Clean existing names (if already numbered)
    atom_names = lpdb.data['HETATM']['atom_name']
    clean_atoms = fp.clean_numbers(atom_names)

    # Sort atoms in the dataframe
    hetatm = lpdb.data['HETATM']
    hetatm['atom_name'] = clean_atoms
    hetatm_sorted = hetatm.sort_values(by=['atom_name', 'atom_id'],ascending=[True, True])
    sorted_atoms = hetatm_sorted['atom_name']

    print(hetatm.head())

    # Replace atom names by numbered names
    asymbol, acounts = np.unique(sorted_atoms, return_counts=True)
    fixed_names = fp.make_names(asymbol,acounts)
    hetatm_sorted['atom_name'] = fixed_names

    #Replace res names
    #res_names = fp.set_res(res_code, hetatm_sorted)
    #hetatm_sorted['res_name'] = res_names

    # Save pdb file
    lpdb.data['MOLECULE'] = mol_title
    lpdb.data['HETATM'] = hetatm_sorted

    lpdb.write_file(out_path, resname=res_code)

    return

def gen_antech_mol2(cleaned_dye, cleaned_link, path_ac, ch_dye=0, ch_link=-1):
    """Generate and save mol2 files with Antechamber and am1-bcc method

    Args:
        cleaned_dye (str): Path to pre-cleaned PDB of dye
        cleaned_link (str): Path to pre-cleaned PDB of linker
        path_ac (str): Path to Antechamber executable
        ch_dye (int, optional): Total charge of dye. Defaults to 0.
        ch_link (int, optional): Total charge of linker. Defaults to -1.
    """
    if cleaned_link: # False if the linker is not given
        commandl = f'{path_ac} -i {cleaned_link} -fi pdb -o {cleaned_link[:-3]}mol2 -fo mol2 -c bcc -s 2 -nc {ch_link} -m 1 -at gaff'
        a = subprocess.Popen(commandl, shell=True)
        a.wait()
    commandd = f'{path_ac} -i {cleaned_dye} -fi pdb -o {cleaned_dye[:-3]}mol2 -fo mol2 -c bcc -s 2 -nc {ch_dye} -m 1 -at gaff'
    b = subprocess.Popen(commandd, shell=True)
    b.wait()
    # can run these commands by themselves in the terminal if antechamber is installed,
    #  but if the calculations are expensive, they can be added to a sh file configured for HPC
    # Note that a Python file, running in hpc, can execute this function as well, in which case
    # the current implementation works fine. 
    '''
    base_hpc = "run_antch.sh"

    if cleaned_link:
        a = subprocess.Popen(f'echo {commandl} >> {base_hpc}', shell=True)
        a.wait()
    b = subprocess.Popen(f'echo {commandd} >> {base_hpc}', shell=True)
    b.wait()
    c = subprocess.Popen(f'echo "wait" >> {base_hpc}', shell=True)
    c.wait()
    d = subprocess.Popen(f'sbatch {base_hpc}', shell=True)
    d.wait()
    '''
    return 


def gen_frcmod(mol2_dye, mol2_link, path_parm):
    """Generate and save frcmod files for dye and linker

    Args:
        mol2_dye (path): Path to dye's mol2 file
        mol2_link (path): Path to linker's mol2 file
        path_parm (path): Path to Amber's parmchk executable
    """

    if mol2_link:
        commandl = f'{path_parm} -i {mol2_link} -f mol2 -o {mol2_link[:-4]}frcmod'
        a = subprocess.Popen(commandl, shell=True)
        a.wait()
    commandd = f'{path_parm} -i {mol2_dye} -f mol2 -o {mol2_dye[:-4]}frcmod'
    b = subprocess.Popen(commandd, shell=True)
    b.wait()
    return


def join_dye_lnk_frcmod(dye_frcmod, lnk_frcmod, file_both):
    """Joining dye and linker frcmod files. NOT TESTED

    Args:
        dye_frcmod (str): Path to dye's frcmod.
        lnk_frcmod (str): Path to linker's frcmod.
        file_both (str): Path to save combined frcmod
    """
    categ = ["MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON", "^$"]

    with open(file_both, "w") as f:
        f.write(f"remark goes here\n")
        for i in range(len(categ)-1):
            f.write(categ[i]+'\n')
            command = f"sed -n '/{categ[i]}/, /{categ[i+1]}/{{ /^$/! {{ /{categ[i]}/! {{ /{categ[i+1]}/! p }} }} }}' "

            commandd = command + lnk_frcmod #+ " > temp1.txt"
            commandl = command + dye_frcmod #+ " > temp2.txt"
            print(commandd, commandl)
            a = subprocess.Popen(commandd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            aout, err = a.communicate()
            b = subprocess.Popen(commandl, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            bout, err = b.communicate()
            #subprocess.Popen("cat temp1.txt temp2.txt > temp1.txt", shell=True)
            print(str(aout), str(bout))
            f.write(str(aout)+str(bout))
            f.write("\n")

    return

def process_mol2(dye_path, lnk_path, dye_out, lnk_out, dye_del, lnk_del, 
                 chg_fix_atoms, dye_code=None, lnk_code=None, num_links=1, extra_del=None):
    """Process mol2 files to reflect the new dye-linker bond.

    Args:
        dye_path (str): Path to dye input mol2
        lnk_path (str): Path to linker input mol2
        dye_out (str): Path to dye output mol2
        lnk_out (str): Path to linker output mol2_
        dye_del (list): List of IDs of atoms to delete on dye
        lnk_del (list): List of IDs of atoms to delete on linker
        chg_fix_atoms (list): List of length (num_links+1) with the atoms participating in d-l 
                              bonding [dye1, dye2, ..., linker] for charge adjustment.
        dye_code (str, optional): Residue name dye. Defaults to None.
        lnk_code (str, optional): Residue name linker. Defaults to None.
        num_links (int, optional): Number of linkers attached to dye. Defaults to 1.
        extra_del (list, optional): Additional atoms to delete (IDs). Defaults to None.

    Returns:
        list: If provided, linker atoms to delete later
    """
    dmol2 = fp.MOL2_DF()

    dmol2.read_file(dye_path)
    d_df = dmol2.data['ATOM']
    dtotal_charge = fp.calculate_charges(d_df)
    # delete extra atoms
    new_dye, new_dbonds, dye_del_data = fp.del_atoms(dmol2, dye_del)

    if lnk_path:
        lmol2 = fp.MOL2_DF()
        lmol2.read_file(lnk_path)
        l_df = lmol2.data['ATOM']
        ltotal_charge = fp.calculate_charges(l_df)*num_links
        # delete extra atoms
        new_lnk, new_lbonds, link_del_data = fp.del_atoms(lmol2, lnk_del, del_later=extra_del)

        # Re-adjust charges
        new_dye, new_lnk, last_charge = fp.fix_charge(new_dye, new_lnk, dtotal_charge+ltotal_charge, chg_fix_atoms)
        print(f'corrected charge {last_charge}, actual is {dtotal_charge+ltotal_charge}')

        # Updating pandas dictionary
        lmol2.data['ATOM'] = new_lnk
        lmol2.data['BOND'] = new_lbonds
        lmol2.write_file(lnk_out, resname=lnk_code)

    else:
        new_dye, last_charge = fp.fix_charge_nolink(new_dye, dtotal_charge, chg_fix_atoms)

    dmol2.data['ATOM'] = new_dye
    dmol2.data['BOND'] = new_dbonds
    dmol2.write_file(dye_out, resname=dye_code)
    return link_del_data

def join_dye_lnk_pdb(dye_path, lnk_path, mol_out, dye_del, lnk_del, dye_bond, lnk_bond,
                     path_bond_info = None):
    """Old implementation of function for joining dye and linker pdb
        ** Refer to the join_pdb function in commons.geom_utils"""
    
    num_links = len(dye_bond)

    # Read dye and linker PDBs
    lpdb = fp.PDB_DF()
    lpdb.read_file(lnk_path)
    lpdbs = [lpdb]*num_links
    dpdb = fp.PDB_DF()
    dpdb.read_file(dye_path)

    lhetatm = lpdb.data['HETATM']
    dhetatm = dpdb.data['HETATM']

    # Translate linker to bond position and merge pdb dataframes
    bonds_info = []
    new_dye, __, __ = fp.move_to_bond(dhetatm, lhetatm, dye_del, lnk_del, 
                                        dye_bond[0], lnk_bond, 
                                        path_bond_info=path_bond_info)
    dpdb.data['HETATM'] = new_dye
    new_pdb = dpdb
    for i in range(num_links):
        __, new_link, bond_info = fp.move_to_bond(dhetatm, lhetatm, dye_del, lnk_del, 
                                            dye_bond[i], lnk_bond) 

        lpdbs[i].data['HETATM'] = new_link
        bonds_info.append(bond_info)
        
        new_pdb = fp.join_molecules(new_pdb, lpdbs[i])

    bonds_info = [element for sublist in bonds_info for element in sublist]
    np.savetxt(path_bond_info, bonds_info, fmt="%s")

    # Write pdb without connections
    new_pdb.write_file(mol_out, print_connect=False, reset_ids=True)
    return bonds_info

def write_leap(mol2_dye, mol2_link, pdb, frcmod, file_save, dye_res='DYE', link_res='LNK', 
               add_ions=None, make_bond=None, water_box=20.0):
    """Write Amber LEaP input for MD optimization of dye+linker 

    Args:
        mol2_dye (str): Path to mol2 of dye.
        mol2_link (str): Path to mol2 of linker.
        pdb (str): Path to pdb of dye+linker. 
        frcmod (str): Path to frcmod of dye+linker. 
        file_save (str): Save path for leap file.
        dye_res (str, optional): Residue name of dye. Defaults to 'DYE'.
        link_res (str, optional): Residue name of linker. Defaults to 'LNK'.
        add_ions (list, optional): Adding ions to neutralize charge ['ion', charge]. 
        make_bond (list, optional): Info on atoms that participate in dye-linker bonding.
        water_box (float, optional): Size of water box in Ams. Defaults to 20.0.
    """
    fprefix = pdb[:-3]
    with open(file_save, 'w') as f:
        f.write("source leaprc.gaff\n")
        f.write("source leaprc.gaff2\n")
        f.write("source leaprc.water.tip3p\n")
        f.write(f"{dye_res} = loadmol2 {mol2_dye} \n")
        if mol2_link:
            f.write(f"{link_res} = loadmol2 {mol2_link} \n")
        f.write(f"loadAmberParams {frcmod} \n")
        f.write(f"mol = loadpdb {pdb} \n")
        if make_bond is not None:
            for i in range(len(make_bond)//2):
                f.write(f"bond mol.1.{make_bond[0+i*2]} mol.{2+i}.{make_bond[1+i*2]} \n")
        if add_ions is not None:
            f.write(f"addIons mol {add_ions[0]} {add_ions[1]} \n")
        f.write(f"solvatebox mol TIP3PBOX {water_box}\n")
        f.write(f"saveAmberParm mol {fprefix}prmtop {fprefix}rst7\n")
        f.write("quit")

    return
