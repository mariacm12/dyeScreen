#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:30:47 2020
@author: mariacm
"""
import numpy as np
from numpy import linalg as LA
from math import sqrt, pi, cos, sin

# MD analysis tools
import MDAnalysis as mda
import dyeScreen.commons.couplingutils as cu

import csv

ams_au = 0.52917721092
cm1toHartree = 4.5563e-6

# Rotation operators


def Rz(th): return np.array([[cos(th), -sin(th), 0],
                             [sin(th), cos(th), 0],
                             [0, 0, 1]])


def Ry(th): return np.array([[cos(th), 0, sin(th)],
                             [0, 1, 0],
                             [-sin(th), 0, cos(th)]])


def Rx(th): return np.array([[1, 0, 0],
                             [0, cos(th), -sin(th)],
                             [0, sin(th), cos(th)]])


def get_namelist(param, traj_path, traj_num, resnum="11"):

    u = mda.Universe(param, traj_path+str(traj_num)+".nc", format="TRJ")

    sel1 = u.select_atoms("resid "+resnum)

    return sel1.atoms.names


def get_dihedral(param, traj_path, ntrajs=8, reps=['A', 'B'], resnum1='11', resnum2='12'):

    both_dih1 = []
    both_dih2 = []

    for rep in reps:

        dih1 = []
        dih2 = []
        for ic in range(1, ntrajs+1):

            # Importing MD trajectories
            u = mda.Universe(param, traj_path+rep+str(ic)+".nc", format="TRJ")
            #u = mda.Universe("%s/Sq%s_dimer.prmtop"%(path,typ), "%s/prod/Sq%sDim_prod%s"%(path,typ,rep)+str(ic)+".nc", format="TRJ")

            tin = 0
            dt = 5

            print("Traj #%s, rep %s" % (str(ic), rep))

            num_sol = 700
            extra_t = 0
            istep = 1

            for ts in u.trajectory[num_sol*(istep-1)+tin+extra_t:num_sol*istep+tin+extra_t:dt]:

                sel1 = u.select_atoms("resid "+resnum1)
                sel2 = u.select_atoms("resid "+resnum2)

                dihedrals1 = sel1.dihedrals
                dihedrals2 = sel2.dihedrals

                dihs1 = dihedrals1.dihedrals()
                dihs2 = dihedrals2.dihedrals()

                dih1.append(dihs1)
                dih2.append(dihs2)

        dihs1 = np.array(dih1)
        dihs2 = np.array(dih2)

        both_dih1.append(dih1)
        both_dih2.append(dih2)

    dih_1 = np.average(np.array(both_dih1), axis=0)
    dih_2 = np.average(np.array(both_dih2), axis=0)

    atom1_1 = dihedrals1.atom1.names
    atom2_1 = dihedrals1.atom2.names
    atom3_1 = dihedrals1.atom3.names
    atom4_1 = dihedrals1.atom4.names

    atom1_2 = dihedrals2.atom1.names
    atom2_2 = dihedrals2.atom2.names
    atom3_2 = dihedrals2.atom3.names
    atom4_2 = dihedrals2.atom4.names

    return dih_1, dih_2, (atom1_1, atom2_1, atom3_1, atom4_1), (atom1_2, atom2_2, atom3_2, atom4_2)


def RAB_calc(u, dt, selA, selB, Rabs=[]):

    num_sol = len(u.trajectory)
    istep = 1

    for ts in u.trajectory[num_sol*(istep-1):num_sol*istep:dt]:

        sel1 = u.select_atoms(selA)
        sel2 = u.select_atoms(selB)

        CofMA = sel1.center_of_mass()
        CofMB = sel2.center_of_mass()

        rab = abs(CofMA-CofMB)
        Rabs.append(rab)

    return Rabs


def get_RAB(param, traj_path, rep, ntrajs=8, traji=1, dt=5, resnum1='11', resnum2='12'):

    selA = "resid "+resnum1
    selB = "resid "+resnum2

    if ntrajs>1:
        Rabs = []
        for ic in range(traji, ntrajs+1):

            # Importing MD trajectories
            u = mda.Universe(param, traj_path+rep+str(ic)+".nc", format="TRJ")

            print("RAB Traj #%s, rep %s" % (str(ic), rep))

            Rabs = RAB_calc(u, dt, selA, selB, Rabs=Rabs)

        Rabs = np.array(Rabs)
        RAB = np.linalg.norm(Rabs, axis=1)
        print(RAB.shape)
    else:
        # Importing single MD trajectory
        u = mda.Universe(param, traj_path+rep+".nc", format="TRJ") 
        Rabs = RAB_calc(u, dt, selA, selB, Rabs=[])
        RAB = np.linalg.norm(Rabs, axis=1) 
        print(RAB.shape, len(u.trajectory))

    # RAB = np.average(np.array(both_RAB),axis=0)
    return RAB


def get_coords_old(path, param_file, select, file_idx=None, dt=2, resnum1='11', resnum2='12', cap=True):
    """
    Given a selected time in a given traj file, it returns the (MDAnalysis) molecule params.    
    Parameters
    ----------
    path : String
        Location of traj files.
    tdye : String
        "I" or "II", the type of the dimer.
    select : int
        The index of the time-frame to extract.
    rep : String
        "A", "B", "C" or "D"
    Returns
    -------
    Parameters of H-capped Dimer at the selected time-frame.
    Format is: ( xyzA, xyzB, namesA, namesB, atom_typesA, atom_typesB, 
                 [list of bonds]A, [list of bonds]B )
    """


    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        select = 0
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        select = 0
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        u = mda.Universe(param_file, traj_file +
                         str(file_idx)+".nc", format="TRJ")
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+str(file_idx)+".nc")

    for fi, ts in enumerate(u.trajectory[::dt]):

        if fi == select:

            agA = u.select_atoms("resid "+str(resnum1))
            agB = u.select_atoms("resid "+str(resnum2))

            # Getting all parameters from MDAnalysis object
            xyzA = agA.positions
            xyzB = agB.positions
            namesA = agA.atoms.names
            namesB = agB.atoms.names
            typA = agA.atoms.types
            typB = agB.atoms.types

            # First 4 bonds aren't accurate
            bondsA = agA.atoms.bonds.to_indices()[4:]
            bondsB = agB.atoms.bonds.to_indices()[4:]
            idsA = agA.atoms.ids
            idsB = agB.atoms.ids

            if cap:
                namesA, xyzA, typA, bondsA = cu.cap_H(
                    u, xyzA, namesA, typA, bondsA, idsA, resnum1)
                namesB, xyzB, typB, bondsB = cu.cap_H(
                    u, xyzB, namesB, typB, bondsB, idsB, resnum2)

            CofMA = agA.center_of_mass()
            CofMB = agB.center_of_mass()

    return xyzA, xyzB, namesA, namesB, typA, typB, bondsA, bondsB, CofMA, CofMB


def get_coords(path, param_file, select, file_idx=None, dt=2, sel_1=['1'], sel_2=None,
               cap=True, del_list=[], cap_list=[[], []], resnames=False):
    """
    Given a selected time in a given traj file, it returns the (MDAnalysis) molecule params.    
    Parameters
    ----------
    path : String
        Location of traj files.
    tdye : String
        "I" or "II", the type of the dimer.
    select : int
        The index of the time-frame to extract.
    rep : String
        "A", "B", "C" or "D"
    Returns
    -------
    Parameters of H-capped Dimer at the selected time-frame.
    (xyz, names, atom_types)
    If sel_2!=None, each is a list of size 2: [sthA, sthB]
    """

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str

    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        select = 0
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        select = 0
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        fidx = str(file_idx) if file_idx is not None else ''
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+fidx+".nc")
        u = mda.Universe(param_file, traj_file+fidx+".nc", format='TRJ')

    for fi, ts in enumerate(u.trajectory[::dt]):
        if fi == select:
            # Getting all parameters from MDAnalysis object
            agA = u.select_atoms(make_sel(sel_1))
            xyz = [agA.positions]
            names = [agA.atoms.names]
            typ = [agA.atoms.types]
            CofM = [agA.center_of_mass()]
            resname = ['0']
            if resname:
                resname = [agA.resnames]

            if sel_2 is not None:  # A dimer
                agB = u.select_atoms(make_sel(sel_2))
                xyz += [agB.positions]
                names += [agB.atoms.names]
                typ += [agB.atoms.types]
                CofM += [agB.center_of_mass()]
                if resname:
                    resname += [agB.resnames]

            if cap:
                names, xyz, typ = cu.cap_H_general(
                    u, agA, sel_1, del_list, cap_list[0])
                xyz = [xyz]
                names = [names]
                typ = [typ]
                if sel_2 is not None:
                    namesB, xyzB, typB = cu.cap_H_general(
                        u, agB, sel_2, del_list, cap_list[1])
                    names += [namesB]
                    xyz += [xyzB]
                    typ += [typB]
                print(del_list)

    return xyz, names, typ, CofM, resname


def get_pyscf(traj_file, param, select, resnums, new_coords=None):
    """
    Given a selected time-frame, it returns the molecule's coords as a PySCF 
        formatted string.
    Parameters
    ----------
    path : String
        Location of traj files.
    tdye : String
        "I" or "II", the type of the dimer.
    select : int
        The index of the time-frame to extract.
    rep : String
        "A", "B", "C" or "D"
    new_coords : numpy array, optional
       If given, modified coordinates are used instead of those in the frame.
    Returns
    -------
    xyzA, xyzB : numpy arrays with coordinates in PySCF format
    """

    dt = 20
    time_i = select * dt

    if select < 100:

        i_file = int(time_i/250) + 1
        idx = select*2 % 25
        dt = 5
    else:

        i_file = int((time_i-2000)/1000) + 1
        idx = (select-100) % 100
        dt = 2

    u = mda.Universe(param, traj_file+str(i_file)+".nc", format="TRJ")
    for fi, ts in enumerate(u.trajectory[::dt]):

        if fi == idx:

            sel1, sel2 = resnums

            xyzA, xyzB, RAB = cu.Process_MD(
                u, sel1, sel2, coord_path="coord_files/MD_atoms", new_coords=new_coords)

    return xyzA, xyzB


def coord_transform(xyzA, xyzB, namesA, namesB, rot_angles, dr=None, assign=None):
    """ Transform given coordinates by rotation and translation. 
            Can be used for any number of molecules of two types, A and B.
    Parameters
    ----------
    xyzA, xyzB : Numpy array or list
        (natoms, 3) array with atoms positions (for A and B-type monomers).
    namesA, namesB : Numpy array or list
        (natoms,) array with atoms names (for A and B-type monomers).
    rot_angles : list of tuples
        [(x1,y1,z1), (x2,y2,z2), ..., (xN,yN,zN)] list of rotation angles
    dr : list, optional
        [[x1,y1,z1], ..., [x1,y1,z1]]. If given, indicates translation displacement
    assign : list, optional
        If given, indicates ordering ot atom types, eg., default [0,1,0,1...] 
         corresponds to N molecules with alternating types.

    Returns
    -------
    xyz_all : List of len Nmolecules with the atomic coords of each.
    atA, atB : List of atomic indexes for each molecule-type.
    atnonH_A, atnonH_B : List of indexes for non-H atoms in each molecule type.
    """
    nmols = len(rot_angles)
    if dr is None:
        dr = [0, 0, 0]*nmols
    if assign is None:
        assign = ([0, 1]*nmols)[:nmols]

    natomsA, natomsB = len(namesA), len(namesB)
    atA = np.arange(natomsA)
    atB = np.arange(natomsB)

    # list of non-H atoms
    nonHsA = np.invert(np.char.startswith(namesA.astype(str), 'H'))
    atnonH_A = atA[nonHsA]

    nonHsB = np.invert(np.char.startswith(namesB.astype(str), 'H'))
    atnonH_B = atB[nonHsB]

    mol_list = np.array([xyzA, xyzB]*len(assign))
    coord_list = mol_list[assign]

    # Loop over each molecules
    xyz_all = []
    for imol in range(nmols):

        xyz0 = coord_list[imol]

        # translate to desired position
        xyz = xyz0 + dr[imol]

        # rotate
        rx, ry, rz = rot_angles[imol]
        xyz_i = np.dot(Rz(rz), np.dot(Ry(ry), np.dot(Rx(rx), xyz.T))).T

        xyz_all.append(xyz_i)

    return xyz_all, atA, atB, atnonH_A, atnonH_B


def coord_transform_single(xyzA, xyzB, namesA, namesB, rot_angles, dr=None, del_ats=None):
    """ Transform given coordinates by rotation and translation. 
        Simplifies version for a single dimer.
    Parameters
    ----------
    xyzA, xyzB : Numpy array or list
        (natoms, 3) array with atoms positions.
    namesA, namesB : Numpy array or list
        (natoms,) array with atoms names.
    rot_angles : list of tuples
        [(x1,y1,z1), (x2,y2,z2)] list of rotation angles
    dr : list, optional
        [[x1,y1,z1], [x2,y2,z2]]. If given, indicates translation displacement
    del_ats : list, optional
        If given indicates the indexes of the atoms to delete from the molecules.
        Must be given as a list [idx_A,idx_B]

    Returns
    -------
    xyzA, xyzB : Atomic coords of A and B molecules.
    atA, atB : List of atomic indexes for A and B molecules.
    atnonH_A, atnonH_B : List of indexes for non-H atoms in each molecule.
    """

    if del_ats is not None:
        namesA = del_atoms(namesA, namesA, del_ats[1])
        namesB = del_atoms(namesB, namesB, del_ats[0])
        xyzA = del_atoms(xyzA, namesA, del_ats[1])
        xyzB = del_atoms(xyzB, namesB, del_ats[0])

    if dr is None:
        dr = [[0, 0, 0], [0, 0, 0]]

    natomsA, natomsB = len(namesA), len(namesB)
    atA = np.arange(natomsA)
    atB = np.arange(natomsB)

    # list of non-H atoms
    nonHsA = np.invert(np.char.startswith(namesA.astype(str), 'H'))
    atnonH_A = atA[nonHsA]

    nonHsB = np.invert(np.char.startswith(namesB.astype(str), 'H'))
    atnonH_B = atB[nonHsB]

    # translate to desired position
    xyzA += dr[0]
    xyzB += dr[1]

    # rotate
    rx1, ry1, rz1 = rot_angles[0]
    rx2, ry2, rz2 = rot_angles[1]
    xyz_A = np.dot(Rz(rz1), np.dot(Ry(ry1), np.dot(Rx(rx1), xyzA.T))).T
    xyz_B = np.dot(Rz(rz2), np.dot(Ry(ry2), np.dot(Rx(rx2), xyzB.T))).T

    return xyz_A, xyz_B, namesA, namesB, atA, atB, atnonH_A, atnonH_B

def del_extra_atoms(u, st_atoms=25):
    """ Helper function to extra atoms left from using a spatial MDAnalysis selections
        (i.e., those belonging to an incomplete residue)
    """

    # 1) Select nucleotides and non DNA groups
    dna = u.select_atoms('nucleic')
    non_dna = u.select_atoms('not nucleic')
    # 2) If the res is a nucleotide (give this at input), check the total number of atoms
    total_valid = 0
    invalid_res = []
    
    for nuc in dna.residues:
        # if the atoms are complete, merge. "Complete" includes having both edge oxygens.
        if len(nuc.atoms.select_atoms("name O3' or name O5'")) > 1 and len(nuc.atoms) >= st_atoms:
            if total_valid==0:
                non_dna = nuc.atoms
            else:
                non_dna = mda.Merge(non_dna.atoms, nuc.atoms)
            total_valid += 1
        else:
            invalid_res.append(nuc.resid)
    #print(f'A total of {len(dna.residues)-total_valid} incomplete residues were deleted')
    updated_unit = non_dna.select_atoms('all')

    # RETURN the updated atom selection with complete nucleotides (and use it in the scanDNA function.)
    return updated_unit


def scan_bond(molecule, fixed_a, fixed_b, fixed_length, nsamples, condition):
    """ Transform coordinates such that by rotating around a fixed_length bond between fixed_a(center) and fixed_b
        New coordinates are valid if they satisfy a given condition (function)

    Args:
        molecule (MDAnalysis AtomGroup): Molecule to transform
        fixed_a (numpy array, list): Coordinates of atom we wish to keep in place in bond (center of rot sphere)
        fixed_b (numpy array, list): Coordinates of atom we that is "rotating" in the bond
        fixed_length (Fixed bond length): The length of the bond we are scanning with resp with
        nsamples (int): How many point to sample on the sphere  
        condition (function(coords)): Functions defining condition by which transformed coords are valid (returns bool)

    Raises:
        ValueError: When none of the scanned tranformed coordinates were valid

    Returns:
        Numpy array: New transformed coordinates
    """

    # Define a sphere centered at fixed_a and whose radius is the length of the fixed bond
    # scan nsamples points in the surface of the sphere and define tvector wrt the fixed bond
    # then translate the entire molecule
    import random
    increment = 2*pi/nsamples
    offset = pi/nsamples

    # We randomize the samples to increase the chance of finding a valid transformation fast
    samples = random.sample(range(nsamples), nsamples)
    origin = fixed_a

    for i in range(nsamples):
        theta = i * increment
        for j in range(nsamples):
            phi = j * increment + offset
            sphere = np.array([sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)])
            sample_point = origin + fixed_length * sphere

            new_molecule = rotate_bond(molecule, fixed_a, fixed_b, sample_point)      

            #tvector = sample_point - fixed_b
            #new_molecule = molecule.atoms.translate(tvector)

            '''
            ref = fixed_a
            ref.atoms.positions = [origin,sample_point]
            new_molecule = align_to_mol(molecule, ref, path_save=None)
            '''

            # Test if configuration is valid
            if condition(new_molecule):
                return new_molecule

    #raise ValueError(
    print(f"A configuration wasn't found for the bond scan with {nsamples} samples. Increase nsamples!!")
    return new_molecule

def write_leap(mol2_dye, mol2_link, pdb, frcmod, file_save, dye_res='DYE', link_res='LNK',
               add_ions=None, make_bond=None, remove=None, water_box=20.0):
    """ Helper function to write tleap file

    Args:
        mol2_dye (str): Path to mol2 file of dye.
        mol2_link (str): Path to mol2 file of linker.
        pdb (str): Path to pdb file for dye+linker.
        frcmod (str): Path to frcmod file of dye+liner.
        file_save (str): Path to save LEaP input file.
        dye_res (str, optional): Dye's residue name. Defaults to 'DYE'.
        link_res (str, optional): Linker's residue name. Defaults to 'LNK'.
        add_ions (list, optional): Adding ions to neutralize charge ['ion', charge]. 
        make_bond (list, optional): Info on atoms that participate in dye-linker bonding.
        remove (list, optional): List of atoms to remove from mol object. Defaults to None.
        water_box (float, optional): Size of water box in Ams. Defaults to 20.0.
    """
    fprefix = pdb[:-3]
    with open(file_save, 'w') as f:
        f.write("source leaprc.DNA.OL15\n")
        f.write("source leaprc.gaff\n")
        f.write("source leaprc.gaff2\n")
        f.write("source leaprc.water.tip3p\n")
        f.write(f"{dye_res} = loadmol2 {mol2_dye} \n")
        if mol2_link:
            f.write(f"{link_res} = loadmol2 {mol2_link} \n")
        f.write(f"loadAmberParams {frcmod} \n")
        f.write(f"mol = loadpdb {pdb} \n")
        if make_bond is not None:
            for m in make_bond:
                f.write(f"bond mol.{m[0]} mol.{m[1]} \n")
        if remove is not None:
            for r in remove:
                f.write(f"remove mol mol.{r} \n")
        if add_ions is not None:
            f.write(f"addIons mol {add_ions[0]} {add_ions[1]} \n")
        f.write(f"solvatebox mol TIP3PBOX {water_box}\n")
        f.write(f"saveAmberParm mol {fprefix}prmtop {fprefix}rst7\n")
        f.write("quit")

    return


def join_pdb(pdb_st, pdb_mob, del_st, del_mob, bond_st, bond_mob, path_save, path_bond_info=None):
    """Joining dye/linker pdb files with MDAnalysis by moving pdb_mob to pdb_st

    Args:
        pdb_st (str): Path to PDB of static molecule.
        pdb_mob (str): Path to PDB of mobile molecule.
        del_st (list): List of IDs to delete in pdb_st.
        del_mob (list): List of IDs to delete in pdb_mob.
        bond_st (list): Atoms (ID) participating on the bond from pdb_st [keep, replace]*num_links.
        bond_mob (list): Atoms (ID) participating on the bond from pdb_mob.
        path_save (str): Path for final PDB.
        path_bond_info (str, optional): Path to save bond information. Default is not saving.         

    Returns:
        list: Bond information
    """

    # Get coordinates and label info   xyz, names, typ, CofM, resname
    xyzm, namesm, typm, CofMm, reslist = get_coords(pdb_mob, None, 0, sel_1=['1'], sel_2=None, cap=False)
    xyzs, namess, typs, CofMs, reslist = get_coords(pdb_st, None, 0, sel_1=['1'], sel_2=None, cap=False)
    xyzm, xyzs = xyzm[0], xyzs[0]
    namesm = namesm[0] 
    namess = namess[0] 
    typm, typs = typm[0], typs[0]
    CofMm, CofMs = CofMm[0], CofMs[0]

    # MDAnalysis atom indexes are zero-based
    del_st  = [i-1 for i in del_st]
    del_mob = [i-1 for i in del_mob]

    typm = del_atoms(typm, namesm, del_mob)
    res_names = ['DYE']
    if isinstance(bond_st[1], int):
        xyzm, CofMm = align_two_molecules_at_one_point(xyzm, xyzs[bond_st[1]-1], xyzm[bond_mob-1], 
                                                       CofMs, CofMm, align=1, accuracy=0.9)
        xyzm = del_atoms(xyzm, namesm, del_mob)
        xyzm = [xyzm]
        typm = [typm]
        res_names += ['LNK']
        # Saving info on atoms that participate in bonding (before deletion)
        bond_info = [namess[bond_st[0]-1], namesm[bond_mob-1]]
        namesm = del_atoms(namesm, namesm, del_mob)
        namesm = [namesm]

    else:
        xyzm1, CofMm1 = align_two_molecules_at_one_point(xyzm, xyzs[bond_st[0][1]-1], xyzm[bond_mob-1], 
                                                         CofMs, CofMm, align=1, accuracy=0.9)
        xyzm2, CofMm2 = align_two_molecules_at_one_point(xyzm, xyzs[bond_st[1][1]-1], xyzm[bond_mob-1],
                                                         CofMs, CofMm, align=1, accuracy=0.9)
        print('bonds',namesm[bond_mob-1], namess[bond_st[0][1]-1], namess[bond_st[1][1]-1])
        xyzm1 = del_atoms(xyzm1, namesm, del_mob)
        xyzm2 = del_atoms(xyzm2, namesm, del_mob)
        xyzm = [xyzm1, xyzm2]
        typm = [typm]*2
        res_names += ['LNK','LNK']
        # Saving info on atoms that participate in A and B bonding (before deletion)
        bond_info = [namess[bond_st[0][0]-1], namesm[bond_mob-1], namess[bond_st[1][0]-1], namesm[bond_mob-1]]
        namesm = del_atoms(namesm, namesm, del_mob)
        namesm = [namesm]*2    

    typs = del_atoms(typs, namess, del_st)
    xyzs = del_atoms(xyzs, namess, del_st)
    namess = del_atoms(namess, namess, del_st)
    typ = [typs] + typm 
    xyz = [xyzs] + xyzm
    names = [namess] + namesm
    

    if path_bond_info is not None:
        np.savetxt(path_bond_info, bond_info, fmt="%s")

    # Save final pdb
    dimer_mol = get_mol(xyz, names, typ, res_names=res_names)

    # save pdb
    dimer_all = dimer_mol.select_atoms('all')
    dimer_all.write(path_save)
    return bond_info


def find_idx(names, to_find, which):
    """Find indexes corresponding to given list of some characteristic
        e.g., names, coordinates, ...

    Args:
        names (ndarray): List to scan.
        to_find (ndarray): List of variables to find in names.
        which (str): String for print statement

    Returns:
        ndarray: List of indexes of found items.
    """
    if type(to_find[0] is int):
        return to_find

    try:
        datom_idx = [np.nonzero(names == n)[0][0] for n in to_find]
    except:
        datom_idx = []
        print(
            f"The {to_find} couldn't be found. No atoms will be deleted in {which} molecule")
    return datom_idx


def del_atoms(var, ref, del_n):
    """Delete an atom from a variable var, given list of items from ref category

    Args:
        var (ndarray): List from which we are going to delete items.
        ref (ndarray): List with atoms to compare with del_n
        del_n (list, int): List of indexes to compare with ref, and delete from var.

    Returns:
        ndarray: New variable
    """
    del_idx = find_idx(ref, del_n, '')
    print('Deleting!',del_idx, ref[del_idx])
    var_new = np.delete(var, del_idx, 0)
    return var_new


def atom_dist(a1, a2, coord1, coord2):
    """
    Distance between two atoms
    Parameters
    ----------
    a1 : int
        index of atom 1.
    a2 : int
        index of atom 2.
    coord1 : ndarray
        array listing molecule's #1 coordinaties.
    coord2 : TYPE
        array listing molecule's #2 coordinaties.
    Returns
    -------
    dist : float
        in Amstrongs
    """
    dist = LA.norm(coord2[a2] - coord1[a1])
    return dist


def check_steric(pos1, pos2, at1, at2, atH1, atH2):
    """_summary_

    Args:
        pos1 (numpy ndarray): Coordinates of mol 1
        pos2 (numpy ndarray): Coordinates of mol 2
        at1 (numpy ndarray): Atom indexes mol1
        at2 (numpy ndarray): Atom indexes mol2
        atH1 (numpy ndarray): Heavy atom indexes mol1
        atH2 (numpy ndarray): Heavy atom indexes mol2

    Returns:
        bool: If a steric clashes between non-H atoms are found
    """
    from scipy.spatial.distance import cdist

    if len(atH1) == 0 or len(atH2) == 0:
        return False

    atH1 -= at1[0]
    atH2 -= at2[0]

    at1 -= at1[0]
    at2 -= at2[0]

    # distsum = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in at2-1] for a1 in at1-1]) #array (natoms1 x natoms2)
    distsum = cdist(pos1, pos2)

    # Distance between non-H atoms
    # distsum_noH = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in atH2-1] for a1 in atH1-1]) #array (natoms1 x natoms2)
    pos_nH1 = pos1[atH1]
    pos_nH2 = pos2[atH2]
    distsum_noH = cdist(pos_nH1, pos_nH2)

    # Test if atoms are too close together (approx fails)
    # if np.count_nonzero((np.abs(distsum_noH) < 2.0)) > 5 or np.any((np.abs(distsum_noH) < 1.0)):
    if np.any((np.abs(distsum_noH) < 1.5)):  # molprobity clash score
        #print('tot = ', distsum[np.abs(distsum) < 2.0],', from:', len(at1),len(at2))
        return True # Means there's a clash
        # print(np.count_nonzero((np.abs(distsum_noH) < 2.0)))
    else:
        return False


def multipole_coup(pos1, pos2, ch1, ch2, at1, at2, atH1, atH2):
    """
    Calculates multiple coupling from inter-atomic distance and atomic excited
        state partial charges
    Parameters
    ----------
    pos1, pos2 : ndarray
        cartesian coord of atoms in molecule 1 and 2.
    ch1, ch2 : ndarray
        array with Lowdin partial charges from tddft for molecules 1 and 2.
    at1, at2 : ndarray
        list of indexes of atoms in molecule 1 and 2.
    Returns
    -------
    Vij : (float) mulitipole coupling
    """
    from scipy.spatial.distance import cdist

    atH1 -= at1[0]
    atH2 -= at2[0]

    at1 -= at1[0]
    at2 -= at2[0]

    # distsum = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in at2-1] for a1 in at1-1]) #array (natoms1 x natoms2)
    distsum = cdist(pos1, pos2)

    # Distance between non-H atoms
    # distsum_noH = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in atH2-1] for a1 in atH1-1]) #array (natoms1 x natoms2)
    pos_nH1 = pos1[atH1]
    pos_nH2 = pos2[atH2]
    distsum_noH = cdist(pos_nH1, pos_nH2)

    # Test if atoms are too close together (approx fails)
    if np.count_nonzero((np.abs(distsum_noH) < 2.0)) > 5 or np.any((np.abs(distsum_noH) < 1.0)):
        #print('tot = ', distsum[np.abs(distsum) < 2.0],', from:', len(at1),len(at2))
        Vij = 9999999999999999
        # print(np.count_nonzero((np.abs(distsum_noH) < 2.0)))
    else:
        Vij = np.sum(np.outer(ch1, ch2)/distsum)
        # np.sum( np.multiply(np.outer(ch1,ch2),1/distsum) ) #SUM_{f,g}[ (qf qg)/|rf-rg| ]
        #print('!!',np.count_nonzero((np.abs(distsum_noH) < 2.0)), 'from:', len(at1),len(at2))

    return Vij


def get_mol(coords, names, types, res_names, segname='1', res_ids=None):
    """ 
    Creates new MDAnalysis object based on given paramenters
    Parameters
    ----------
    coords : list
        List of NumPy arrays/lists of length Nmolecules with atomic coordinates.
    names : list
        List of NumPy arrays/lists of length Nmolecules with atomic names.
    types : list
        List of NumPy arrays/lists of length Nmolecules with atomic types.
    bondsA : list
        List of NumPy arrays/lists of length Nmolecules with atomic bonds.
    res_names : list
        List of strings with residue names.
    segname : string
    res_ids : list
    Returns
    -------
    mol_new : MDAnalisys.AtomGroup
        Transformed molecule object.
    """
    if not len(coords) == len(names) == len(types):  # == len(res_names):
        raise ValueError("All input arrays must be of length Nmolecules")

    n_residues = len(res_names)
    n_mols = len(names)
    # Creating new molecules
    resids0 = []
    natoms = 0
    if res_ids is None:
        for imol in range(n_mols):
            natom = len(names[imol])
            resid = [imol]*natom
            resids0.append(resid)
            #natoms += natom
        resids = np.concatenate(tuple(resids0))
    else:
        resids = res_ids
    natoms = 0
    for imol in range(len(names)):
        natoms += len(names[imol]) 

    assert len(resids) == natoms
    segindices = [0] * n_residues

    atnames = np.concatenate(names, axis=0)#tuple(names))
    attypes = np.concatenate(types, axis=0)#tuple(types))
    # if isinstance(res_names[0], str):
    resnames = res_names
    # else:
    #resnames = np.concatenate(tuple(res_names))

    mol_new = mda.Universe.empty(natoms,
                                 n_residues=n_residues,
                                 atom_resindex=resids,
                                 residue_segindex=segindices,
                                 trajectory=True)

    mol_new.add_TopologyAttr('name', atnames)
    mol_new.add_TopologyAttr('type', attypes)
    mol_new.add_TopologyAttr('resname', resnames)
    mol_new.add_TopologyAttr('resid', list(range(1, n_residues+1)))
    mol_new.add_TopologyAttr('segid', [segname])
    mol_new.add_TopologyAttr('id', list(range(natoms)))
    mol_new.add_TopologyAttr('record_types', ['HETATM']*natoms)

    # Adding positions
    coord_array = np.concatenate(coords, axis=0) #np.concatenate(tuple(coords))
    assert coord_array.shape == (natoms, 3)
    mol_new.atoms.positions = coord_array

    # Adding bonds
    """
    n_acum = 0
    all_bonds = []
   
    for imol in range(n_residues):
        bond = bonds[imol] - np.min(bonds[imol])
        d_bd = [bond[i] + n_acum for i in range(len(bond))]
        all_bonds.append(d_bd)
        n_acum += len(names[imol])
    """

    #bonds0 = np.concatenate(tuple(all_bonds),axis=0)
    #atbonds = list(map(tuple, bonds0))

    #mol_new.add_TopologyAttr('bonds', atbonds)

    return mol_new


def get_charges(u, pos_ion, neg_ion):
    """
    Given a trajectory frame, it returns the coordinates of CofM
    for all charged elements (DNA residues and ions) in the environment.
    This function is to be called inside a trajectory loop.
    Parameters
    ----------
    u : MDAnalysis universe
        Traj frame
    pos_ion: Tuple (str, int)
        (resname, charge) of the positive solvent ion 
    neg_ion: Tuple (str, int)
        (resname, charge) of the negative solvent ion 
    Returns
    -------
    Coordinates of charged elements as (coord, charge) tuples: 
    (DNA, pos ions, neg ions)
    """
    pos_name, pos_charge = pos_ion
    neg_name, neg_charge = neg_ion

    dna_res = u.select_atoms("nucleic")
    dna_coords = []
    dna_charges = np.array([-1.0] * dna_res.residues.n_residues)
    for res in dna_res.residues:
        cofm = res.atoms.center_of_geometry()
        dna_coords.append(cofm)

    pos_res = u.select_atoms("resname " + pos_name)
    pos_coords = []
    pos_charges = np.array([pos_charge] * pos_res.residues.n_residues)
    for res in pos_res.residues:
        cofm = res.atoms.center_of_geometry()
        pos_coords.append(cofm)

    neg_res = u.select_atoms("resname " + neg_name)
    neg_coords = []
    neg_charges = np.array([neg_charge] * neg_res.residues.n_residues)
    for res in neg_res.residues:
        cofm = res.atoms.center_of_geometry()
        neg_coords.append(cofm)

    coords = np.concatenate((dna_coords, pos_coords, neg_coords), axis=0)
    charges = np.concatenate((dna_charges, pos_charges, neg_charges), axis=0)
    return coords, charges


def solvent_coords(path, tdye, select, rep):
    """
    Given a selected time-frame, it returns the (MDAnalysis) molecule params.    
    Parameters
    ----------
    path : String
        Location of traj files.
    tdye : String
        "I" or "II", the type of the dimer.
    select : int
        The index of the time-frame to extract.
    rep : String
        "A", "B", "C" or "D"
    Returns
    -------
    Parameters of H-capped Dimer at the selected time-frame.
    Format is: ( xyzA, xyzB, namesA, namesB, atom_typesA, atom_typesB, 
                 [list of bonds]A, [list of bonds]B )
    """

    typ = 'Opp' if tdye == 'II' else ''
    if rep == 'C':
        def param_files(ty): return 'Sq' + ty + '_dimer_g1.prmtop'
    elif rep == 'D':
        def param_files(ty): return 'Sq' + ty + '_dimer_g2.prmtop'
    else:
        def param_files(ty): return 'Sq' + ty + '_dimer.prmtop'

    dt_both = 20
    time_i = select * dt_both

    if select < 100:
        i_file = int(time_i/250) + 1
        idx = select*2 % 25
        dt = 5
    else:
        last_file = 8
        i_file = int((time_i-2000)/1000) + 1 + last_file
        idx = (select-100) % 100
        dt = 2

    u = mda.Universe("%s/%s" % (path, param_files(typ)),
                     "%s/prod/Sq%sDim_prod%s" % (path, typ, rep) + str(i_file) + ".nc", format="TRJ")
    print("The param file is: %s/%s \n" % (path, param_files(typ)),
          "And the traj files is: %s/prod/Sq%sDim_prod%s" % (path, typ, rep) + str(i_file) + ".nc")
    for fi, ts in enumerate(u.trajectory[::dt]):

        if fi == idx:
            t_i = round((ts.frame*10), 2)
            print("The time-step is: ", t_i)

            coord, charge = get_charges(u, ("MG", 2.0), ("Cl-", -1.0))

    return coord, charge


def get_pdb(traj_path, param_path, path_save, resnums, select=(0, 0), dt=2, MDA_selection='all',
            del_list=[], cap_list=[[], []], resnames=['MOA', 'MOB'], mol_name='ABC'):

    i_file, idx = select

    xyz, names, type, __, __ = get_coords(traj_path, param_path,
                                          idx, file_idx=i_file, dt=dt,
                                          sel_1=resnums[0], sel_2=resnums[1],
                                          del_list=del_list, cap_list=cap_list)
    xyza, xyzb = xyz[0], xyz[1]
    namesA, namesB = names[0], names[1]
    typeA, typeB = type[0], type[1]
    orig_mol = get_mol([xyza, xyzb], [namesA, namesB], [
                       typeA, typeB], res_names=resnames, segname=mol_name)

    # save pdb
    orig_all = orig_mol.select_atoms(MDA_selection)
    orig_all.write(path_save)
    print("Saved PDB successfully as", path_save)
    return orig_all


def max_pdb(Vi, traj_path, param_path, resnums, path_save, sel='all'):
    """
    Generate a pdb of the dimer coordinates at max VFE/VCT
    Returns
    -------
    None.
    """

    max_V = np.argmax(Vi)
    print(max_V)

    dt_both = 20
    time_i = max_V * dt_both

    '''
    if max_V < 100:
        i_file = int(time_i/250) + 1
        idx = max_V*2 % 25
        dt = 5
    
    else:
    '''
    last_file = 6
    i_file = int((time_i-2000)/1000) + 1 + last_file
    idx = (max_V-100) % 100
    dt = 1

    obj = get_pdb(traj_path, param_path, path_save, resnums, select=(i_file, idx),
                  dt=dt, MDA_selection=sel)

    return obj


def COM_atom(COM, xyz, names):
    diff = abs(xyz - COM)
    diff_glob = np.mean(diff, axis=1)
    closer_idx = np.argmin(diff_glob)
    print(diff.shape)

    # We are not interested in H or O
    while names[closer_idx][0] == 'H' or names[closer_idx][0] == 'O':
        print(names[closer_idx])
        diff = np.delete(diff, closer_idx, axis=0)
        diff_glob = np.mean(diff, axis=1)
        closer_idx = np.argmin(diff_glob, axis=0)

    return closer_idx, xyz[closer_idx], names[closer_idx]


def energy_diagram(file_list, global_min, time):
    eV_conv = 27.211399
    energies = np.empty((0, 2))
    for f in file_list:
        data = np.loadtxt(f)
        energies = np.concatenate((energies, data))

    energies = eV_conv*(energies-global_min)

    return energies


def displaced_dimer(sel1, sel2, cof_dist, disp,
                    atom_orig='N1', atom_z='N2', atom_y='C2',
                    res_names=['SQA', 'SQB']):

    xyz1 = sel1.positions
    xyz2 = sel2.positions

    idx1A = np.nonzero(sel1.atoms.names == atom_orig)[0][0]
    idx1B = np.nonzero(sel1.atoms.names == atom_z)[0][0]
    idx1C = np.nonzero(sel1.atoms.names == atom_y)[0][0]
    idx2A = np.nonzero(sel2.atoms.names == atom_orig)[0][0]
    idx2B = np.nonzero(sel2.atoms.names == atom_z)[0][0]
    idx2C = np.nonzero(sel2.atoms.names == atom_y)[0][0]

    x_unit = np.array([1, 0, 0]).reshape(1, -1)
    y_unit = np.array([0, 1, 0]).reshape(1, -1)
    z_unit = np.array([0, 0, 1]).reshape(1, -1)

    # Rodrigues formula
    def rodrigues(xyz, phi, v_unit):

        xyz_new = ((xyz.T*np.cos(phi)).T + np.cross(v_unit[0, :], xyz.T, axis=0).T * np.sin(phi)
                   + (v_unit * np.dot(v_unit, xyz.T).reshape(-1, 1)) * (1-np.cos(phi)))

        return xyz_new

    # placing atomA of both molecules in the origin
    xyz1 -= xyz1[idx1A]
    xyz2 -= xyz2[idx2A]

    # rotating molecule to yz plane
    rot1 = xyz1[idx1B]
    #phiz1 = - np.arccos(rot1[2]/LA.norm(rot1))
    num = rot1[1]*np.sqrt(rot1[0]**2+rot1[1]**2) + rot1[2]**2
    theta1 = -np.arccos(num/LA.norm(rot1)**2)
    xyz1_new = rodrigues(xyz1, theta1, z_unit)

    # rotating to z axis
    rot1 = xyz1_new[idx1B]
    phi1 = np.arccos(rot1[2]/LA.norm(rot1))
    xyz1_new = rodrigues(xyz1_new, phi1, y_unit)

    # Rotating the other atom axis
    rot1 = xyz1_new[idx1C]
    psi1 = np.arccos(rot1[1]/LA.norm(rot1))
    xyz1_new = rodrigues(xyz1_new, psi1, x_unit)

    ###
    rot2 = xyz2[idx2B]
    num = rot2[1]*np.sqrt(rot2[0]**2+rot2[1]**2) + rot2[2]**2
    theta2 = -np.arccos(num/LA.norm(rot2)**2)
    xyz2_new = rodrigues(xyz2, theta2, z_unit)

    # rotating to z axis
    rot2 = xyz2_new[idx2B]
    phi2 = np.arccos(rot2[2]/LA.norm(rot2))
    xyz2_new = rodrigues(xyz2_new, phi2, y_unit)

    # Rotating the other atom axis
    rot2 = xyz2_new[idx2C]
    psi2 = np.arccos(rot2[1]/LA.norm(rot2))
    xyz2_new = rodrigues(xyz2_new, psi2, x_unit)

    #xyz2_new -= xyz2_new[idx2A]

    # displacing sel2 on the x axis only
    xyz2_new[:, 0] += cof_dist

    # displacing sel2 on y axis
    xyz2_new[:, 2] += disp

    # create new MDA object
    mol = get_mol([xyz1_new, xyz2_new], [sel1.atoms.names, sel2.atoms.names],
                  [sel1.atoms.types, sel2.atoms.types], [
                      sel1.atoms.bonds, sel2.atoms.bonds],
                  res_names=res_names)

    # print(rot1[2]*LA.norm(rot1))

    return mol





def calc_displacement(path, param_file, atom1, atom2, file_idx=None, dt=2, sel_1=['11'], sel_2=['12']):

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str

    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        fidx = str(file_idx) if file_idx is not None else ''
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+fidx+".nc")
        u = mda.Universe(param_file, traj_file+fidx+".nc")

    for fi, ts in enumerate(u.trajectory[::dt]):

        agA = u.select_atoms(make_sel(sel_1) + " and name " + atom1)
        agB = u.select_atoms(make_sel(sel_2) + " and name " + atom1)

        # Calculating displacement
        xyzA = agA.positions
        xyzB = agB.positions
        disp = abs(LA.norm(xyzA - xyzB))

    return disp


def pa_angle(molA, molB, pa=2):
    from numpy import pi
    mol1 = recenter_mol(molA, align_vec=None)
    mol2 = recenter_mol(molB, align_vec=None)
    paA = mol1.principal_axes()[pa]
    paB = mol2.principal_axes()[pa]
    thetaAB = np.arccos(np.dot(paA, paB)/(LA.norm(paA)*LA.norm(paB)))
    if thetaAB/pi > 0.5:
        thetaAB = pi - thetaAB

    return thetaAB


def calc_angle(path, param_file, file_idx=None, dt=2, sel_1=['11'], sel_2=['12'], pa=2):

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str

    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        fidx = str(file_idx) if file_idx is not None else ''
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+fidx+".nc")
        u = mda.Universe(param_file, traj_file+fidx+".nc")

    angles = []
    for fi, ts in enumerate(u.trajectory[::dt]):

        agA = u.select_atoms(make_sel(sel_1))
        agB = u.select_atoms(make_sel(sel_2))

        # Calculating angle
        thetaAB = pa_angle(agA, agB, pa)
        angles.append(thetaAB)

    return np.array(angles)


def recenter_mol(mol, align_vec=[1, 0, 0]):
    '''
    Re centers molecule to the origin and aligns its long axis.
    Parameters
    ----------
    mol : MDAnalysis object
    align_vec : array, optional
        vector to align long axis of mol to. The default is [1,0,0]: the x axis.

    Returns
    -------
    mol : TYPE
        DESCRIPTION.
    '''
    xyz = mol.atoms.positions
    cofm = mol.center_of_mass()
    new_xyz = xyz - cofm
    mol.atoms.positions = new_xyz

    if align_vec is not None:
        paxis = 2  # Aligning the long axis of the molecule
        mol = mol.align_principal_axis(paxis, align_vec)
    return mol


def align_to_mol(mol, ref, path_save=None):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    if type(mol) is str:
        mol = mda.Universe(mol, format="PDB")
    if type(ref) is str:
        ref = mda.Universe(ref, format="PDB")
    align.alignto(mol, ref, select="all", weights="mass")

    mol = mol.select_atoms('all')
    if path_save:
        mol.write(path_save)
    return mol

def rotation_to_vector(mobile_vector, ref_vector):
    """Returns the rotation matrix required to align mobile_vector with ref_vector

    Args:
        mobile_vector (ndarray): vector to be rotated
        ref_vector (ndarray): reference vector
    """

    from scipy.spatial.transform import Rotation as R

    # Normalize 
    mobile_vector = mobile_vector / LA.norm(mobile_vector)
    ref_vector = ref_vector / LA.norm(ref_vector)

    # Axis of rotation
    k = np.cross(ref_vector, mobile_vector)
    k /= LA.norm(k)

    # Angle of rotation
    dot = np.dot(ref_vector, mobile_vector)
    angle = np.arccos(dot) # Given the vectors are normalized

    rotvec = -k*angle
    rotation_mat = R.from_rotvec(rotvec)

    return rotation_mat

def align_two_molecules_at_one_point(mol, pt1, pt2, com1, com2, align=-1, accuracy=0.5):

    from scipy.spatial.transform import Rotation as R

    # rotate mol to align line segments: Find R such that pta2 vector aligns with pta1 vector    
    ## Axis of rotation
    def rotate_around_segment(mol, pt, com_mol, com_fixed):
        axis1 = np.cross(com_mol, com_fixed)
        axis1 /= LA.norm(axis1)
        dot = np.dot(com_mol, com_fixed)
        angle2 = pi - np.arccos(dot/(LA.norm(com_fixed)*LA.norm(com_mol))) 
        rotvec2 = axis1*angle2 
        R2 = R.from_rotvec(rotvec2)
        mol_shifted = R2.apply(mol)
        com_shifted = R2.apply(com_mol)
        pt_shifted = R2.apply(pt)
        return mol_shifted, com_shifted, pt_shifted

    # The com1 and com2 vectors point in opp directions if the dot product 
    #  between the normalized vetors is approx 1, so we have to repeat the rotation until that's true.

    attempts = 0
    mol_shifted, com2_shifted, pt2_shifted = rotate_around_segment(mol, pt2, com2, com1)
    while align*np.dot(com2_shifted/LA.norm(com2_shifted), com1/LA.norm(com1))<accuracy and attempts<10:
        mol_shifted, com2_shifted, pt2_shifted = rotate_around_segment(mol_shifted, pt2_shifted, com2_shifted, com1)
        attempts += 1

    # translate mol to mol1

    disp = -pt2_shifted + pt1
    mol_shifted += disp
    com2_shifted += disp
    pt2_shifted += disp

    return mol_shifted, com2_shifted

def align_two_molecules_at_two_points(mol, pt1a, pt1b, pt2a, pt2b, com1, com2, align="apar"):
    """ Align mol so that the line segment (pt1a, pt1b) in the frame of reference for the mol
    overlaps with the line segment (pt2a, pt2b) in the frame of reference for an static molecule.
    The RMS distance between the alignment points is minimized,
    & the distance between the COM of each molecule is maximized.

    Args:
        mol  (numpy array): coordinates of mol we wich to align
        pt1a (numpy array): coordinates of 1st attachement target point
        pt1b (numpy array): coordinates of 2nd attachement target point
        pt2a (numpy array): coordinates of 1st attachement mobile point
        pt2b (numpy array): coordinates of 2nd attachement mobile point
        com1 (numpy array): center of masss of static object
        com2 (numpy array): center of masss of mobile object
        align (string): "apar" antiparallel align, "orth"

    Returns:
        numpy array: aligned coordinates of mol2
    """

    from scipy.spatial.transform import Rotation as R

    # translate line segments to the origin
    shift1 = 0.5*(pt1a + pt1b) #midpoint
    shift2 = 0.5*(pt2a + pt2b)
    pt1a_shifted = pt1a - shift1
    pt2a_shifted = pt2a - shift2
    com1_shifted = com1 - shift1
    com2_shifted = com2 - shift2
    mol_shifted = mol - shift2

    # rotate mol to align line segments: Find R such that pta2 vector aligns with pta1 vector
    ## Axis of rotation
    axis1 = np.cross(pt1a_shifted, pt2a_shifted)
    axis1 /= LA.norm(axis1)
    dot1 = np.dot(pt1a_shifted, pt2a_shifted)
    angle1 = np.arccos(dot1/(LA.norm(pt1a_shifted)*LA.norm(pt2a_shifted)))

    if align == "apar":
        # Calculate angle between the two segments: Formula α = arccos[(a · b) / (|a| * |b|)] 
        rotvec1 = -axis1 * angle1
    elif align == "orth":
        angle1 = np.radians(90)  # 90 degrees in radians
        rotvec1 = axis1 * angle1
    else:
        raise ValueError("align value must be 'apar' ot 'orth'")
    rotvec1 = -axis1 * angle1
    R1 = R.from_rotvec(rotvec1)
    # Rotate molecule 2
    mol_shifted = R1.apply(mol_shifted)
    pt2a_shifted = R1.apply(pt2a_shifted)
    com2_shifted = R1.apply(com2_shifted)

    # rotate mol around the pta2 axis, such that COM2 is aligned opposite to COM1 (minimizes steric)
    axis2 = pt2a_shifted/LA.norm(pt2a_shifted)
    eq1 = com1_shifted - axis2*np.dot(axis2,com1_shifted)
    eq2 = com2_shifted - axis2*np.dot(axis2,com2_shifted)
    axis3 = np.cross(eq1, eq2)
    axis3 /= LA.norm(axis3)
    dot2 = np.dot(eq1,eq2)
    angle2 = pi - np.arccos(dot2/(LA.norm(eq1)*LA.norm(eq2))) # antiparallel alignment
    if align == "apar":    
        rotvec2 = axis3*angle2
    elif align == "orth":
        angle2 = np.radians(90)  # 90 degrees in radians
        rotvec2 = -axis3*angle2
    else:
        raise ValueError("align value must be 'apar' ot 'orth'")
    R2 = R.from_rotvec(rotvec2)
    mol_shifted = R2.apply(mol_shifted)
    com2_shifted = R2.apply(com2_shifted)

    # translate mol to mol1 & adjust COM2 to be compatible
    mol_shifted += shift1
    com2_shifted += shift1

    return mol_shifted, com2_shifted

def rotate_bond(molecule, fixed_atom, mobile_atom, target_atom):
    """ Rotates a molecule wrt to a bond, such that mobile_atom is aligned with target_atom

    Args:
        molecule (MDAnalysis Universe): MDAnalysis object with molecule.
        fixed_atom (ndarray): Coordinates of atom in the bond to remain fixed in rotation
        mobile_atom (ndarray): Coordinates of atom we want to move to target
        target_atom (ndarray): Coordinates of target atom
    """
    # Calculate the vectors
    mobile_vector = mobile_atom - fixed_atom
    reference_vector = target_atom - fixed_atom

    rot_mat = rotation_to_vector(mobile_vector, reference_vector)

    molecule.atoms.rotate(rot_mat)

    return molecule


def displace_dimer(mol1, mol2, inter_dist, h_disp, v_disp, path_save,
                   inter_ax=1, long_axis=0, short_ax=2, resnames=['A', 'B'], mol_name='ABC'):

    align_vec = [0]*3
    align_vec[long_axis] = 1
    # Re-center to origin and align paxis to given axis
    mol1_0 = recenter_mol(mol1, align_vec=align_vec)
    mol2_0 = recenter_mol(mol2, align_vec=align_vec)

    xyz1 = mol1_0.atoms.positions
    xyz2 = mol2_0.atoms.positions
    names1 = mol1_0.atoms.names
    names2 = mol2_0.atoms.names
    types1 = mol1_0.atoms.types
    types2 = mol2_0.atoms.types

    # Distance between molecules
    xyz2[:, inter_ax] += inter_dist
    # Vertical displacement
    xyz2[:, short_ax] += v_disp
    # Horizontal displacement
    xyz2[:, long_axis] += h_disp

    new_mol = get_mol([xyz1, xyz2], [names1, names2], [types1, types2],
                      res_names=resnames, segname=mol_name)
    dimer = new_mol.select_atoms('all')
    dimer.write(path_save)

    return dimer


def rotation_mat(angle, rot_axis):
    from numpy import sin, cos
    '''
    Rx = np.array([[1, 0, 0],
                   [0, cos(angle), sin(angle)],
                   [0,-sin(angle), cos(angle)]])
    Ry = np.array([[cos(angle),0,-sin(angle)],
                   [0, 1, 0],
                   [sin(angle),0, cos(angle)]])
    Rz = np.array([[cos(angle),sin(angle) ,0],
                   [-sin(angle), cos(angle), 0],
                   [0 ,0, 1]])
    '''
    if rot_axis == 'x':
        return Rx(angle)
    elif rot_axis == 'y':
        return Ry(angle)
    else:
        return Rz(angle)


def vec_two_atoms(u, sel, atom1, atom2):
    ag1 = u.select_atoms(f"resid {sel} and name {atom1}")
    ag2 = u.select_atoms(f"resid {sel} and name {atom2}")
    p1 = np.array(ag1.positions[0])
    p2 = np.array(ag2.positions[0])
    return p1-p2, p1, p2

def plane_three_atoms(u, sel, atom1, atom2, atom3):
    ''' Given three atoms, calculate a plane
    '''
    # Get coordinates
    ag1 = u.select_atoms(f"resid {sel} and name {atom1}")
    ag2 = u.select_atoms(f"resid {sel} and name {atom2}")
    ag3 = u.select_atoms(f"resid {sel} and name {atom3}")
    p1 = np.array(ag1.positions[0])
    p2 = np.array(ag2.positions[0])
    p3 = np.array(ag3.positions[0])
    # Calculate two vectors on the plane
    vec1 = p2 - p1
    vec2 = p3 - p1
    # Calculate normal 
    normal_vec = np.cross(vec1, vec2)
    
    return normal_vec, vec1, p1, p2, p3

def angle_two_vectors(u1, u2, selA, selB, atoms_vec=['N', 'N1']):
    '''
       Formula α = arccos[(a · b) / (|a| * |b|)] 
    '''

    vecA, _, _ = vec_two_atoms(u1, selA, atoms_vec[0], atoms_vec[1])
    vecB, _, _ = vec_two_atoms(u2, selB, atoms_vec[0], atoms_vec[1])

    dot = np.dot(vecA, vecB)
    mult = LA.norm(vecA) * LA.norm(vecB)
    angle = np.arccos(dot/mult)

    return angle

def angle_two_vectors_in_plane(u1, u2, selA, selB, atoms_vec=['N', 'N1', 'C2']):

    normal_vec, vecA, p1, p2, p3 = plane_three_atoms(u1, selA, atoms_vec[0], atoms_vec[1], atoms_vec[2])
    vecB, _, _ = vec_two_atoms(u2, selB, atoms_vec[0], atoms_vec[1])

    # Angle between vector A and B
    dot = np.dot(vecA, vecB)
    mult = LA.norm(vecA) * LA.norm(vecB)
    angle = np.arccos(dot/mult)
    proj_vecB = vecB - (np.dot(vecB, normal_vec) / np.dot(normal_vec, normal_vec)) * normal_vec
    proj_angle = np.arccos(np.dot(vecA, proj_vecB) / LA.norm(vecA)*LA.norm(proj_vecB))
    return proj_angle


def disp_two_vectors(u1, u2, selA, selB, atoms_vec=['N', 'N1']):
    '''
       Given a vectors defined by the atoms_vec in molecules in selA and selB,
       returns the displacement with respect to the axis of the vector for selA
    '''

    vecA, a1, a2 = vec_two_atoms(u1, selA, atoms_vec[0], atoms_vec[1])
    vecB, b1, b2 = vec_two_atoms(u2, selB, atoms_vec[0], atoms_vec[1])

    # Calculate the displacement vector    
    if np.dot(vecA, vecB) > 0: # same direction
        disp = [b2[0] - a2[0], b2[1] - a2[1], b2[2] - a2[2]]
    elif np.dot(vecA, vecB) < 0: # opp direction
        disp = [b2[0] - a2[0], b2[1] - a2[1], b2[2] - a2[2]]
    else: # orthogonal
        disp = [b2[0] - a1[0], b2[1] - a1[1], b2[2] - a1[2]]

    # Displacement projected on the axis of the first vector
    proj_disp = np.dot(disp, vecA) / np.dot(vecA, vecA)

    return proj_disp


def com_distance(u1, u2, selA, selB):
    '''
       Center of mass distance between to universe objects
    '''

    ag1 = u1.select_atoms(f"resid {selA}")
    ag2 = u2.select_atoms(f"resid {selB}")

    CofMA = ag1.center_of_mass()
    CofMB = ag2.center_of_mass()

    rab = abs(CofMA-CofMB)

    return LA.norm(rab)

def find_term_res_pdb(pdb, dist_min=6):
    """Given a PDB file, returns the residue ids of the terminal DNA residues

    Args:
        pdb (str): Path to DNA file
        nres (int): Total number of residues 
        start_res (int, optional): The first DNA residue in the pdb file. Defaults to 1.
    """
    from scipy.spatial import cKDTree
    u = mda.Universe(pdb, format="PDB")
    u_dna = u.select_atoms("nucleic")
    u_dye = u.select_atoms("not nucleic")

    # res next to the dyes are not terminal
    dye_ids = np.unique(u_dye.atoms.resids)
    non_ter = np.unique(np.append(np.append(dye_ids-1, dye_ids), dye_ids+1))
    # Make a list of each residue center of mass
    cofms = []
    resids = []
    for res in u_dna.residues:
        cofms.append(res.atoms.center_of_mass())
        resids.append(res.resid)

    kd_tree = cKDTree(cofms)
    edges = []
    # Find the edge residues: Those that don't have more than 1 neighboring residue
    for i, cofm in enumerate(cofms):
        nn_ids = kd_tree.query_ball_point(cofm, r=dist_min)   
        nn_ids.remove(i) # don't count the point itself     
        # If there's no more that 1 NN, the residue is an edge
        if len(nn_ids) < 2:
            # Make sure is not adjacent to any of the dyes
            if not (resids[i] in non_ter):
                edges.append(resids[i])
    '''
    # Calculation using the the TER residues
    import subprocess
    # Use grep to find the line marked as TER 
    output = subprocess.Popen(f"grep 'TER' {pdb}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    out_lines, err_lines = output.communicate()
    ter_residues = out_lines.splitlines()

    ter_ids = []
    for ters in ter_residues:
        ter_ids.append(int(ters.split()[-1]))
    
    if len(ter_ids) > 0:
        ter_list = [start_res, nres-start_res+1, ter_ids[0], ter_ids[0]+1]
    else: 
        print("No TER residues found. May need to clean the PDB file.")
    print('** Edges the old was', np.unique(ter_list))
    '''

    return edges