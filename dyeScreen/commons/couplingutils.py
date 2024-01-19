#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupling functions for DFT calculation

Created on Mon Apr 12 15:55:49 2021

@author: mariacm
"""

import numpy as np
import scipy.linalg
from pyscf import gto, scf, tdscf, lib, dft, lo, solvent
from functools import reduce

#MD analysis tools
import MDAnalysis as mda
from csv import reader

#For cambrlyp and above
from pyscf.dft import xcfun
#dft.numint.libxc = xcfun
# =============================================================================
# QM functions from Ardavan/Qimin
# =============================================================================

def td_chrg_lowdin(mol, dm):
    """
    Calculates Lowdin Transition Partial Charges
    
    Parameters
    ----------
    mol. PySCF Molecule Object
    dm. Numpy Array. Transition Density Matrix in Atomic Orbital Basis
    
    Returns
    -------
    pop. Numpy Array. Population in each orbital.
    chg. Numpy Array. Charge on each atom.
    """
    #Atomic Orbital Overlap basis
    s = scf.hf.get_ovlp(mol)
    
    U,s_diag,_ = np.linalg.svd(s,hermitian=True)
    S_half = U.dot(np.diag(s_diag**(0.5))).dot(U.T)
    
    pop = np.einsum('ij,jk,ki->i',S_half, dm, S_half)

    print(' ** Lowdin atomic charges  **')
    chg = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
        
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        print('charge of  %d%s =   %10.5f'%(ia, symb, chg[ia]))
    
    return pop, chg

def jk_ints_eff(molA, molB, tdmA, tdmB, calcK=False):
    """
    A more-efficient version of two-molecule JK integrals.
    This implementation is a bit blackbox and relies and calculation the HF
    potential before trying to calculate couplings. 

    Parameters
    ----------
    molA/molB : PySCF Mol Obj. Molecule A and Molecule B.
    tdmA/tdmB : Numpy Array. Transiiton density Matrix

    Returns
    -------
    cJ ~ Coulomb Coupling
    cK ~ Exchange Coupling
    
    V_{ab} = 2J - K
    """
    
    from pyscf.scf import jk, _vhf
    naoA = molA.nao
    naoB = molB.nao
    assert(tdmA.shape == (naoA, naoA))
    assert(tdmB.shape == (naoB, naoB))

    molAB = molA + molB
    
    #vhf = Hartree Fock Potential
    vhfopt = _vhf.VHFOpt(molAB, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf',
                         'CVHFsetnr_direct_scf_dm')
    dmAB = scipy.linalg.block_diag(tdmA, tdmB)
    #### Initialization for AO-direct JK builder
    # The prescreen function CVHFnrs8_prescreen indexes q_cond and dm_cond
    # over the entire basis.  "set_dm" in function jk.get_jk/direct_bindm only
    # creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt.set_dm(dmAB, molAB._atm, molAB._bas, molAB._env)
    # Then skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None
    ####

    # Coulomb integrals
    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0        , molA.nbas , 0        , molA.nbas,
                      molA.nbas, molAB.nbas, molA.nbas, molAB.nbas)  # AABB
        vJ = jk.get_jk(molAB, tdmB, 'ijkl,lk->s2ij', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, tdmA)
        
    if calcK==True:
        # Exchange integrals
        with lib.temporary_env(vhfopt._this.contents,
                               fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0        , molA.nbas , molA.nbas, molAB.nbas,
                          molA.nbas, molAB.nbas, 0        , molA.nbas)  # ABBA
            vK = jk.get_jk(molAB, tdmB, 'ijkl,jk->il', shls_slice=shls_slice,
                           vhfopt=vhfopt, aosym='s1', hermi=0)
            cK = np.einsum('ia,ia->', vK, tdmA)
            
        return cJ, cK
    
    else: 
        return cJ, 0

# =============================================================================
# Functions to calculate QM properties
# =============================================================================
def V_Coulomb(molA, molB, tdmA, tdmB, calcK=False):
    '''
    Full coupling (slower, obviously)
    Parameters
    ----------
    molA/molB : PySCF Mol Obj. Molecule A and Molecule B.
    tdmA/tdmB : Numpy Array. Transiiton density Matrix
    calcK : Boolean, optional
       Whether to calculate exchange integral. The default is False.

    Returns
    -------
    Coulombic coupling, Vij.

    '''
    cJ,cK = jk_ints_eff(molA, molB, tdmA, tdmB, calcK=False)
    return 2*cJ - cK, cK

def V_multipole(molA,molB,chrgA,chrgB):
    """
    Coupling according to the multiple approximation
    
    Parameters
    ----------
    molA : pyscf mol
        molecule A.
    molB : pyscf mol
        molecule B.
    chrgA : ndarray
    chrgB : ndarray

    Returns
    -------
    Vij : float
        The Coulombic coupling in the monpole approx

    """
    
    from scipy.spatial.distance import cdist,pdist
    
    mol_dist = cdist(molA.atom_coords(),molB.atom_coords()) 
    Vij = np.sum( np.outer(chrgA,chrgB)/mol_dist ) #SUM_{f,g}[ (qf qg)/|rf-rg| ]

    return Vij

def V_pdipole(td1,td2,rAB):
    """
    Coupling according to the transition dipole approximation
    
    Parameters
    ----------
    tdq, td2A : numpy array
        Transition dipole vector for each molecule.
    rAB : numpy array
        COM distance vexctor between two molecules.

    Returns
    -------
    Vij : float
        The Coulombic coupling in the t dipole approx approx

    """
    
    const = 1 #a.u.
    rAB *= 1.8897259886
    miuAnorm = abs(np.linalg.norm(td1))
    miuBnorm = abs(np.linalg.norm(td2))
    RABnorm = np.linalg.norm(rAB)
    num = np.dot(td1,td2)-3*np.dot(td1,rAB)*np.dot(td2,rAB)
    
    Vij = (miuAnorm*miuBnorm/const)*num/RABnorm**3

    return Vij

def transfer_CT(molA,molB,o_A,o_B,v_A,v_B):
    '''
    Calculating the electron/hole transfer integrals from 1e- overlap matrix elements
    (Approximation)

    Parameters
    ----------
    molA, molB : Pyscf HF molecules
    o_A, o_B  : ndarray
        Occupied orbitals.
    v_A,v_B : ndarray
        Virtual orbitals.

    Returns
    -------
    te : float
        Electron transfer integral.
    th : float
        Hole transfer integral.

    '''
    from pyscf import ao2mo

    naoA = molA.nao #of atomic orbitals in molecule A
    
    #1 electron integrals between molA and molB, AO basis
    ovpi_AB = gto.intor_cross('int1e_ovlp',molA,molB) 
    # Transform integral to from AO to MO basis
    ovpi_ab = lib.einsum('pq,pa,qb->ab', ovpi_AB, v_A, v_B) #virtual
    ovpi_ij = lib.einsum('pq,pi,qj->ij', ovpi_AB, o_A, o_B) #occupied

    te = ovpi_ab[0][0]
    th = -ovpi_ij[-1][-1]

    print("**transfer integrals=",te,th)
    print(ovpi_ab[0][0],ovpi_ab[-1][-1])
    print(ovpi_ij[0][0],ovpi_ij[-1][-1])
    return te,th

def transfer_local(mfA,mfB,mfAB):
    '''
    Calculating the electron/hole transfer integrals with localized MOs

    Parameters
    ----------

    Returns
    -------
    te : float
        Electron transfer integral.
    th : float
        Hole transfer integral.

    '''
    moA = mfA.mo_coeff
    moB = mfB.mo_coeff
    moAB = mfAB.mo_coeff
    eps_AB = mfAB.mo_energy
    nao, nmo = moA.shape

    homo_idx = moA[:,mfA.mo_occ!=0].shape[1]
    lumo_idx = homo_idx+1

    # c_dimer^loc = c_dimer^T . c_loc
    c_loc = np.block([[moA.T, np.zeros((nmo,nao))], [np.zeros((nmo,nao)), moB.T]])
    c_dloc = lib.einsum("ji,ik->jk", c_loc, moAB.T) 

    e_diag = np.diag(eps_AB)
    f_loc = lib.einsum("ji,ii,ik->jk", c_dloc, e_diag, c_dloc) 

    th = f_loc[homo_idx][homo_idx+int(nmo/2)]
    te = f_loc[lumo_idx][lumo_idx+int(nmo/2)]

    return te,th

def transfer_lowdin(molA,molB,mfA,mfB,xc_f='b3lyp'):
    '''
    Calculating the electron/hole transfer integrals with FMO approach

    Parameters
    ----------
    molA, molB : Pyscf gto molecules
    mfA, mfB   : Pyscf mean field object from monomer dft
    xc_f       : DFT functional. Default is 'b3lyp'
    
    Returns
    -------
    te : float
        Electron transfer integral.
    th : float
        Hole transfer integral.

    '''

    def get_fmo(mf):
        mo = mf.mo_occ
        LUMO = np.argwhere(mo==0)[0][0]
        HOMO = LUMO-1
        return HOMO, LUMO 

    # Preparing initial DM

    cA = mfA.mo_coeff
    cB = mfB.mo_coeff
    nao1, nmo1 = cA.shape
    nao2, nmo2 = cB.shape

    c_loc = np.block([[cA, np.zeros((nao1,nmo2))], [np.zeros((nao2,nmo1)), cB]])
    occ_AB = np.concatenate((mfA.mo_occ, mfB.mo_occ))
    dmAB = scf.hf.make_rdm1(c_loc,occ_AB)

    # Calculating KS matrix
    molAB = molA + molB
    molAB.verbose = 4
    mf = scf.RKS(molAB)
    dft.xcfun.define_xc_(mfA._numint, xc_f)
    #dft.numint.NumInt.libxc = dft.xcfun
    #mf.xc = xc_f 
    mf.conv_tol = 1e-5
    mf.kernel(dm0=dmAB)

    ks_DA = mf.get_fock() #KS Fock matrix
    S_DA  = mf.get_ovlp()
    homo, lumo = get_fmo(mf)
    print(homo,lumo, S_DA.shape, ks_DA.shape)
    
    v_fmo = np.zeros((2,2))
    
    for n, i in enumerate([homo,lumo]):
        for m, j in enumerate([homo,lumo]):
            v_fmo[n,m] = ks_DA[i,j] - S_DA[i,j]*(ks_DA[i,i]+ks_DA[j,j])/2
            v_fmo[n,m] /= 1-S_DA[i,j]**2

    print('Couplings:',v_fmo)
    te = v_fmo[0,0]
    th = v_fmo[1,1]
    
    return te, th    


def test(molA,molB,mfA,mfB,o_A,o_B,v_A,v_B):
    #MO energies
    epsA = mfA.mo_energy
    epsB = mfB.mo_energy
    homo_idx = o_A.shape[1]
    lumo_idx = homo_idx+1

    #1 electron integrals between molA and molB, AO basis
    nuci_AB = gto.intor_cross('int1e_nuc',molA,molB)
    kini_AB = gto.intor_cross('int1e_kin',molA,molB)
    tfi_AB = nuci_AB + kini_AB
    ovpi_AB = gto.intor_cross('int1e_ovlp',molA,molB)
    # Transform integral to from AO to MO basis
    tfi_ab = lib.einsum('pq,pa,qb->ab', tfi_AB, v_A, v_B)
    tfi_ij = lib.einsum('pq,pi,qj->ij', tfi_AB, o_A, o_B) 
    ovpi_ab = lib.einsum('pq,pa,qb->ab', ovpi_AB, v_A, v_B)
    ovpi_ij = lib.einsum('pq,pi,qj->ij', ovpi_AB, o_A, o_B) 

    tfe = tfi_ab[0][0]
    tfh = tfi_ij[-1][-1]
    e1h, e2h = epsA[homo_idx], epsB[homo_idx]
    e1e, e2e = epsA[lumo_idx], epsB[lumo_idx]

    se = ovpi_ab[0][0]
    sh = ovpi_ij[-1][-1]

    th = (tfh - 0.5*(e1h+e2h)*sh)/(1-sh**2)
    te = (tfe - 0.5*(e1e+e2e)*se)/(1-se**2)
    cm_conv = 219474.6
    print("**transfer integrals=",te*cm_conv,th*cm_conv)
    print("non-orth", tfe*cm_conv,tfh*cm_conv)
    print("overlap mat", se*cm_conv, sh*cm_conv)
    return te,th


def V_CT(te,th,rab,mf=None,Egap=0):
    """
    CT coupling

    Parameters
    ----------
    te, th : float
        electron/hole transfer int.
    mf : Mean-field PySCF object for the dimer
    rab : ndarray (3,n)
        center of mass distance vector between chromophores.
    Egap : float
        Transition energy from TDDFT

    Returns
    -------
    float
        CT coupling.

    """
    RAB = rab
    if isinstance(rab[0],np.ndarray):
        RAB = np.linalg.norm(rab,axis=0)*1.8897259886 #Ang to a.u.

    if mf is not None: 
        #Energy of frontier orbitals
        EL = mf.mo_energy[mf.mo_occ==0][0]
        EH = mf.mo_energy[mf.mo_occ!=0][-1]
    
        #Fundamental gap
        Eg = EL - EH
        #optical gap
        Eopt = Egap
        #Local Binding energy
        U = Eg - Eopt 
    else:
        U = 0.7*0.0367493 # Set to a constant
    #Coulomb Binding energy
    perm = 1 #4*pi*e_0 in a.u.
    er = 77.16600 #water at 301.65K and 1 bar
    
    elect = 1 #charge of e-
    V = elect**2/(perm*er*RAB)
    domega = U-V

    return -2*te*th/(domega), domega, np.linalg.norm(rab)



def transfer_sym(mfAB):
    '''
    e- transfer integrals assuming dimer is symmetric
    via the energy splitting method

    Parameters
    ----------
    mfAB : Pyscf mean fied obj for dimer

    Returns
    -------
    te : float
        e- transfer integral
    th : float
        h+ transfer integral

    '''
    
    #MOs for the dimer

    mo_en = mfAB.mo_energy
    E_v = mo_en[mfAB.mo_occ==0]
    E_o = mo_en[mfAB.mo_occ!=0]
    #Frontier Energies
    EH,EHm1 = E_o[-1],E_o[-2]
    EL,ELp1 = E_v[0],E_v[1]
    
    #transfer integrals
    th = (EH-EHm1)/2
    te = (ELp1-EL)/2
    
    return te,th

def dimer_dft(molA,molB,xc_f='b3lyp',verb=4):
    mol = molA+molB
    mol.verbose = verb
    mf = scf.RKS(mol)
    mf.xc= xc_f
    #Run with COSMO implicit solvent model
    mf = solvent.ddCOSMO(mf).run()
    
    mo = mf.mo_coeff #MO Coefficients
    occ = mo[:,mf.mo_occ!=0] #occupied orbitals
    virt = mo[:,mf.mo_occ==0] #virtual orbitals   

    return mol,mf,occ,virt

def do_dft(coord,basis='6-31g',xc_f='b3lyp',mol_ch=0,spin=0,verb=4,scf_cycles=200,opt_cap=None):
    from pyscf.dft import xcfun
    #dft.numint.libxc = xcfun #comment for b3lyp and below
        
    #Make SCF Object, Diagonalize Fock Matrix
    mol = gto.M(atom=coord,basis=basis,charge=mol_ch,spin=0)
    mol.verbose = verb

    #optimize cap
    if opt_cap is not None:
        mf = scf.RHF(mol)
        mol = contrained_opt(mf, opt_cap)

    mf = scf.RKS(mol)
    mf.xc= xc_f
    #if xc_f is 'wb97x_v':
        #mf._numint.libxc = xcfun

    mf.max_cycle = scf_cycles
    mf.conv_tol = 1e-6
        
    #Run with COSMO implicit solvent model
    mf = solvent.ddCOSMO(mf).run()#mf.run()
    
    mo = mf.mo_coeff #MO Coefficients
    occ = mo[:,mf.mo_occ!=0] #occupied orbitals
    virt = mo[:,mf.mo_occ==0] #virtual orbitals   

    return mol,mf,occ,virt

def do_tddft(mf,o_A,v_A,state_id=0,tda=True):
    """  

    Parameters
    ----------
    mf : pyscf scf object
        result from DFT
    o_A : ndarray
        Occupied orbitals
    v_A : ndarray
        Virtual orbitals
    state_ids : list 
        Wanted excitated states 
        (e.g., [0,1] returns fro 1st and 2nd exc states)

    Returns
    -------
    None.

    """
    nstates = 1 if isinstance(state_id,int) else len(state_id)    
    if tda:
        td = mf.TDA().run(nstates=nstates) #Do TDDFT-TDA
        print(td.e)
    else:
        td = mf.TDDFT().run(nstates=nstates) #Do TDDFT without TDA approx

    if isinstance(state_id,list):
        Tenergy = [td.e[i] for i in state_id]
        Tdipole = [td.transition_dipole()[i] for i in state_id]
        O_st = [Tenergy[i] * np.linalg.norm(Tdipole[i])**2 for i in range(len(Tenergy))]
        if len(np.where(np.array(O_st)>0.1)[0]) > 0:
            ost_idx = np.argwhere(np.array(O_st)>0.1)[0][0]
        else: # None of the states have Osc strength > 0.1
            ost_idx = np.argmax(O_st)
        tdm = []
        for i in state_id:
            cis_A = td.xy[i][0]
            tdm.append(np.sqrt(2) * o_A.dot(cis_A).dot(v_A.T))
    else:
        Tdipole = td.transition_dipole()[state_id]
        Tenergy = td.e[state_id]
        O_st = Tenergy * np.linalg.norm(Tdipole)**2 
        ost_idx = state_id
        
        cis_A = td.xy[ost_idx][0] #[state_id][0]
        #Calculate Ground to Excited State (Transition) Density Matrix
        tdm = np.sqrt(2) * o_A.dot(cis_A).dot(v_A.T)

    # The CIS coeffcients, shape [nocc,nvirt]
    # Index 0 ~ X matrix/CIS coefficients, Index Y ~ Deexcitation Coefficients
 
    
    return Tenergy, Tdipole, O_st, tdm, ost_idx


def do_tdhf(coord,basis='6-31g',mol_ch=0,spin=0,verb=4,scf_cycles=200,state_id=0):

    #Make SCF Object, Diagonalize Fock Matrix
    mol = gto.M(atom=coord,basis=basis,charge=mol_ch,spin=0)
    mol.verbose = verb
    mf = scf.RHF(mol)
    mf.max_cycle = scf_cycles
    mf.conv_tol = 1e-6

    nstates = 1 if isinstance(state_id,int) else len(state_id)
    #Run with COSMO implicit solvent model
    mf = solvent.ddCOSMO(mf).run()#mf.run()
    td = mf.TDHF().run(nstates=nstates)
    
    energy0 = mf.energy_tot()
 
    if isinstance(state_id,list):
        Tenergy = [td.e[i] for i in state_id]
        Tdipole = [td.transition_dipole()[i] for i in state_id]
        O_st = [Tenergy[i] * np.linalg.norm(Tdipole[i])**2 for i in range(len(Tenergy))]
        if len(np.where(np.array(O_st)>0.1)[0]) > 0:
            ost_idx = np.argwhere(np.array(O_st)>0.1)[0][0]
        else: # None of the states have Osc strength > 0.1
            ost_idx = np.argmax(O_st)
    else:
        Tdipole = td.transition_dipole()[state_id]
        Tenergy = td.e[state_id]
        O_st = Tenergy * np.linalg.norm(Tdipole)**2
        ost_idx = state_id

    return energy0, Tenergy, Tdipole, O_st, ost_idx




#Adding Hs
def add_H_standard(u, xyz, names, sel):
    #atoms to delete
    op1 = np.nonzero(names=='OP1')[0][0]
    op2 = np.nonzero(names=='OP2')[0][0]
    p = np.nonzero(names=='P')[0][0]
    xyz_new = np.delete(xyz,[op1,op2,p],0)
    names_new = np.delete(names,[op1,op2,p],0)
        
    #capping with H
    O3 = u.select_atoms("resid "+str(sel)+" and name O3*")
    O5 = u.select_atoms("resid "+str(sel)+" and name O5*")
    O3_coord = O3.positions + 0.6
    O5_coord = O5.positions + 0.6
    xyz_add = np.append(xyz_new,O3_coord,axis=0)
    xyz_add = np.append(xyz_add,O5_coord,axis=0)
    names_add = np.append(names_new,['H']*2,axis=0)
    return names_add,xyz_add

def capH_bond(pos_capped, pos_cap, atm_capped):
    ''' Auxiliary function to calculate the correct position of the H cap
    '''
    bond0  = np.linalg.norm(pos_capped-pos_cap) #The original bond length to be broken
    if atm_capped == 'C':
        bondH  = 1.09
    elif atm_capped == 'O':
        bondH  = 0.97 
    else:
        raise ValueError('Only C and O should be capped with H')  
    pos_Hcap = pos_capped + (pos_cap-pos_capped)*(bondH/bond0)
    
    return pos_Hcap

def cap_H_general(u,sel,sel_id,res_list,H_loc=[]):
    '''
    Will cap a residue with H atoms, given a list of atoms to delete/replace
    
    Parameters
    ----------
    sel : MDAnalysis atom selection
        current atom selection
    sel_id : list
        The residue ids in current selection
    res_list : list
        Includes the list of atoms to delete and those to delete that must be 
        replaced with H, per residue.
        [res_pos ,[atoms_to_del],[atom_to_del_and_cap]]
        - res_pos is with respect to the sel_id list
    H_loc : list, optional
        If provided, determines the atom to be replaced by H [res_pos,[(resid,atom_name)].
        Must be of same length as res_pos and must give the position for every atom to be capped.
        If not provided, the H position is the same as the capped position but displaced.

    Returns
    -------
    None.

    '''

    all_names = np.empty((0))
    all_types = np.empty((0))
    all_xyz = np.empty((0,3))
    for i in range(len(res_list)):
        r = res_list[i]
        res_i = sel_id[r[0]]
        sel_2 = sel.select_atoms("resid "+str(res_i))
        xyz = sel_2.positions
        names = sel_2.atoms.names
        types = sel_2.atoms.types
 
        #atoms to delete
        try:
            datom_idx = [np.nonzero(names==i)[0][0] for i in r[1]]
            #print('***',res_i,datom_idx)
        except:
            datom_idx = []
            print("The atom_name to delete couldn't be found. No atoms will be deleted in res "+str(res_i))
        try:
            capped_idx = [np.nonzero(names==i)[0][0] for i in r[2]]
            capped_pos = xyz[capped_idx]
            capped_atm = names[capped_idx]  # The atom that is to be capped
        except:
            capped_idx = []
            capped_pos = []
            print("The atom_name to cap couldn't be found. No atoms will be replaced with H in res "+str(res_i))
        if len(H_loc)>0 and len(H_loc[i])>0:
            cap_i = H_loc[i][1]
            assert len(cap_i) == len(r[2]) # Number of H caps 
            H_pos = []
            for c_idx, ci in enumerate(cap_i):
                cap_sel = u.select_atoms("resid "+str(ci[0])+" and name "+ci[1])
                cap0_pos = cap_sel.positions
                H_pos.append(capH_bond(capped_pos[c_idx], cap0_pos, capped_atm[c_idx][0])[0])
            H_pos = np.array(H_pos)   
    
        else: 
            H_pos = capped_pos + 0.4
        num_caps = H_pos.shape[0]
        
        xyz_new = np.delete(xyz,datom_idx,0)
        names_new = np.delete(names,datom_idx,0)
        types_new = np.delete(types,datom_idx,0)

        #capping with H
        xyz_add = np.vstack((xyz_new,H_pos))
        names_add = np.append(names_new,['H']*num_caps,axis=0)
        types_add = np.append(types_new,['h1']*num_caps,axis=0)

        all_xyz = np.vstack((all_xyz,xyz_add))
        all_names = np.append(all_names,names_add,0)
        all_types = np.append(all_types,types_add,0)
        
    return all_names,all_xyz,all_types

def cap_H_mopac(u,sel,sel_id,res_list,a=None):
    '''
    Will cap a residue with H atoms, given a list of atoms to delete/replace
    
    Parameters
    ----------
    sel : MDAnalysis atom selection
        current atom selection
    sel_id : list
        The residue ids in current selection
    res_list : list
        Includes the list of atoms to delete and those to delete that must be 
        replaced with H, per residue.
        [res_pos ,[atoms_to_del],[atom_to_del_and_cap]]
        - res_pos is with respect to the sel_id list

    Returns
    -------
    None.

    '''

    all_names = np.empty((0))
    all_types = np.empty((0))
    all_xyz = np.empty((0,3))
    for i in range(len(res_list)):
        r = res_list[i]
        res_i = sel_id[r[0]]
        sel_2 = sel.select_atoms("resid "+str(res_i))
        xyz = sel_2.positions
        names = sel_2.atoms.names
        types = sel_2.atoms.types
 
        #atoms to delete
        try:
            datom_idx = [np.nonzero(names==i)[0][0] for i in r[1]]
            print('***',res_i,datom_idx)
        except:
            datom_idx = []
            print("The atom_name to delete couldn't be found. No atoms will be deleted in res "+str(res_i))
        try:
            cap_idx = [np.nonzero(names==i)[0][0] for i in r[2]]
        except:
            cap_idx = []
            print("The atom_name to cap couldn't be found. No atoms will be replaced with H in res "+str(res_i))
    
        cap_pos = xyz[cap_idx] + 0.4
        num_caps = cap_pos.shape[0]

        xyz_new = np.delete(xyz,datom_idx,0)
        names_new = np.delete(names,datom_idx,0)
        types_new = np.delete(types,datom_idx,0)

        xyz_add = np.vstack((xyz_new,cap_pos))
        names_add = np.append(names_new,['H']*num_caps,axis=0)
        types_add = np.append(types_new,['h1']*num_caps,axis=0)

        all_xyz = np.vstack((all_xyz,xyz_add))
        all_names = np.append(all_names,names_add,0)
        all_types = np.append(all_types,types_add,0)

    # Make temporary pdb
    new_mol = get_mol([all_xyz], [all_names], [all_types],
                       res_names=['PDA'], segname='PDI')
    new_mol = new_mol.select_atoms('all')
    new_mol.write('dimer_del.pdb')
    
    # Mopac hydrogenation
    cap_mol = mopac_cap('dimer_del.pdb')


    xyz = cap_mol.positions
    names = cap_mol.atoms.names
    types = cap_mol.atoms.types

    return names,xyz,types


def contrained_opt(mf, constrain_str):
    from pyscf.geomopt.geometric_solver import optimize
    #writin the constrains on a file 
    f = open("constraints.txt", "w")
    f.write("$freeze\n")
    f.write("xyz " + constrain_str)
    f.close()

    params = {"constraints": "constraints.txt",}
    mol_eq = optimize(mf, **params)
    mol_eq = mf.Gradients().optimizer(solver='geomeTRIC').kernel(params)
    return mol_eq

def pdb_cap(u,sel_1,sel_2,cap=cap_H_general,del_list=[],resnames=['A','B'],path_save='dimer.pdb',MDA_selection='all',cap_list=[[],[]],mol_name='ABC'):
    """
        Writes pdb file with dimer extracted coordinates
    """

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str
    agA = u.select_atoms(make_sel(sel_1))
    agB = u.select_atoms(make_sel(sel_2))

    namesA_cap, xyzA_cap, typesA = cap(u,agA,sel_1,del_list,cap_list[0])
    namesB_cap, xyzB_cap, typesB  = cap(u,agB,sel_2,del_list,cap_list[1])

    orig_mol = get_mol([xyzA_cap, xyzB_cap], [namesA_cap, namesB_cap], [typesA, typesB],
                       res_names=resnames, segname=mol_name)

    #save pdb
    orig_all = orig_mol.select_atoms(MDA_selection)
    orig_all.write(path_save)
    orig_A = orig_mol.select_atoms('resid 1')
    orig_B = orig_mol.select_atoms('resid 2')
    return orig_A, orig_B

def write_qchem(u,sel_1,sel_2,cap=cap_H_general,del_list=[],path_save='dimer.in',
                cap_list=[[],[]], charge=0, mult=1, basis='cc-pvdz', write_mode='w'):
    """
        Writes qchem input file with dimer extracted coordinates
        The file is meant for fragment-based calculations of CT coupling
    """

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str
    agA = u.select_atoms(make_sel(sel_1))
    agB = u.select_atoms(make_sel(sel_2))
    
    CofMA = agA.center_of_mass()
    CofMB = agB.center_of_mass()    
    
    if cap is not None:
        namesA_cap, xyzA_cap, _ = cap(u,agA,sel_1,del_list,cap_list[0])
        namesB_cap, xyzB_cap, _ = cap(u,agB,sel_2,del_list,cap_list[1])
    else:
        xyzA_cap = agA.positions
        namesA_cap = agA.atoms.names
        xyzB_cap = agB.positions
        namesB_cap = agB.atoms.names
        
    #write qchem file
    f = open(path_save, write_mode)
    f.write("$molecule\n")
    f.write(" " + str(charge+charge) + " " + str(mult)+"\n")
    f.write("--\n")
    # monomer 1
    f.write(" " + str(charge) + " " + str(mult)+"\n")
    for i_atom,x in enumerate(xyzA_cap):
        coord_str = np.array2string(x, formatter={'float_kind':lambda x: "%.7f" % x}, separator='   ', suppress_small=True)
        f.write("  " + namesA_cap[i_atom][0] + "   "+ coord_str[1:-1]+"\n")
    f.write("--\n")
    # monomer 2
    f.write(" " + str(charge) + " " + str(mult)+"\n")
    for i_atom,x in enumerate(xyzB_cap):
        coord_str = np.array2string(x, formatter={'float_kind':lambda x: "%.7f" % x}, separator='   ', suppress_small=True)
        f.write("  " + namesB_cap[i_atom][0] + "   "+ coord_str[1:-1]+"\n")
    f.write("$end\n \n")
    f.write("$rem")
    f.write("""
    method              =  lrcwpbe
    omega               =  370
    basis               =  %s
    scf_print_frgm      =  true
    sym_ignore          =  true
    scf_guess           =  fragmo
    sts_dc              =  fock
    sts_trans_donor     =  2-2
    sts_trans_acceptor  =  2-2"""%basis)
    f.write("\n$end")

    Rab = CofMA - CofMB    
    return Rab


def Process_MD(u,sel_1,sel_2,cap=cap_H_general,del_list=[],cap_list=[[],[]]):
    """
    

    Parameters
    ----------
    u   : MDAnalysis universe
        Object containing MD trajectory
    sel_1 : list
        list of strings with the residue ids of Molecule A.
    sel_2 : list
        list of strings with the residue ids of Molecule B.
    coord_path: string, optional
        The path were the temp csv coordinates files are to be saved
    add_H: The function to be used to cap the linkers with H
    Returns
    -------
    coordA,coordB,Rab.

    """
    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str
    agA = u.select_atoms(make_sel(sel_1))
    agB = u.select_atoms(make_sel(sel_2))

    CofMA = agA.center_of_mass()
    CofMB = agB.center_of_mass()    
    
    def coord_save(xyz,names):
        #Convert to pyscf format
        atoms = []
        for i in range(len(xyz)):
            new_atom = [names[i][0],tuple(xyz[i])]
            atoms.append(new_atom)
            
        return atoms

    if cap is not None:
       namesA_cap, xyzA_cap, _ = cap(u,agA,sel_1,del_list,cap_list[0])
       namesB_cap, xyzB_cap, _ = cap(u,agB,sel_2,del_list,cap_list[1])
    else:
        #sel_A = sel.select_atoms("resid "+str(sel_1))
        #sel_B = sel.select_atoms("resid "+str(sel_2))
        xyzA_cap = agA.positions
        namesA_cap = agA.atoms.names
        xyzB_cap = agB.positions
        namesB_cap = agB.atoms.names


    coordA = coord_save(xyzA_cap,namesA_cap)
    coordB = coord_save(xyzB_cap,namesB_cap)
    
    Rab = CofMA-CofMB    
    return coordA,coordB,Rab

def get_mol(coords, names, types, res_names, segname):
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
    res_names : list
        List of strings with residue names.

    Returns
    -------
    mol_new : MDAnalisys.AtomGroup
        Transformed molecule object.

    """
    if not len(coords) == len(names) == len(types) == len(res_names):
        raise ValueError("All input arrays must be of length Nmolecules")

    n_residues = len(res_names)
    #Creating new molecules
    resids = []
    natoms = 0
    for imol in range(n_residues):
        natom = len(names[imol])
        resid = [imol]*natom
        resids.append(resid)
        natoms += natom

    resids = np.concatenate(tuple(resids))
    assert len(resids) == natoms
    segindices = [0] * n_residues

    atnames = np.concatenate(tuple(names))
    attypes = np.concatenate(tuple(types))
    resnames = res_names

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

    # Adding positions
    coord_array = np.concatenate(tuple(coords))
    assert coord_array.shape == (natoms, 3)
    mol_new.atoms.positions = coord_array

    return mol_new

def align_to_mol(mol,ref, path_save=None):
    from MDAnalysis.analysis import align

    if type(mol) is str:
        mol = mda.Universe(mol, format="PDB")
    if type(ref) is str:
        ref = mda.Universe(ref, format="PDB")
    align.alignto(mol, ref, select="all", weights="mass")

    mol= mol.select_atoms('all')
    if path_save is not None:
        mol.write(path_save)
    return mol

def mol_to_pyscf(mol1, mol2):
    xyzA = mol1.positions
    xyzB = mol2.positions
    namesA = mol1.atoms.names
    namesB = mol2.atoms.names
    CofMA = mol1.center_of_mass()
    CofMB = mol2.center_of_mass()

    def coord_save(xyz,names):
        #Convert to pyscf format
        atoms = []
        for i in range(len(xyz)):
            new_atom = [names[i][0],tuple(xyz[i])]
            atoms.append(new_atom)

        return atoms

    coordA = coord_save(xyzA,namesA)
    coordB = coord_save(xyzB,namesB)

    Rab = CofMA-CofMB
    return coordA,coordB,Rab

def mopac_cap(mol_file):
    '''
    Given a forced H capping will fix it using mopac
    Parameters
    ----------
    sel : MDAnalysis atom selection
        current atom selection
    sel_id : list
        The residue ids in current selection
    Returns
    -------
    None.

    '''
    import subprocess


    # Write MOPAC hydrogenation
    #f = open('dimer-cap.mop', "w")
    #f.write(f'GEO_DAT="{mol_file}" ADD-H PDBOUT')
    #f.close()
    #subprocess.run(["mopac","dimer-cap.mop"])

    #Write relaxation
    f = open('dimer-relax.mop', "w")
    f.write(f'GEO_DAT="{mol_file}" ADD-H NOOPT OPT-H CHARGE=0 PDBOUT') #OPT-H PDBOUT')
    f.close()
    subprocess.run(["mopac","dimer-relax.mop"])

    mol_cap = mda.Universe("dimer-relax.pdb", format="PDB")

    mol = mol_cap.select_atoms('all')

    return mol

def calculate_MOs(mol1, mol2, path, basis="6-31G", molname='dimer'):
    from pyscf import gto, scf, lo, tools

    coordA,coordB,Rab = mol_to_pyscf(mol1, mol2)
    print(coordA, coordB)
    mol = gto.M(atom=coordA+coordB, basis=basis)
    mf = scf.RHF(mol).run()

    mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    mo_virt = mf.mo_coeff[:,mf.mo_occ==0]

    o_energies = mf.mo_energy[mf.mo_occ>0]
    v_energies = mf.mo_energy[mf.mo_occ==0]

    n_occ = mo_occ.shape[1]
    occ_print = 1
    for i in range(n_occ-occ_print,n_occ):
        print(f'Printing occupied orbital {i+1} with E = {o_energies[i]}')
        tools.cubegen.orbital(mol, f'{path}/{molname}_{int(i+1)}.cube', mo_occ[:,i])

    print(f"Number of occ orbitals: {n_occ}")
    v_start = 0
    v_print = 1
    for i in range(v_start,v_start+v_print):
        print(f'Printing virtual orbital {i+n_occ+1} with E = {v_energies[i]}')
        tools.cubegen.orbital(mol, f'{path}/{molname}_v{i+n_occ+1}.cube', mo_virt[:,i])
