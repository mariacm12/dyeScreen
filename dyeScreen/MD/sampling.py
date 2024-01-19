''' Sampling dyes in DNA structure'''

import os
import numpy as np
from numpy import linalg as LA

# MD analysis tools
import MDAnalysis as mda
import dyeScreen.commons.geom_utils as gu

import dyeScreen.FF.file_process as fp
from dyeScreen.FF.gen_dye_lnk import gen_frcmod

import csv
import subprocess

# Scipy
from scipy.spatial import cKDTree, KDTree

class scan_cofigs():

    def __init__(self, type_of_attach='double', dna_stericBox=20) -> None:
        """Class for DNA dimer sample scanning 

        Args:
            type_of_attach (str, optional): Dye attaches to one ("single") or  two ("double")
                                            nucleotides. ONLY "double" is tested. 
            dna_stericBox (int, optional): Size of DNA box for steric interaction in Ams. 
                                           Defaults to 20.

        Raises:
            NotImplementedError: Only implemented options are "single" and "double"
        """
        if type_of_attach == 'single':
            self.num_attachs = 1
        elif type_of_attach == 'double':
            self.num_attachs = 2
        else:
            raise NotImplementedError(
                "Only single and double attachments to DNA are allowed")

        self.attach_idxs = []

        # How big of a "box" should we consider when calculating steric interactions
        self.dna_sbox = dna_stericBox

        self.pairs = []
        self.bonds = []
        self.del_res = ""


    class DNA():
        def __init__(self, dnaU, chain=None) -> None:
            """DNA fragment class

            Args:
                dnaU (MDAnalysis Universe): Universe with the DNA residues
                chain (str, optional): DNA chain label to be selected. Defaults to None.
            """
            # Select only viable chain
            if chain: 
                self.chain_sel = dnaU.select_atoms("segid " + chain)
            else: 
                self.chain_sel = dnaU.select_atoms("all")
            
            # Properties
            self.atoms = self.chain_sel.atoms
            self.residues = self.chain_sel.residues
            self.positions = self.atoms.positions
            self.names = self.atoms.names
            self.com = self.atoms.center_of_mass()
            # list of non-H atoms
            self.natoms = len(self.names)
            self.atIds = np.arange(self.natoms)
            nonHs = np.invert(np.char.startswith(self.names.astype(str), 'H'))
            self.heavyIds = self.atIds[nonHs]

        def get_ter(self):
            '''Return terminal residues of DNA fragment
            '''
            ter_res = []
            for r in self.chain_sel.residues:
                if "HO3'" in r.atoms.names:
                    ter_res.append(r.ix)
            
            return ter_res
        
        def select_atoms(self, string):
            return self.chain_sel.select_atoms(string)

    def extract_monomer(self, pdb_monomer, resD, resL, attach_info):
        """Extract coords and info from monomer pdb

        Args:
            pdb_monomer (string): Dir location of the monomer pdb to scan
            resD (int): resid of the dye according to its pdb
            resL (int, optional): resid of the linker (if any) according to its pdb. Defaults to None.
            attach_info (list): List with the atom_names of the OPO3 attachment points in the molecule. 
        """
        # extract info on opo3 group
        self.OH = []
        self.Os = []
        self.Pname = []
        self.preO = []
        for attach_group in attach_info:
            self.preO.append(attach_group[0])
            self.Pname.append(attach_group[1])
            self.Os.append([attach_group[2], attach_group[3]])
            # These atoms just get deleted
            self.OH += [attach_group[4], attach_group[5]]


        sel_1 = [resD, resL] if resL else [resD]
        xyz, names, typ, com, reslist = gu.get_coords(pdb_monomer, None, 0, sel_1=sel_1, cap=False, resnames=True)
        self.xyzA = np.array(xyz[0])
        self.xyzB = np.array(xyz[0])

        self.comA = com[0]
        self.comB = com[0]

        self.names = names[0]
        self.types = typ[0]
        self.reslist = reslist

        # list of non-H atoms
        self.natoms = len(self.names)
        self.atIds = np.arange(self.natoms)
        nonHs = np.invert(np.char.startswith(self.names.astype(str), 'H'))
        self.heavyIds = self.atIds[nonHs]

        DNA_Os = ["O5'", "O3'"]
        DNA_P = "P"
        DNA_OP = ["OP1","OP2"]
        for a in range(self.num_attachs):
            attach_idx = np.where(self.names == self.Pname[a])[0][0]
            self.attach_idx = attach_idx
            self.attach_idxs.append(attach_idx)
            # rename pre-attachment oxygens for DNA FF name
            self.names[self.names == self.preO[a]] =  DNA_Os[a]
         # Rename PO2 atoms in O5' side
        for a, OPs in enumerate(self.Os[0]):
            self.names[self.names == OPs] =  DNA_OP[a]
        self.names[self.names == self.Pname[0]] =  DNA_P

        return

    def extract_DNA(self, pdb_dna, chain=None):
        """Extract info from DNA PDB given the allowed chain segid

        Args:
            pdb_dna (string): Dir location of the DNA scaffold
            chain (string, optional): segname of the DNA chain we wish to scan (if any). Defaults to None.
        """
        self.u_DNA = mda.Universe(pdb_dna, format="PDB")
        self.dna = self.DNA(self.u_DNA, chain=chain) 

        # Extract positions and info of all P atoms in every nucleotide
        # FIX: The 3' side doesn't have a phosphate. Loop through all residues and save P or False if it doesn't
        reslist = self.dna.chain_sel.select_atoms("nucleic").residues
        self.res_idx = []
        self.bond_pos = []
        self.bond_resi = []
        self.bond_resn = []
        for res in reslist:
            # index atom (Choosing atom that's always kept)
            self.res_idx.append((res.atoms.select_atoms("name O3'").positions[0]).tolist())
            # Phosphate where dye bonds
            patom = res.atoms.select_atoms("name P").positions
            self.bond_resi.append(res.resid)
            self.bond_resn.append(res.resname)
            if len(patom>0):
                self.bond_pos.append(patom[0])
            else:
                self.bond_pos.append(np.array([9999,9999,9999]))
        self.bond_labels = [self.bond_resn[i] +
                    str(self.bond_resi[i]) for i in range(len(self.bond_resn))]
        print("atoms count", self.dna.natoms,
              len(self.dna.chain_sel.atoms.positions))

        return

    def move_single_att(self, a, b):
        """Moving coordinate for a specific sampling, single attachement mode. 
            NOT tested (usually not needed.)

        Args:
            a (int): Attachement point of monomer 1
            b (int): Attachement point of monomer 2
        """
        # Define DNA box
        self.middnaA = self.bond_pos[a]
        self.middnaB = self.bond_pos[b]
        dna_stericA = self.get_DNABox(self.dna_sbox, mid=self.bond_pos[a])
        dna_stericB = self.get_DNABox(self.dna_sbox, mid=self.bond_pos[b])

        new_xyzA, newcomA = gu.align_two_molecules_at_one_point(self.xyzA, self.bond_pos[a], self.xyzA[self.attach_idx], 
                                                                dna_stericA.com, self.comA, align=-1, accuracy=0.7)
        new_xyzB, newcomB = gu.align_two_molecules_at_one_point(self.xyzB, self.bond_pos[b], self.xyzB[self.attach_idx],
                                                                dna_stericB.com, self.comB, align=-1, accuracy=0.7)

        # Appending current bond information
        pairs = self.bond_labels[a]+'-'+self.bond_labels[b]
        self.bonds.append([pairs, self.bond_pos[a][0], self.bond_pos[b][0]])
  
        # Have to "manually" make the list of resnames and resids so it matches the mol2 files
        resname1 = self.reslist[0][0]
        resname2 = self.reslist[0][-1]
        self.reslist_dimer = [resname1, resname2]*2  # 4 total residues
        natoms1, natoms2 = np.count_nonzero(
            self.reslist[0] == resname1), np.count_nonzero(self.reslist[0] == resname2)
        self.resid_list = [0]*natoms1 + [1]*natoms2 + [2]*natoms1 + [3]*natoms2

        self.dimer_mol = gu.get_mol([new_xyzA, new_xyzB], [self.names]*2, [self.types]*2,
                                  res_names=self.reslist_dimer, res_ids=self.resid_list)

        return


    def move_double_att(self, a, b):
        """Moving coordinate for a specific sampling, double attachement mode. 

        Args:
            a (int): First attachement point of monomer 1
            b (int): First attachement point of monomer 2
        """
        
        # Define DNA box 
        self.middnaA = (self.bond_pos[a] + self.bond_pos[a+1])/2
        dna_stericA = self.get_DNABox(self.dna_sbox, mid=self.middnaA)
        self.middnaB = (self.bond_pos[b] + self.bond_pos[b+1])/2 
        dna_stericB = self.get_DNABox(self.dna_sbox, mid=self.middnaB)

        # Is the dimer Serial or Parallel
        dimer_type, nt = self.get_dimer_type(a, b)
        dimer_type += str(nt) + "-"
        if "S" in dimer_type:
            align = "apar" 
        else:
            align = "orth"
        align = "apar"
        print(dimer_type, nt) #, align)

        new_xyzA, newcomA = gu.align_two_molecules_at_two_points(self.xyzA, self.bond_pos[a], self.bond_pos[a+1], 
                                                              self.xyzA[self.attach_idxs[0]], self.xyzA[self.attach_idxs[1]], 
                                                              dna_stericA.com, self.comA, align=align)
        new_xyzB, newcomB = gu.align_two_molecules_at_two_points(self.xyzB, self.bond_pos[b], self.bond_pos[b+1], 
                                                              self.xyzB[self.attach_idxs[0]], self.xyzB[self.attach_idxs[1]], 
                                                              dna_stericB.com, self.comB, align=align)
        
        # Appending current bond information (x position of an atom in the bonding residues)
        #  With double atachement, the res in-between is deleted

        pairs = dimer_type + self.bond_labels[a-1]+'&'+self.bond_labels[a]+'-'+self.bond_labels[b]+'&'+self.bond_labels[b+1]
        btwn =  self.res_idx[a+1] + self.res_idx[b-1]
        if a+1 == b: # Serial 0nt 
            print("This is a 0nt")
            pairs = dimer_type + self.bond_labels[a-1]+'&'+self.bond_labels[a]+'-'+self.bond_labels[b]+'&'+self.bond_labels[b+1]
            #p1 = (new_xyzA[self.attach_idxs[1]]).tolist()
            p1 = (new_xyzB[self.attach_idxs[0]]).tolist()
            o5 = (new_xyzA[self.names=="O5'"][0]).tolist()
            btwn = p1 + o5
        
        coords = self.res_idx[a-1] + btwn + self.res_idx[b+1]
        coords = [ '%.4f' % elem for elem in coords ]
        self.bonds.append([pairs] + coords)
        self.pairs.append(pairs)

        # Have to "manually" make the list of resnames and resids so it matches the mol2 files
        resname1 = self.reslist[0][0]
        resname2 = self.reslist[0][-1]
        self.reslist_dimer = [resname1, resname2]  
        natoms1, natoms2 = np.count_nonzero(
            self.reslist[0] == resname1), np.count_nonzero(self.reslist[0] == resname2)
        self.resid_list = [0]*natoms1 + [1]*natoms2
    
        self.dimer_mol = gu.get_mol([new_xyzA, new_xyzB], [self.names]*2, [self.types]*2,
                                  res_names=self.reslist_dimer, res_ids=self.resid_list)
        
        self.del_ids = [int(self.bond_resi[a]),int(self.bond_resi[b])]
        self.del_res = f" and not (resname {self.bond_resn[a]} and resid {self.bond_resi[a]})"
        self.del_res +=  f" and not (resname {self.bond_resn[b]} and resid {self.bond_resi[b]})"
        self.dye_del = " "

        # Always delete OH group
        for dels in self.OH:
            self.dye_del += f" and not name {dels}" 
        # Delete O's and P on the O3' side
        for dels in self.Os[1]:
            self.dye_del += f" and not name {dels}" 
        self.dye_del += f" and not name {self.Pname[1]}"

        return 

    def save_dimer_pdb(self, DNABox, path, idx, box_type):
        """Save dimer pdb with new coordinates

        Args:
            DNABox (int, optional): The size of the DNA distance box to include with samples (in A).
            path (string): Folder to save samples.
            idx (int): Valid sample index.
            box_type (str, optional): How to calculate the DNA box: Default is "doubleAtt" for a box around each dye's 
                                      attachment midpoint.

        Raises:
            ValueError: Provided box_type is not implemented.

        Returns:
            MDAnalysis object: Moleculed saved in PDB.
        """
        dimer_mol = self.dimer_mol.select_atoms('all')
        pdb_name = f'{path}/dimer_{idx}.pdb'

        if DNABox:
            nres = np.unique(self.reslist[0])
            if box_type=='singleCenter': # A single box around dimer
                mid = dimer_mol.atoms.center_of_geometry()
            elif box_type=='doubleCenter': # Add a box around each molecule
                mid = [dimer_mol.select_atoms(f'resid 1 or resid {len(nres)}').atoms.center_of_geometry(),
                    dimer_mol.select_atoms(f'resid {len(nres)+1} or resid {len(nres)*2}').atoms.center_of_geometry()]
            elif box_type=='doubleAtt': # Add a box around the bond point of each molecule
                mid = [self.middnaA, self.middnaB]
            else: 
                raise ValueError('Invalid box_type: Choose between "singleCenter", "doubleCenter", "doubleAtt"')
            nuc_box = self.get_DNABox(DNABox, mid)
            dimer_mol = dimer_mol.select_atoms('all'+self.dye_del)

            if nuc_box:
                nuc_box = nuc_box.select_atoms('all'+self.del_res)

                # Delete incomplete nucleotides
                nuc_box = gu.del_extra_atoms(nuc_box)
                nuc_box = self.cap_edge(nuc_box, self.del_ids)
                res_list =[r.resid for r in nuc_box.residues] 

                #An attempt to reorder the residues so the dimer is where the deleted residue was
                parta = nuc_box.select_atoms(f'resid 0:{self.del_ids[0]-1}')
                partb = nuc_box.select_atoms(f'resid {self.del_ids[0]+1}:{self.del_ids[1]-1}')
                partc = nuc_box.select_atoms(f'resid {self.del_ids[1]+1}:{max(res_list)}')

                combined = dimer_mol.select_atoms(f'resid {self.del_ids[0]}')
                if len(parta)>0:
                    combined = mda.Merge(parta, dimer_mol.select_atoms(f'resid 1'))
                if len(partb)>0:
                    combined = mda.Merge(combined.atoms, partb)
                combined = mda.Merge(combined.atoms, dimer_mol.select_atoms(f'resid 2'))
                if len(partc)>0:
                    combined = mda.Merge(combined.atoms, partc)
                
                dimer_mol = combined.select_atoms("all") 
                
            else:
                print('!!! No DNA', idx, dimer_mol.atoms.center_of_geometry(), DNABox)
        dimer_mol.write(pdb_name)

        return dimer_mol

    def check_valid_sample(self, a, b, dist_min):
        """Checks that a-b pair is within dist_min, or is not a terminal residue

        Args:
            a,b (int): Index of molecule
            dist_min (double): Minimum distance on Ams.

        Returns:
            bool: Whether the pair is valid
        """
        if a in range(len(self.bond_pos)) and b in range(len(self.bond_pos)): 
            is_edge = self.bond_pos[a][0] == 9999 or self.bond_pos[b][0] == 9999
            if not is_edge:
                return LA.norm(self.bond_pos[a] - self.bond_pos[b]) < dist_min
        return False
    
    def check_valid_pairs(self, dist_min):
        """ Uses KDTree algorithm to find nn pairs within dist_min

        Args:
            dist_min (int, optional): The minimum distance the monomers should be (in A).
        Returns:
            list: List of 2D tuples with pair indexes
        """
        kd_tree = cKDTree(self.bond_pos)
        pairs = kd_tree.query_pairs(dist_min, p=2)

        # Add the (i,i), i.e., 0nt pairs
        pairs.update([(i, i) for i in range(len(self.bond_pos))])

        return pairs

    def find_edge_residues(self, atom_sel):
        """Helper function to return atoms on the edge of a DNA fragment
        Returns positions of P atoms (on left-edge residues) and O3' (right edge)

        Args:
            atom_sel (MDAnalysis object): Atom selection.

        Returns:
            list: List of edge atoms IDs
        """
        reslist = atom_sel.select_atoms("nucleic").residues
        atom_list = list(atom_sel.atoms.ids)
        edge_atoms = []
        for r, res in enumerate(reslist):
            if r>0:
                P = res.atoms.select_atoms("name P").ids
                if P:
                    P = np.argwhere(atom_list==P[0])[0][0]
                    #print(list(atom_sel.atoms)[P], list(atom_sel.atoms)[P-1])
                    bond = atom_sel.atoms[[P, P-1]].bond    
                    if bond.value()>7:
                        edge_atoms.append(res.resid)      
        return edge_atoms
    
    def cap_edge(self, mol, dye_locs):
        """Turn the O5' edges in the mol object into terminal nucleotides by deleting PO2

        Args:
            mol (MDAnalysis object): DNA box
            dye_locs (list): IDs bond residues

        Returns:
            MDAnalysis object: Capped box
        """
        res_list = [r.resid for r in mol.residues]
        first = min(res_list)
        dye1, dye2 = dye_locs[0], dye_locs[1]
        dye_edges = [dye1-1, dye1+1, dye2-1, dye2+1]

        edges = [first] + self.find_edge_residues(mol)
        edges = [x for x in edges if x not in dye_edges]

        #print('edges: ', [first]+self.find_edge_residues(mol))
        for e in edges:
            mol = mol.select_atoms(f'all and not (resid {e} and (name P or name OP1 or name OP2))')
        return mol

    def get_DNABox(self, DNABox_size, mid):

        """ Return a DNA box of size DNABox_size around the midpoint(s) coordinates
        Args:
            DNABox_size (float or int): Size of the DNA box (equal dist from center point)
            mid (np.ndarray or list): Center point(s). 
            Either an array with the coordinates of the single center, or a list of centers 
            (when we wish to return mutiple "boxes" around multiple centers).

        Returns:
            DNA class object with the nucleotide selection
        """        
        if not isinstance(mid[0],np.ndarray):
            nuc_box = self.u_DNA.select_atoms(
                        f'point {mid[0]} {mid[1]} {mid[2]} {DNABox_size}')
            
        else:
            nuc_box = self.u_DNA.select_atoms(
                        f'point {mid[0][0]} {mid[0][1]} {mid[0][2]} {DNABox_size}' + 
                        f' or point {mid[1][0]} {mid[1][1]} {mid[1][2]} {DNABox_size}',sorted=False)
            
        nuc_box = self.DNA(nuc_box)
        
        if nuc_box.natoms > 0:
            return nuc_box
        else: 
            return None
        
    def get_dimer_type(self, a, b):
        """Auxiliary function to determine dimer type (e.g., serial/parallel) and nt separation
        Args:
            a (int): index monomer 1
            b (int): index monomer 2

        Returns:
            str: dimer_type, int: nt_separation
        """

        par_dist = 17.51
        # Is the dimer Serial or Parallel
        dimer_type = "S"
        nt = abs(a-b+1)
        if nt > 4:
            dimer_type = "P"
            # Find 0nt position
            points = [point for i, point in enumerate(self.bond_pos) if i != a and i != a+1]
            tree = KDTree(self.bond_pos)
            closest_points_a1 = tree.query_ball_point(self.bond_pos[a+1], par_dist, p=2)
            # Find the point closest to the target distance
            min_diff = float('inf')
            closest_b = -1
            for i in closest_points_a1:
                if i != a+1:
                    pdist_a1_b = np.linalg.norm(self.bond_pos[a+1] - self.bond_pos[i])
                    if abs(pdist_a1_b - par_dist) <= 0.1:
                        closest_b = i

            nt = abs(b-closest_b)
            print('**', a, a+1, b, closest_b, np.linalg.norm(self.bond_pos[a] - self.bond_pos[closest_b]))

        return dimer_type, nt


def scan_DNAconfig(pdb_monomer, pdb_dna, path_save, resD, resL=None, 
                   chainDNA=None, dist_min=20, DNABox=20, DNASt=10,
                   attachment='single', attach_points=['O','P'], 
                   box_type='doubleAtt', max_pairs=100):
    """Scans all possible dimer configurations in the given DNA scaffold, for the given monomer pdb file.

    Args:
        pdb_monomer (string): Dir location of the monomer pdb to scan
        pdb_dna (string): Dir location of the DNA scaffold
        path_save (string): Folder to save samples
        resD (int): resid of the dye according to its pdb
        resL (int, optional): resid of the linker (if any) according to its pdb. Defaults to None.
        chainDNA (string, optional): segname of the DNA chain we wish to scan (if any). Defaults to None.
        dist_min (int, optional): The minimum distance the monomers should be (in A). Defaults to 20.
        DNABox (int, optional): The size of the DNA distance box to include with samples (in A). Defaults to 20.
        DNASt (int, optional): The size of the DNA steric box that should be accounted around a molecule (in A). Defaults to 10.
        attachment (str, optional): Whether the molecule has a 'single' attachment to DNA, or two ('double'). Defaults to 'single'.
        attach_points (list, optional): List with the atom_names of the OPO3 attachment points in the molecule. 
            Order is from dye to end of linker (starting from the O that bonds to P, the order of other O's doesn't matter)
        box_type (str, optional): How to calculate the DNA box: Default is "doubleAtt" for a box around each dye's attachment midpoint
                                  "doubleCenter" for a box around each dye's center of geometry
                                  "SingleCenter" for a box around dimer's center of geometry
        max_pairs (int, optional): Max number of samples to produce. Defaults to 100.

    Raises:
        ValueError: Only 'single' or 'double' attachments are implemented.

    Returns:
        int: Number of samples generated
    """
 
    configs = scan_cofigs(type_of_attach=attachment, dna_stericBox=DNASt)

    configs.extract_monomer(pdb_monomer, resD, resL, attach_points) 
    configs.extract_DNA(pdb_dna, chain=chainDNA)

    print("atoms count", len(configs.dna.positions),
          len(configs.dna.chain_sel.atoms.positions))
    
    # Returns unique pairs for which distance is < dist_min
    all_pairs = len(configs.bond_pos) * (len(configs.bond_pos) - 1) / 2

    # Check which nucleotide pairs are valid O(nlogn)
    valid_pairs = configs.check_valid_pairs(dist_min)

    all_pairs = 0 #len(valid_pairs)

    idx = 0
    check = True
    for (i, j) in valid_pairs:
        if attachment=='single':
            # Single attachment
            configs.move_single_att(i, j)
        elif attachment=='double': 
            # Double attachment
            ### i is 2nd attachment of 1st molecule, j is 1st atachement of 2nd molecule
            ### We need to check if i-1 and i, and j and j+1 are next to each other (<7Ams), and if attachments have a P atom 
            if configs.check_valid_sample(i-1,i,7.0) and configs.check_valid_sample(j,j+1,7.0) and configs.check_valid_sample(i-1,j,999):
                all_pairs += 1
                if len(configs.pairs) <= max_pairs:
                    configs.move_double_att(i-1, j)
                    print(len(configs.pairs)-1, max_pairs)
                    check = True
            else:
                check = False
                continue
        else:
            raise ValueError('The only valid attachments are "single" and "double"')

        # Save dimer pdb with new coordinates 
        if len(configs.pairs) <= max_pairs and check==True:
            configs.save_dimer_pdb(DNABox, path_save, idx, box_type)
            print(f"{idx}: ({i},{j}), {configs.pairs[idx]}")
            idx += 1

    # Print number of samples vs all possible
    #all_pairs = len(valid_pairs)
    print(f'All possible: {int(all_pairs)}, valid pairs: {len(configs.pairs)}')

    # Save a file with all bonding info (which residues the monomers are attached to)
    with open(f'{path_save}/name_info.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(configs.bonds)

    return len(configs.pairs)

def gen_leap_nolinker(path_samples, amber_path, dye_attach, dna_attach, 
                      dye_mol2, frcmod=None, dye_name='DYE', 
                      ions=['NA', '0'], wbox=20):
    """Clean pdb file for AMBER and writes and runs tleap

    Args:
        path_samples (str): Path to directory with samples.
        amber_path (str): Path to AmberTools executables.
        dye_attach (list): The atom names that participate in the bonding for the dye.
        dna_attach (list): The atom names that participate in the bonding for the DNA.
        dye_mol2 (str): Path to mol2 file.
        frcmod (str): Path to frcmod file.
        dye_name (str, optional): Residue name. Defaults to 'DYE'.
        ions (list, optional): List with counter-ion name and final charge. 
                               Defaults to ['NA', '0'].
        wbox (int, optional): Size of water box. Defaults to 20.
    """

    # Loop through the samples
    bonds = open(path_samples+"/name_info.txt")
    lines = csv.reader(bonds, delimiter='\t')

    for i, line in enumerate(lines):

        # 1) Clean the pdb samples
        subprocess.run(f"{amber_path}pdb4amber -i {path_samples}/dimer_{i}.pdb -o {path_samples}/dimer_{i}_clean.pdb".split(' '))

        pdb_name = f'{path_samples}/dimer_{i}_clean.pdb'
        # Amber thinks that the residue after (or before) the dye is terminal, so have to delete the "TER" 
        # when it's followed or preceded by the HETATM
        nline = 'n'
        a = subprocess.Popen(f"sed -i '' '/^TER/{{N;/\{nline}HETATM.*{dye_name}/D;}}' {pdb_name}", shell = True) # Followed by dye
        a.wait()
        b = subprocess.Popen(f"sed -i '' '/^TER.*{dye_name}/d' {pdb_name}", shell = True) # dye is terminal
        b.wait()

        # 2) Search for the new res ids of the DNA nucleotides bonding to the monomers
        u = mda.Universe(pdb_name, format="PDB")
        pos = u.atoms.positions
        resids = u.atoms.resids

        make_bond = []
        print("** Generating input for", pdb_name)
        
        add = [1,-1]
        res_loc = 99999

        for b in range(1, len(line[1:])+1, 3):
            i_bond = np.array([float(line[b]), float(line[b+1]), float(line[b+2])])
            att = (b - 1) % 2 # 0, 1, 0, 1
            res_loc0 = resids[np.argwhere((abs(pos - i_bond)<0.1).all(axis=1))][0][0]
            if res_loc0+add[att] == res_loc:
                res_loc = res_loc0
                continue # Skip the 3rd bond (repeated) when dimer is 0nt
            res_loc = res_loc0
            make_bond.append([f"{res_loc+add[att]}.{dye_attach[att]}",f"{res_loc}.{dna_attach[att]}"])
            print(b, res_loc+add[att], dye_attach[att], res_loc, dna_attach[att])

        # 3) Create leap files 
        tleap_file = f"{path_samples}/dimer_{i}.in"
        if not frcmod:
            frcmod = dye_mol2[:-4]+"frcmod"
        gu.write_leap(dye_mol2, None, pdb_name, frcmod, f'{path_samples}/dimer_{i}.in', 
                    dye_res=dye_name, add_ions=ions,
                    make_bond=make_bond, water_box=wbox)

        # Deleting Amber cache files 
        for ftrash in ['sslink', 'renum.txt', 'nonprot.pdb']:
            myfile = f"{path_samples}/dimer_{i}_clean_{ftrash}"
            if os.path.isfile(myfile):
                os.remove(myfile)
            else:
                print("Error: %s file not found" % myfile)

        # 4) Running tleap
        subprocess.Popen(f"{amber_path}tleap -f {tleap_file} > {path_samples}/leap{i}.log", shell=True)

    print("** Last file ", i, " was completed")
    return

## Pending: Currently double attachment is specific to the no-linker case. Generalize functions to support dye+linker 
def gen_leap_linker(path_samples, amber_path, dye_attach, dna_attach, dye_mol2, linker_mol2, frcmod):
    return 

def modify_mol2(opo3h, mol2_file, mol2_out=None, attachment='double', dye_file=None, parmchk2=None):
    """ Modify the mol2 file of the linker or dye (in the case when there's no linker)

    Args:
        opo3h (list): List (of lists) of OPO3H atom names [mol1, mol2].
        mol2_file (str): Path to initial mol2 file.
        mol2_out (str, optional): Path to modified mol2 file. Defaults to None.
        attachment (str, optional):Type of attachement ('single' or 'double'). 
                                   Defaults to 'double'.
        dye_file (str, optional): file for not-yet-implemented frcmod joining function.
        parmchk2 (str, optional): Path to parmchk2 executable. Defaults to None.

    Raises:
        NotImplementedError: Actually haven't implemented 'single' because 
                             it'll likely not be needed.
    """

    # Select which atoms to delete and rename
    if attachment=='double':
        opo3h_1, opo3h_2 = opo3h[0], opo3h[1]
        rename = [[opo3h_2[0], opo3h_1[0], opo3h_1[1], opo3h_1[2], opo3h_1[3]], 
                  ["O3'", "O5'", "P", "OP1", "OP2"]]
        retype =  ["OS", "OS", "P", "O2", "O2"]
        del_names = opo3h_2[1:]+opo3h_1[-2:] # O3' and O5' side respectively

    else:
        raise NotImplementedError('Only double attachment implemented')
        

    # Create mol2 object and delete atoms
    dmol2 = fp.MOL2_DF()

    dmol2.read_file(mol2_file)

    new_dye, new_dbonds, dye_del_data = fp.del_atoms(dmol2, del_names, ids_col='atom_name')
    # rename oxygens
    new_dye = fp.rename_atoms(new_dye, rename[0], rename[1])

    # Make sure that the atom types of dye OP3 match the atom types in DNA
    new_dye = fp.change_atom_type(new_dye, rename[1], retype)

    dmol2.data['ATOM'] = new_dye
    dmol2.data['BOND'] = new_dbonds

    if not mol2_out: # rewrite file
        mol2_out = mol2_file
    dmol2.write_file(mol2_out)

    # Re-make frcmod
    if dye_file: # There's separate linker and dye FF
        gen_frcmod(dye_file, mol2_out, path_parm=parmchk2)
        # And Join the file, which is not yet implemented...
    gen_frcmod(mol2_out, None, path_parm=parmchk2)

    return
