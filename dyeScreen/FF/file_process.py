#%%
import numpy as np
import pandas as pd

def is_close(v1, v2, tol=0.0005):
    if abs(v1-v2) > tol:
        return False
    return True

class MOL2_DF():

    def __init__(self) -> None:
        """Class storing mol2 info in DataFrame.
        """
        # Initializing mol2 dictionary
        self.data = {}
        self.data['MOLECULE'] = ""
        self.data['ATOM'] = pd.DataFrame()
        self.data['BOND'] = pd.DataFrame()
        self.data['SUBSTRUCTURE'] = []

        # Defining keys
        self.atom_keys = ['atom_id', 'atom_name', 'x', 'y', 'z', 'atom_type', 'res_id', 'res_name', 'charge']
        self.bond_keys = ['bond_id', 'atom1', 'atom2', 'bond_type','x']

        # Defining formatters
        self.atom_formatter = [lambda x: f'    {x}', lambda x: f'{x:<3}        ',
                        lambda x: f'{x:.4f}   ', lambda x: f'{x:.4f}   ', lambda x: f'{x:.4f}',
                        lambda x: f'{x:<2}        ', lambda x: f'{x}', lambda x: f'{x}     ', lambda x: f'{x:.6f}']

        self.bond_formatter = [lambda x: f'   {x}   ', lambda x: f'{x}  ', lambda x: f'{x}', lambda x: f'{x:<2}']


    def read_file(self, mol2_file):
        """Read mol2 file into initialized DataFrame

        Args:
            mol2_file (str): Path to mol2 file
        """
        
        data = {}
        
        with open(mol2_file) as f:
            lines = f.readlines()
            total_lines = len(lines)
            # Save line indexing of groups
            #print(lines)
            for num, line in enumerate(lines):
                if 'MOLECULE' in line:
                    header_line = num
                if 'ATOM' in line:
                    atom_line = num
                if 'BOND' in line:
                    bond_line = num
                if 'SUBSTRUCTURE' in line:
                    substruc_line = num
            self.data['MOLECULE'] = lines[header_line+2].split()
            self.data['SUBSTRUCTURE'] = lines[substruc_line+1]
        
        #print(total_lines)
        
        self.data['ATOM'] = pd.read_csv(mol2_file, sep=' ', header=None, names=self.atom_keys, 
                                        skiprows=atom_line+1, nrows=bond_line-atom_line-1, 
                                        skipinitialspace=True, engine='python') 
        
        self.data['BOND'] = pd.read_csv(mol2_file, sep=' ', header=None, names=self.bond_keys, 
                                        skiprows=bond_line+1, nrows=substruc_line-bond_line-1, 
                                        skipinitialspace=True).drop('x', axis=1)

        
        return 

    def write_file(self, path, resname=None):
        """Write DataFrame into mol2 file 

        Args:
            path (str): Save path for mol2 file
            resname (str, optional): Residue name of molecule. 
                                     Defaults to None, and original resname will be used.
        """
        
        atom_data = self.data['ATOM']
        bond_data = self.data['BOND']
        # Fix atom and bond numbers
        atom_number = len(atom_data)
        bond_number = len(bond_data)
        mol_info = self.data['MOLECULE'][2:]

        # Overwrite resname
        if resname is not None:
            atom_data['res_name'] = np.array([resname]*len(atom_data))

        resname = atom_data['res_name'].iloc[0]

        # Make sure atom names are uppercase
        atom_data['atom_name'] = atom_data['atom_name'].str.upper()

        # Print header
        with open(path, "w") as f:
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(resname + "\n")
            f.write(f"  {atom_number}   {bond_number}     {mol_info[0]}     {mol_info[1]}     {mol_info[2]}\n")
            f.write("SMALL\n")
            f.write("bcc\n")
            f.write("\n\n")
            f.write("@<TRIPOS>ATOM\n")

            # Print atom section
            atom_str = atom_data.to_string(header=None, col_space=[2,3,7,7,7,2,1,3,7], index=False, 
                                            justify='right', formatters = self.atom_formatter)
            f.write(atom_str+"\n")

            # Print bond section
            f.write("@<TRIPOS>BOND\n")
            bond_str = bond_data.to_string(header=None, col_space=[2,2,2,2], index=False, 
                                            justify='right', formatters = self.bond_formatter)
            f.write(bond_str+"\n")

            f.write("@<TRIPOS>SUBSTRUCTURE\n")
            f.write(self.data["SUBSTRUCTURE"])

        return

def calculate_charges(df, charge_col='charge'):
    """Calculate total for the column charge. Function for mol2 option only. 

    Args:
        df (DataFrame): Pandas DataFrame with mol2 info.
        charge_col (str, optional): Column name for charge. Defaults to 'charge'.

    Returns:
        int: total charge
    """
    charge = df[charge_col].sum()
    return charge

def del_atoms(df, del_ids, ids_col='atom_id', pdb=False, del_later=None):
    """Search for the ids in 'atom_id' and delete those entries
       Saves the x coordinates of atoms we want to delete later (before the id is changed)

    Args:
        df (DataFrame): Pandas DataFrame with mol2 info.
        del_ids (list): List with IDs of atoms to delete.
        ids_col (str, optional): Name of column with atom IDs. Defaults to 'atom_id'.
        pdb (bool, optional): Wether the function is applied to a pdb. Defaults to False (mol2).
        del_later (list, optional): IDs of atoms we want to delete later. Defaults to None.

    Returns:
        list: X coordinates of atoms we want to delete later
    """
    if pdb:
        df_at = df
        df_bonds = pd.DataFrame([])
    else:
        df_at = df.data['ATOM']
        df_bonds = df.data['BOND']
    deleted_bnd = []

    if ids_col != 'atom_id':
        del_ids = [df_at.loc[df_at[ids_col] == other, 'atom_id'].tolist()[0] for other in del_ids]
        del_ids.sort()
        ids_col='atom_id'
    del_ids = np.array(del_ids)
    print(del_ids)

    coords_for_later = []
    if del_later:
        coords_for_later = [df_at.loc[df_at[ids_col] == dl, 'atom_name'].tolist()[0] for dl in del_later]
        print('for later', coords_for_later)


    for idx in range(len(del_ids)):
        ids = del_ids[idx] 
 
        deleted_row = df_at.loc[df_at[ids_col] == ids]
        df_at = df_at[df_at[ids_col] != ids]

        # Delete bonds (only available for mol2 files)
        if not pdb:
            # decrease ID 
            # df_at[ids_col] = np.where(df_at[ids_col] < ids, df_at[ids_col], df_at[ids_col]-1)
            mask = df_at[ids_col] >= ids
            df_at.loc[mask, ids_col] = df_at[ids_col]-1

            del_ids -= 1 # also for the list of atoms to delete
            df_bonds, deleted_bnd = delete_bonds(df_bonds, ids, deleted_bnd)

    return df_at, df_bonds, coords_for_later

def rename_atoms(df, old_names, new_names):
    '''
    Search for the atom names and replace them
    '''
    for r, replace in enumerate(old_names):
        df['atom_name'].replace([replace], new_names[r], regex=True, inplace=True)

    return df

def change_atom_type(df, names, new_type):
    '''
    Search for the atom names and replace the atom types
    '''
    for r, replace in enumerate(names):
        idx = df.index[df['atom_name']==replace].tolist()[0]
        df.loc[idx,'atom_type'] = new_type[r]

    return df

def fix_charge(dye_df, link_df, expected_chg, atom_adj, num_links=1):
    ''' Only for mol2!
    If the total charge after deleting the connecting atoms is incorrect,
    fix the charge by adjust the charges of the atoms in atom_adj

    Parameters
    ----------
    dye_df : pandas dataframe
        DESCRIPTION.
    link_df : pandas dataframe
        DESCRIPTION.
    expected_chg : double
       Correct total charge dye+linker
    atom_adj : list 
        List of length (num_links+1) with the atom(s) to adjust in the dye and the one in the linker.
    num_links : int
        Number of likers attached to dye (default is 1)

    Returns
    -------
    Dye DatFrame, Linker DataFrame, Corrected charge

    '''
    dc = calculate_charges(dye_df)
    lc = calculate_charges(link_df)*num_links

    if not is_close(dc+lc, expected_chg):
        cdiff = expected_chg  - (dc + lc)

        # charge difference divided equally between the bonding atoms
        alpha = cdiff/len(atom_adj)
        link_idx = link_df.loc[link_df['atom_id'] == atom_adj[-1]].index.item()
        # Adjust the charge
        link_df.at[link_idx, 'charge'] +=  alpha
        for i in range(num_links):
            dye_idx = dye_df.loc[dye_df['atom_id'] == atom_adj[i]].index.item()
            dye_df.at[dye_idx, 'charge'] +=  alpha

        dc = calculate_charges(dye_df)
        lc = calculate_charges(link_df)*num_links

    corrected_chg = dc+lc

    return dye_df, link_df, corrected_chg

def fix_charge_nolink(dye_df, expected_chg, atom_adj):
    ''' Fix the mol2 charges when no linker is given
    If the total charge after deleting the connecting atoms is incorrect,
    fix the charge by adjust the cahrges of the atoms in atom_adj

    Parameters
    ----------
    dye_df : TYPE
        DESCRIPTION.
    expected_chg : double
       Correct total charge dye+linker
    atom_adj : list 
        List of length 2 with the atom to adjust in the dye and the one in the linker.

    Returns
    -------
    Dye DataFrame, Corrected charge

    '''
    dc = calculate_charges(dye_df)

    if not is_close(dc, expected_chg):
        cdiff = expected_chg  - dc

        # charge difference divided equally between the two atoms
        alpha = cdiff/len(atom_adj)
        dye_idx = dye_df.loc[dye_df['atom_id'] == atom_adj[0]].index.item()

        # Adjust the charge
        dye_df.at[dye_idx, 'charge'] +=  alpha

        dc = calculate_charges(dye_df)

    corrected_chg = dc

    return dye_df, corrected_chg
        
def delete_bonds(df, ids, deleted_data):
    """Delete provided bonds in DataFrame (mol2)

    Args:
        df (DataFrame): Pandas DataFrame of molecule
        ids (int): Atom whose bonds we wish to delete
        deleted_data (list): Deleted indexes so far

    Returns:
        list: Accumulated deleted indexes after
    """

    deleted_row = df.loc[(df['atom1'] == ids) | (df['atom2'] == ids)]
    #print('**',ids, deleted_row)
    deleted_data.append(deleted_row['atom2'].index)
    # Delete bonds involving ids
    df = df[(df['atom1'] != ids) & (df['atom2'] != ids)] 

    # Decrease index of atom1 >= ids
    #df['atom1'] = np.where(df['atom1'] < ids, df['atom1'], df['atom1']-1)
    mask = df['atom1'] >= ids
    df.loc[mask, 'atom1'] = df['atom1']-1
    
    # Decrease index of atom2 >= ids
    #df['atom2'] = np.where(df['atom2'] < ids, df['atom2'], df['atom2']-1)
    mask = df['atom2'] >= ids
    df.loc[mask, 'atom2'] = df['atom2']-1

    return df, deleted_data


class PDB_DF():

    def __init__(self) -> None:
        """Class storing pdb info in DataFrame.
        """
        # Initializing pdb dictionary
        self.data = {}
        self.data['MOLECULE'] = ""
        self.data['AUTHOR'] = ""
        self.data['ATOM'] = pd.DataFrame()
        self.data['HETATM'] = pd.DataFrame()
        self.data['CONNECT'] = []
        self.data['MASTER'] = []

        # Defining keys
        self.atom_keys = ['type', 'atom_id', 'atom_name', 'res_name', 'res_id', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'atom_type']

        # Defining formatters
        self.atom_formatter = [lambda x: f'{x}  ', lambda x: f'{x} ', lambda x: f'{x:<3}', lambda x: f'{x}    ',
                               lambda x: f'{x}     ', lambda x: f'{x:.3f} ', lambda x: f'{x:.3f} ', lambda x: f'{x:.3f} ',
                               lambda x: f'{x:.2f} ', lambda x: f'{x:.2f} ', lambda x: f'        {x}']

    def read_file(self, pdb_file, names=None):
        """Read pdb file into initialized DataFrame

        Args:
            pdb_file (str): Path to input pdb file
            names (list, optional): Optinally give column names for df. Defaults to None.
        """
        name_str = ""
        author_str = ""
        a_noindex = []
        h_noindex = []
        c_line = ""
        m_line = []
    
        with open(pdb_file) as f:
            lines = f.readlines()
            total_lines = len(lines)
            # Save line indexing of groups
            #print(lines)
            for num, line in enumerate(lines):
                if 'COMPND' in line:
                   name_str = line
                if 'AUTHOR' in line:
                    author_str += line[:-2]
                if 'ATOM' not in line:
                    a_noindex.append(num)
                if 'HETATM' not in line:
                    h_noindex.append(num)
                if 'CONECT' in line:
                    c_line += line
                if 'MASTER' in line:
                    m_line = line.split()[1:]
        self.data['MOLECULE'] = "".join(name_str.split()[1:])
        self.data['AUTHOR'] = author_str
        self.data['CONNECT'] = c_line
        self.data['MASTER'] = m_line
        
        keys = names if names else self.atom_keys
        if total_lines-len(a_noindex)>0:
            self.data['ATOM'] = pd.read_csv(pdb_file, sep=' ', header=None, names=keys, index_col=False, 
                                                skiprows=a_noindex, skipinitialspace=True, engine='python') 
        if total_lines-len(h_noindex)>0:
            self.data['HETATM'] = pd.read_csv(pdb_file, sep=' ', header=None, names=keys, index_col=False,
                                                skiprows=h_noindex, skipinitialspace=True, engine='python') 
     
        return 

    def write_file(self, path, resname=None, print_connect=True, reset_ids=False):
        """Write DataFrame into pdb file 
        Args:
            path (str): Save path for pdb file
            resname (str, optional): Residue name of molecule. 
                                     Defaults to None, and original resname will be used.
            print_connect (bool, optional): Print connection info on pdb. Defaults to True.
            reset_ids (bool, optional): Reset atom IDs order. Defaults to False.
        """
        atom_data = self.data['ATOM']
        hetatom_data = self.data['HETATM']
        
        # Fix atom and bond numbers
        atom_number = len(atom_data)
        hetatom_number = len(hetatom_data)
        mol_name = self.data['MOLECULE']
        connect_data = self.data['CONNECT']
        if len(self.data['AUTHOR']) >0:
            author_line = self.data['AUTHOR'] #+ "AND POST PROCESSED WITH DYE-SCREEN"
        else:
            author_line = "AUTHOR    GENERATED WITH DYE-SCREEN"
        m_data = self.data['MASTER']

        # Print header
        with open(path, "w") as f:
            f.write(f"COMPND    {mol_name}\n")
            f.write(author_line+ "\n")

            # Print atom section
            if not atom_data.empty:
                if reset_ids:
                    atom_data = reset_atomids(atom_data)
                if resname is not None:
                    atom_data['res_name'] = np.array([resname]*len(atom_data))
                atom_data = atom_data.sort_values(by=['atom_id'])
                atom_str = atom_data.to_string(header=None, col_space=[6,1,1,3,1,5,5,5,4,4,1], index=False, 
                                               justify='right', formatters = self.atom_formatter)
                f.write(atom_str+"\n")
            # Print hetatm section               
            if not hetatom_data.empty:
                if reset_ids:
                    hetatom_data = reset_atomids(hetatom_data)
                if resname is not None:
                    hetatom_data['res_name'] = np.array([resname]*len(hetatom_data))
                    #print(hetatom_data.head())
                hetatom_data = hetatom_data.sort_values(by=['atom_id'])
                hetatm_str = hetatom_data.to_string(header=None, col_space=[6,1,1,3,1,5,5,5,4,4,1], index=False,
                                                    justify='right', formatters = self.atom_formatter)
                f.write(hetatm_str+"\n")
            #Print bonds section
            if print_connect:
                f.write(connect_data)
            #f.write(f"MASTER        0    0    0    0    0    0    0    0  ")
            f.write("END")


        return

def clean_numbers(thearray):
    """Return clean list of atoms w/o numbers

    Args:
        thearray (list): Initial array of atoms
    Returns:
        Clean list
    """
    def check_st(string):
        only_alpha = ""
        for char in string:
            if char.isalpha():
                only_alpha += char
        return only_alpha
    return [check_st(astring) for astring in thearray]

def make_names(symbols,counts):
    """Return list of atom symbols with numbers

    Args:
        symbols (list): List of atom symbols
        counts (list): List of IDs

    Returns:
        list: List with atom names as <symbol><number>
    """
    all_names = []
    for iatm, sym in enumerate(symbols):
        irange = np.arange(counts[iatm])+1
        sym_array = [ sym + str(aindex) for aindex in irange]
        all_names += sym_array
    return np.array(all_names)

def set_res(resname, df):
    old_res = df['res_name']
    new_res = np.array([resname]*len(old_res))
    return new_res

def move_to_bond(df_st, df_mob, del_st, del_mob, bond_st, bond_mob, path_bond_info=None):
    """Translate mobile molecule to the specified position 
        to facilitate a bond with static molecule

    Args:
        df_st (pandas DataFrame): Dataframe of static molecule
        df_mob (pandas DataFrame): Dataframe of mobile molecule
        del_st (list): Atoms (id) that will be deleted from static molecule
        del_mob (list): Atoms (id) that will be deleted from mobile molecule
        bond_st (list): Atoms involved in the new bond in df_st
                        [id of atom to be kept, id of atom where df_mob atom will be placed]
        bond_mob (int): Atom in df_mob that will be part of the bond
        path_bond_info (string, optional):Path to save atoms in bond. Defaults to None.

    Returns:
        pandas DataFrane: the modified dataframes for the static and mobile atoms
    """
    
    # Move the coordinates of the mobile mol such that the atom bond_mob[0] sits top of bond_st[1]
    mobx = df_mob['x'] - find_atom_data(df_mob, bond_mob, 'x')
    mobx += find_atom_data(df_st, bond_st[1], 'x')
    moby = df_mob['y'] - find_atom_data(df_mob, bond_mob, 'y')
    moby += find_atom_data(df_st, bond_st[1], 'y')
    mobz = df_mob['z'] - find_atom_data(df_mob, bond_mob, 'z')
    mobz += find_atom_data(df_st, bond_st[1], 'z')

    # update coordinates of mobile molecule dataframe
    df_mob['x'] = mobx
    df_mob['y'] = moby
    df_mob['z'] = mobz
    
    # Use del_atoms function to delete needed atoms
    new_st, __, s_del_data = del_atoms(df_st, del_st, pdb=True)
    new_mob, __, m_del_data = del_atoms(df_mob, del_mob, pdb=True)
    #print(s_del_data, '\n', m_del_data, '\n')

    # Store atom name in the st-mob bond 
    bond_info = [find_atom_data(df_st, bond_st[0], 'atom_name'),
                find_atom_data(df_mob, bond_mob, 'atom_name')]    
    print(bond_info, type(bond_info[0]))
    if path_bond_info is not None:
        np.savetxt(path_bond_info, bond_info, fmt="%s")
    
    return new_st, new_mob, bond_info

def join_molecules(pdb1, pdb2, idx_add=1):
    """ Merge 2 pdb data frames 
        Only supported for joining pdbs without connections

    Args:
        pdb_data1 (dict): Dictionary with data of first pdb
        pdb_data2 (dict): Dictionary with data of second pdb
        idx_add (int): Index to add to res_id 
            (for when needing to joing multiple mols). Default 1. 
    
    Returns:
        PDB_DF object: Merged pdb
    """
    # Assign different residue id
    if not pdb2.data['ATOM'].empty:
        pdb2.data['ATOM']['res_id'] += idx_add
    if not pdb2.data['HETATM'].empty:
        pdb2.data['HETATM']['res_id'] += idx_add

    pdb_data1 = pdb1.data
    pdb_data2 = pdb2.data

    merged_pdb = PDB_DF()
    merged_pdb.data['AUTHOR'] = pdb_data1['AUTHOR']
    merged_pdb.data['MOLECULE'] = pdb_data1['MOLECULE'] + ' + ' + pdb_data2['MOLECULE']
    merged_pdb.data['ATOM'] = pd.concat([pdb_data1['ATOM'], pdb_data2['ATOM']],axis=0,sort=False)
    merged_pdb.data['HETATM'] = pd.concat([pdb_data1['HETATM'], pdb_data2['HETATM']],axis=0,sort=False)
 

    return merged_pdb

def find_atom_data(mol_df, atom_id, req_col):
    """Retrieve data from atom from DataFrame

    Args:
        mol_df (DataFrame)
        atom_id (int): Requested atom ID
        req_col (str): Requested column info

    Returns:
        Requested value
    """
    req_val = mol_df.loc[mol_df['atom_id'] == atom_id][req_col].item()
    return req_val

def reset_atomids(mol_df):
    '''Reset order of atom IDs
    '''
    natoms = len(mol_df)
    new_ids = np.arange(1,natoms+1)
    mol_df['atom_id'] = new_ids
    return mol_df