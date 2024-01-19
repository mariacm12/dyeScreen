import sys
import os
import subprocess
import numpy as np
from pyscf import lib
# MD analysis tools
import MDAnalysis
import pandas as pd
import matplotlib.pyplot as plt

# All functions to calculate coupling
import dyeScreen.commons.couplingutils as cp
# Geometry functions
import dyeScreen.commons.geom_utils as gu
import time


H_to_eV = 27.211
H_to_cm = 219474.6
eV_to_cm = 8065.54

def import_trajs(traj_file, param=None, dye_res='DYE', dt=1, dt_write=10, num_sol=0, 
                 mode='cofm', vec_ats=[], save_path=""):
    """Given a traj file, calculates the requested features and saves them on a text file

    Args:
        traj_file (str): Path to trajectory file. Format must be ASCII AMBER TRJ 
                         coordinate files or PDB (with param set to None).
        param (str, optional): Path to AMBER parameter file, in .prmtop format. 
                               Set to None when PDB is given as trajectory.
        dye_res (str, optional): Residue name for the dye. Defaults to 'DYE'.
        dt (int, optional): Time step for generating features. Defaults to 1.
        dt_write (int, optional): Time step in fs used in traj which will be used for print. 
                                  Defaults to 10.
        num_sol (int, optional): The number of trajectory steps (counting before end-of-traj) 
                                 we want to calculate feature for. Defaults to 0.
        mode (str, optional): The type of feature to calculate. Options are {"cofm", "angle", 
                              "disp_short", "disp_long", "time_step"}, defaults to "cofm".
        vec_ats (list, optional): List with atom names (str) defining vector for angle and disp. 
                                  Not required for "cofm" and "time_step".
        save_path (str, optional): Path where files will be saved.

    Raises:
        ValueError: When dye_res doesn't correspond to residue name in PDB file.
        ValueError: When vec_ats is not provided and mode is "angle", "disp_short" or "disp_long"
        NotImplementedError: Incorrect feature name for mode option. 

    Returns:
        Numpy array with feature data.
    """
    if param is not None:
        print("traj file is ", traj_file)
        u = MDAnalysis.Universe(param, traj_file, format='TRJ')
        traj_len = int(len(u.trajectory))      
        istep = traj_len - num_sol                 
        if num_sol == 0:
            num_sol = traj_len
            istep = 1
    else:
        u = MDAnalysis.Universe(traj_file, format='PDB')
        num_sol = 1

    dyes = u.select_atoms("resname " + dye_res).atoms.resids
    if len(dyes) == 0:
        raise ValueError("Verify the dye_res name given corresponds to the PDB")

    del_list = [[0,["P","OP1","OP2"],["O3'","O5'"]]]
    dye_file = traj_file.split("/")[-1].split("_")
    res_data = []
    print(istep, traj_len+1)
    
    for fi, ts in enumerate(u.trajectory[istep:traj_len+1:dt]):   
        molA, molB = cp.pdb_cap(u, [dyes[0]], [dyes[-1]], resnames=[dye_res[:-1]+'A', dye_res[:-1]+'B'], 
                                del_list=del_list, path_save=save_path+'dimer.pdb', MDA_selection='all', mol_name=dye_res)
        t_i = round((ts.frame*dt_write),2)

        save_file = f"{save_path}{dye_file[0]}_{dye_file[1]}_{mode}"
        # Verify vector atoms are given
        if len(vec_ats) == 0 and mode in ["angle", "disp_short", "disp_long"]:
            raise ValueError("If 'angle' or 'disp' mode is chosen, you need to provide the atoms defining the vectors")
        if mode == "angle":
            a_rad = gu.angle_two_vectors(molA, molB, '1', '2', atoms_vec=vec_ats)
            res_data.append([t_i, np.degrees(a_rad)])
            save_file += f"{vec_ats[0]}-{vec_ats[1]}"
        elif mode == "cofm":
            # Universe must be redefined as molA/B doesn't contain weight information
            umol = MDAnalysis.Universe(save_path+'dimer.pdb', format='PDB')
            cofm = gu.com_distance(umol, umol, '1', '2')
            res_data.append([t_i, cofm])
        elif "disp" in mode:
            # angle_ats define the vectors for the long/short axis
            disp = gu.disp_two_vectors(molA, molB, '1', '2', atoms_vec=vec_ats)
            res_data.append([t_i, disp])
        elif mode == "time_step":
            res_data.append([t_i, fi])
        else:
            raise NotImplementedError("I've only implemented angles, cofm's and displacements so far, sorry :(")
    res_data = np.array(res_data)
    np.savetxt(save_file + ".txt", res_data, fmt='%.3f')

    return res_data

def make_feat_matrix(traj_files, feat_list, params, dye_res='DYE', dt=1, dt_write=10, num_sol=0, 
                     vec_atoms=[], traj_labels=[], save_path=None, 
                     print_tstep=True, feat_file='/feats.txt'):
    """Generates feature matrix for the requested features and saves as csv file. 

    Args:
        traj_files (list): List with trajectory files locations as strings.
        feat_list (list): List of features as strings. See import_trajs for implemented feats. 
        params (list): List with parameter files locations as strings.
        dye_res (str, optional): Residue name for the dye. Defaults to 'DYE'.
        dt (int, optional): Time step for generating features. Defaults to 1.
        dt_write (int, optional): Time step in fs used in traj which will be used for print. 
                                  Defaults to 10.
        num_sol (int, optional): The number of trajectory steps (counting before end-of-traj) 
        vec_atoms (list, optional): List with vector atom names for each feature. See import_trajs.
        traj_labels (list, optional): List with trajectory labels. If not given, index is used.
        save_path (str, optional): Path where files will be saved.
        print_tstep (bool, optional): Whether to include the time step feature. Defaults to True.
        feat_file (str, optional): Name of feature text file. Defaults to '/feats.txt'.

    Returns:
        Numpy array with feature matrix.
    """

    all_data = [] 
    # Print time step 
    fmt = '%i ' + '%.4f ' * len(feat_list)
    feats = feat_list.copy()
    vecs = vec_atoms.copy()
    if print_tstep:
        feats.insert(0, 'time_step')
        vecs.insert(0, '')
        fmt = '%i ' + fmt
        print(feats, feat_list)
        
    for m, mode in enumerate(feats):
        labels = []
        data_mode = np.array([])
        for t, traj_file in enumerate(traj_files):
            res_data = import_trajs(traj_file, param=params[t], dye_res=dye_res, dt=dt, dt_write=dt_write, 
                                    num_sol=num_sol, mode=mode, vec_ats=vecs[m], save_path=save_path)
            data_mode = np.concatenate((data_mode, res_data[:,1]), axis=None)
            if len(traj_labels) > 0:
                labels.append([traj_labels[t]] * len(res_data))
            else:
                labels.append([t] * len(res_data))
                print(len(res_data))
            print('***', mode, vecs[m], traj_file, len(res_data))
        all_data.append(data_mode)
    
    feat_data = np.array(all_data).T
    labels = np.array(labels).reshape((len(traj_files)*len(res_data)))
    feat_matrix = np.hstack((labels.reshape(-1, 1), feat_data))
    np.savetxt(save_path + feat_file, feat_matrix, fmt=fmt)

    return feat_data

def do_cluster(scores, cluster_sizes=[3,4,5], fixed_size=0):
    """Performs clustering on data 

    Args:
        scores (ndarray): PCA scores
        cluster_sizes (list, optional): List of possible cluster sizes for optimization. 
                                        Defaults to [3,4,5].
        fixed_size (int, optional): Give number>0 if a pre-set cluster size is desired (otherwise optimized)

    Returns:
        sklearn KMeans object
    """
    from sklearn.cluster import KMeans
    sum_sq = []
    for cs in cluster_sizes:
        kmeans_pca = KMeans(n_clusters=cs, random_state=42, n_init=10).fit(scores)
        sum_sq.append(kmeans_pca.inertia_)

    # Find opt number of clusters
    # Using the kneedle algorithm https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve 
    if fixed_size == 0:
        npoints = len(sum_sq)
        coords = np.vstack((np.arange(npoints), sum_sq)).T
        line = coords[-1] - coords[0]
        line /= np.linalg.norm(line)
        # find the distance from each point to the line
        vec_dist = coords - coords[0]
        dot = np.dot(vec_dist, line)
        vec_par_2line = np.outer(dot, line)
        vec2line = vec_dist - vec_par_2line
        # distance to line is the norm of vecToLine
        dist = np.linalg.norm(vec2line, axis=1)
        # "Elbow" is the point with maximum distance to the line
        best_idx = np.argmax(dist)
        opt_clusters = cluster_sizes[best_idx]
    else:
        opt_clusters = fixed_size
    print(f'Keeping {opt_clusters} clusters', best_idx)
    # Re-do k-means
    kmeans_pca = KMeans(n_clusters=opt_clusters, random_state=42, 
                        init="k-means++", n_init=10).fit(scores)

    return kmeans_pca

def do_PCA(data):
    """ Does PCA on feature data

    Args:
        data (pandas dataframe): Data to perform PCA on

    Returns:
        scores: ndarray of shape (n_samples, n_components). Scores from PCA fit.
        pca: sklearn final PCA object.
        exp_var: Percentage of variance explained by each of the selected components. 
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Scale the data 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # perform PCA using num of components needed to preserve 80% of var
    pca = PCA().fit(data_scaled)
    exp_var = pca.explained_variance_ratio_.cumsum()
    keep_idx = np.where(exp_var >= 0.8)[0][0] + 1
    print(f"Keeping {keep_idx} components")
    pca = PCA(n_components=keep_idx)
    scores = pca.fit_transform(data_scaled)
    return scores, pca, exp_var

def PCA_cluster(df, feat_list, dimers=None, print_tstep=True, label_path="name_info.txt"):
    """ Does PCA-clustering in feature data

    Args:
        df (pandas dataframe): DataFrame with calculated features.
        feat_list (list): List of features requested.
        dimers (list, optional): Dimer labels for additional column. Defaults to None.
        print_tstep (bool, optional): Whether to add time step column. Defaults to True.
        label_path (str, optional): In case dimers is given, the location of the file
                                    containing the types of each dimer. Defaults to "name_info.txt".

    Returns:
        -Dataframe after PCA-KMeans
        -PCA-KMeans object
        -PCA scores
        -Features explaining PCA Variance
    """
    # The last column is a blank space 
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    # Add the traj and time step labels 
    add_columns = ['traj'] + ['t_step'] if print_tstep else ['traj']
    df.columns = add_columns + feat_list

    # Do PCA
    num_cols = list(df.columns[2:]) # numerical columns
    scores, pca, var = do_PCA(df[num_cols])
    max_comp = np.argmax(abs(pca.components_), axis=1)
    print([f'PCA comp {i} var explained by {feat_list[i]}' for i in max_comp])

    # Do clustering
    kmeans_pca = do_cluster(scores, cluster_sizes=np.arange(1,7))
    df_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores[:,0:2])], axis=1)
    df_pca_kmeans.columns.values[-2:] = ["Component 1", "Component 2"]
    df_pca_kmeans["K-means PCA Cluster"] = kmeans_pca.labels_
    
    def gen_cluster_label(cluster_num):
        return f'C{cluster_num + 1}'
    df_pca_kmeans["Cluster"] = df_pca_kmeans["K-means PCA Cluster"].apply(gen_cluster_label)
    
    # Inserting additional column with dimer file labels
    def map_label(labels):
        label_dic = {}
        for el, lab in enumerate(labels):
            label_dic[el] = lab
        return label_dic
    
    if dimers:    
        
        df_copy = df_pca_kmeans.copy()
        df_copy["traj"] = df_pca_kmeans["traj"].map(map_label(dimers))
        
        df_names = pd.read_csv(label_path, sep="\t", header=None)
        df_names.columns = ['names'] + [str(i) for i in range(1,13)]
        def dimer_to_type(d, df):
            if 0 <= d < len(df):
                return df['names'].iloc[int(d)].split('-')[0]
            else:
                return None
        dimer_labels = [dimer_to_type(d, df_names) for d in dimers]
        df_copy["dtype"] = df_pca_kmeans["traj"].map(map_label(dimer_labels))
        
        return df_copy, kmeans_pca, scores, max_comp

    return df_pca_kmeans, kmeans_pca, scores, max_comp

def find_cluster_rep(kmeans_fit, data):
    """Find the representative samples for each cluster

    Args:
        kmeans_fit (sklearn.KMeans): pre-fitted kmean object
        data (pandas dataframe): DataFrame with feature data.


    Returns:
        list: Indexes of representative samples
    """
    cluster_centers = kmeans_fit.cluster_centers_
    cluster_labels = kmeans_fit.labels_
    # Compute distance between data points and cluster centers
    distances = np.linalg.norm(data - cluster_centers[cluster_labels], axis=1)
    nclusters = kmeans_fit.n_clusters

    # Find the samples closest to each center
    rep_samples = []
    for k in range(nclusters):
        cluster_indices = np.where(cluster_labels == k)[0]  
        rep_index = cluster_indices[np.argmin(distances[cluster_indices])]
        rep_samples.append(rep_index)

    return rep_samples

def find_cluster_limits(kmeans_fit, data, num_max=1):
    """Find the samples that are furthest from the rep on each cluster

    Args:
        kmeans_fit (sklearn.KMeans): pre-fitted kmean object.
        data (pandas dataframe): DataFrame with feature data.


    Returns:
        list: Indexes of limit samples
    """
    cluster_centers = kmeans_fit.cluster_centers_
    cluster_labels = kmeans_fit.labels_
    # Compute distance between data points and cluster centers
    distances = np.linalg.norm(data - cluster_centers[cluster_labels], axis=1)
    nclusters = kmeans_fit.n_clusters

    # Find the samples furthest to each center
    limit_samples = []
    for k in range(nclusters):
        cluster_indices = np.where(cluster_labels == k)[0]  
        sorted_dist = np.argsort(distances[cluster_indices])[::-1]
        furthest_idxs = []
        for j in range(num_max):
            furthest_idxs.append(cluster_indices[sorted_dist[j]])
        limit_samples.append(furthest_idxs)

    return limit_samples

def get_sample_pdb(df, sample_idx, traj_path, path_save, dt, cluster, dye_res='DYE', tcol='traj'):
    """Generate PDB file of a cluster sample

    Args:
        df (pandas dataframe): DataFrame with feature data.
        sample_idx (int): Index of sample to save.
        traj_path (str): Path where traj file is saved. Must be AMBER format. 
        path_save (str): Path where files will be saved.
        dt (int): Time step for the trajectory files. 
        cluster (int): Number of the cluster where the sample is taken (for naming purpose only)
        dye_res (str, optional): Residue name for the dye. Defaults to 'DYE'.
        tcol (str, optional): Name of trajectory column on DataFrame. Defaults to 'traj'.

    Returns:
        -MDAnalysis mol object saved on PDB.
        -Path of PDB file saved. 
    """
    time_idx = df['t_step'].iloc[sample_idx]
    dimer_idx = df[tcol].iloc[sample_idx]
    param =  f'{traj_path}/dimer_{dimer_idx}_clean.prmtop'
    traj = f'{traj_path}/dimer_{dimer_idx}_prod.nc'
    print(dimer_idx)

    u = MDAnalysis.Universe(param, traj, format='TRJ')
    dyes = u.select_atoms("resname " + dye_res).atoms.resids

    del_list = [[0,["P","OP1","OP2"],["O3'","O5'"]]]
    file_save = f"sample_{dimer_idx}_c{cluster}.pdb"
    resnums = [dyes[0]], [dyes[-1]]
    mol = gu.get_pdb(traj[:-3], param, path_save+file_save, resnums, select=(None, time_idx),
                    dt=dt, MDA_selection='all', del_list=del_list,
                    resnames=[dye_res[:-1]+'A', dye_res[:-1]+'B'], mol_name=dye_res)
    return mol, path_save+file_save

def QM_of_sample(df, sample_idx, traj_path, path_save, dt, cluster, dye_res='DYE', tcol='traj',
                basis = "cc-pvdz", xc = "b3lyp", ch=0, sp=1, scf_cycles=500, mode="Vexc", slurm_prefix=""):
    """Does QM calculations on a cluster sample

    Args:
        df (pandas dataframe): DataFrame with feature data.
        sample_idx (int): Index of sample to save.
        traj_path (str): Path where traj file is saved. Must be AMBER format. 
        path_save (str): Path where files will be saved.
        dt (int): Time step for the trajectory files. 
        cluster (int): Number of the cluster where the sample is taken (for naming purpose only)
        dye_res (str, optional): Residue name for the dye. Defaults to 'DYE'.
        tcol (str, optional): Name of trajectory column on DataFrame. Defaults to 'traj'.
        basis (str, optional): Basis set for DFT. Defaults to "cc-pvdz".
        xc (str, optional): Exchange Functional. Defaults to "b3lyp".
        ch (int, optional): Total charge of molecule. Defaults to 0.
        sp (int, optional): Total spin of molecule. Defaults to 1.
        scf_cycles (int, optional): Number of SCF cycles. Defaults to 500.
        mode (str, optional): Which property to calculate: {:Vexc", "VCT", "both"}. 
                              Defaults to "Vexc".
        slurm_prefix (str, optional): String with Slurm prefix for runs. Defaults to "".

    Raises:
        ValueError: Incorrect mode provided.

    Returns:
        -Results on coupling
        -Absorption spectra results (for "Vexc" mode), th for "VCT"
        -List of couplings for excitations with significant oscillator strength, te for "VCT"
    """
    start_time = time.time()
    # Extract pdb of sample
    mol, pdb_file = get_sample_pdb(df, sample_idx, traj_path, path_save, dt, cluster, dye_res=dye_res, tcol=tcol)

    u = MDAnalysis.Universe(pdb_file, format="PDB")
    if mode in ["Vexc", "both"]:
        save_V, save_abs, V_list, rab = do_QM(u, ["1","2"], dels=[], caps=[[],[]], basis=basis, xc=xc, ch=ch, spin=sp,
                                              scf_cycles=scf_cycles, start_time=start_time)
    if mode in ["VCT", "both"]:
        write_mode = 'w'
        save_path = pdb_file[:-4] + "_VCT.in"
        rab = cp.write_qchem(u, "1", "2", path_save=save_path, charge=ch, mult=sp, basis=basis,
                             cap=None, write_mode=write_mode)

        vct, th, te = extract_qchem(sample_idx, save_path, path_save, rab, slurm_prefix=slurm_prefix)

        if mode == "VCT":
            return vct, te, th
    if mode not in ["VCT", "Vexc", "both"]:
        raise ValueError("mode must be either 'Vexc', 'VCT' or 'both'")

    return save_V, save_abs, V_list

def extract_qchem(isample, file, path_save, rab, slurm_prefix="", max_time=10):
    """Run Q-Chem file for CT calculation and extract coupling. 

    Args:
        isample (int): Sample index (for labeling).
        file (str): Path for Q-Chem input file.
        path_save (str): Location of where file is read and output generated.
        rab (ndarray): Distance vector between monomers.
        slurm_prefix (str, optional): String with Slurm prefix for runs. Defaults to "".
        max_time (int, optional): Max time in mins allocated for the Q-Chem run. Defaults to 10.

    Returns:
        VCT in Hartree
        th in Hartree
        te in Hartree
    """
    f = open(path_save + "run_qchem.sh", "w")
    f.write(slurm_prefix)
    f.write(f"module load qchem\npath={file[:-3]}\n")
    f.write("# Fragment-based calculation\n")
    f.write("qchem -slurm -nt 64 ${path}.in ${path}.qcout\n")
    f.write(f"i={isample}\n")

    # Extract HOMO 
    f.write("homo=$(grep -m 1 -o 'HOMO = [0-9]*' ${path}.qcout | awk '{print $NF}')\n")
    # Extract LUMO 
    f.write("lumo=$(grep -m 1 -o 'LUMO = [0-9]*' ${path}.qcout | awk '{print $NF}')\n")
    f.write('grep "V( $homo, $homo)" ${path}.qcout | ' + "awk '{print $5}' > th_integrals_${i}.dat\n")
    f.write('grep "V( $lumo, $lumo)" ${path}.qcout | ' + "awk '{print $5}' > te_integrals_${i}.dat\n")
    f.write("paste th_integrals_${i}.dat te_integrals_${i}.dat >> t_integrals_${i}.dat\n")

    f.write("awk '{ print $1*(-1), $2}' t_integrals_${i}.dat > tintegrals_${i}.dat\n")
    f.write("rm t*_integrals_*.dat")
    f.close()
    process = subprocess.Popen(f"sbatch --parsable run_qchem.sh", shell = True, cwd=path_save)
    process.wait()

    if process.returncode == 0:

        # Monitor the file until it appears
        max_wait_time = max_time*60  # Maximum wait time in seconds
        wait_interval = int(max_wait_time/10)  # Time between checks in seconds

        file_path = f"{path_save}tintegrals_{isample}.dat"

        waited_time = 0
        while not os.path.exists(file_path) and waited_time < max_wait_time:
            time.sleep(wait_interval)
            waited_time += wait_interval

        if os.path.exists(file_path):
            # File was generated successfully, load it using numpy
            tintegrals = np.loadtxt(file_path)

            if len(tintegrals) == 0:
                return 0, 0, 0

            th = tintegrals[0]
            te = tintegrals[1] # in eV
            vct, _, _ = cp.V_CT(te, th, rab, mf=None, Egap=0)

        else:
            print("Error: File not found after waiting.")
            return 0, 0, 0
    else:
        print("Job submission failed.")
        return 0, 0, 0

    return vct*H_to_cm, th*eV_to_cm, te*eV_to_cm

def QM_along_traj(u, dye_name, save_path, dt=0.02, dt_write=10, istep=1, tin=0, num_sol=1,
                  basis = "cc-pvdz", xc = "b3lyp", ch=0, scf_cycles=500):
    """QM along a trajectory. NOT TESTED YET

    Args:
        u (_type_): _description_
        dye_name (_type_): _description_
        save_path (_type_): _description_
        dt (float, optional): _description_. Defaults to 0.02.
        dt_write (int, optional): _description_. Defaults to 10.
        istep (int, optional): _description_. Defaults to 1.
        tin (int, optional): _description_. Defaults to 0.
        num_sol (int, optional): _description_. Defaults to 1.
        basis (str, optional): _description_. Defaults to "cc-pvdz".
        xc (str, optional): _description_. Defaults to "b3lyp".
        ch (int, optional): _description_. Defaults to 0.
        scf_cycles (int, optional): _description_. Defaults to 500.
    """
    start_time = time.time()

    #The P and O3 atoms from the linkers to be replaced by H
    del_list = [[0,["P","OP1","OP2"],["P"]]]

    a_save = []
    b_save = []
    for ts in u.trajectory[num_sol*(istep-1):num_sol*istep:dt]:

        print("BEFORE: ",lib.num_threads())
        print("--- %s seconds ---" % (time.time() - start_time))

        resnums = np.unique(u.select_atoms(f'resname {dye_name}').resid)

        #The H will be placed in the position of P of res F06 & of P on the DNA res 
        cap_pos = [[0,[(resnums[0],'P')]]]

        # Replacing with opt molecules (saving pdb to check) 
        mol1, mol2 = cp.pdb_cap(u,resnums[0], resnums[1], resnames=[dye_name[:2]+'A', dye_name[:2]+'B'],
                                del_list=del_list, path_save=save_path+'dimerCap.pdb', MDA_selection='all',
                                cap_list=[cap_pos,cap_pos], mol_name=dye_name)


        save_V, save_abs, V_list = do_QM(u, resnums, del_list, [cap_pos,cap_pos], basis=basis, xc=xc, ch=ch, scf_cycles=scf_cycles, start_time=start_time)

        t_i = round((ts.frame*dt_write + tin),2)

        a_save.append([t_i] + save_V)
        b_save.append(save_abs)

        # Create target Directory if it doesn't exist
        label = ''
        if not os.path.exists(save_path+label):
            os.mkdir(save_path+label)
            print("Directory ", label,  " Created ")
        else:
            print("Directory ", label,  " already exists")

        i_step = str(istep)
        np.savetxt(save_path+label+"/couplings_"+i_step+".txt", np.array(a_save), fmt=['%.1f','%1.4e','%1.4e','%.4f'])
        np.savetxt(save_path+label+"/absspectest_"+i_step+".txt", np.array(b_save), fmt='%1.4e')

    return

def do_QM(u, resnums, dels=[], caps=[[],[]], basis = "cc-pvdz", xc = "b3lyp",
          ch=0, spin=1, scf_cycles=500, start_time=0):
    """Auxiliary function for TDDFT calculation.

    Args:
        u (MDAnalysis Universe): The dimer system.
        resnums (list): Residue IDs for the dimer monomers
        dels (list, optional): List with atom to delete. Defaults to [].
        caps (list, optional): List with atoms to cap. Defaults to [[],[]].
        basis (str, optional): Basis set for DFT. Defaults to "cc-pvdz".
        xc (str, optional): Exchange Functional. Defaults to "b3lyp".
        ch (int, optional): Total charge of molecule. Defaults to 0.
        spin (int, optional): Total spin of molecule. Defaults to 1.
        scf_cycles (int, optional): Number of SCF cycles. Defaults to 500.
        start_time (int, optional): Time where Q-Chem was started. Defaults to 0.

    Returns:
        -Results on coupling
        -Absorption spectra results
        -List of couplings for excitations with significant oscillator strength
        -Dimer distance vector
    """
    if len(caps[0]) == 0:
        xyzA, xyzB, RAB = cp.Process_MD(u, resnums[0], resnums[1], cap=None)
    else:
        xyzA, xyzB, RAB = cp.Process_MD(u, resnums[0], resnums[1], del_list=dels, cap_list=caps)

    counter = 0

    molA,mfA,o_A,v_A = cp.do_dft(xyzA, basis=basis, xc_f=xc, mol_ch=ch, spin=spin,
                                verb=4, scf_cycles=scf_cycles)
    molB,mfB,o_B,v_B = cp.do_dft(xyzB, basis=basis, xc_f=xc, mol_ch=ch, spin=spin,
                                verb=4, scf_cycles=scf_cycles)

    print("AFTER dft: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))

    #TDDFT
    print(o_A.shape, v_A.shape)
    TenA, TdipA, O_stA, tdmA, osidA = cp.do_tddft(mfA,o_A,v_A,state_id=0,tda=True)
    TenB, TdipB, O_stB, tdmB, osidB = cp.do_tddft(mfB,o_B,v_B,state_id=0,tda=True)

    #Picking state with highest Osc strength (in case the first excitation is dark)
    all_states = 0
    V_Clist = []
    idx_list = []

    if isinstance(O_stA,list):
        all_idxA = np.argwhere(np.array(O_stA)>0.1)[:,0] if np.any(np.array(O_stA)>0.1) else [np.argmax(O_stA)]
        all_idxB = np.argwhere(np.array(O_stB)>0.1)[:,0] if np.any(np.array(O_stB)>0.1) else [np.argmax(O_stB)]
        all_states = np.concatenate((TenA, O_stA, TenB, O_stB))
        O_stA = O_stA[osidA]
        TenA = TenA[osidA]
        print('State with max OSt in A',osidA)
        O_stB = O_stB[osidB]
        TenB = TenB[osidB]
        print('State with max OSt in B',osidB)
        V_Cfull, cK = cp.V_Coulomb(molA, molB, tdmA[osidA], tdmB[osidB], calcK=True)
        for i in all_idxA:
            for j in all_idxB:
                if i!=j:
                    V, cK = cp.V_Coulomb(molA, molB, tdmA[i], tdmB[j], calcK=False)
                    V_Clist.append(V)
                    idx_list.append(str(i)+str(j))
        print('V indexes:',idx_list)
        #padding the list to avoid problems with saving the files. 10 is just the max size we'd probably have
        if len(idx_list)<=10:
            l_left = 10-len(idx_list)
            padd = [0]*l_left
            V_Clist += padd
            idx_list += padd
        else: # In case is longer than 10
            V_Clist = V_Clist[:10]

    else:
        V_Cfull, cK = cp.V_Coulomb(molA, molB, tdmA, tdmB, calcK=True)
    print("AFTER TDDFT: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    #Calculating V Coulombic with transition monopoles
    #__, chgA = cp.td_chrg_lowdin(molA, tdmA)
    #__, chgB = cp.td_chrg_lowdin(molB, tdmB)

    #Vtdipole = cp.V_multipole(molA,molB,chgA,chgB) #using monopole approx

    print("\n*** cK= %10.3E \n" % cK)

    print("AFTER VCoulomb: ",lib.num_threads())

    save_V = [V_Cfull*H_to_cm, RAB]#, Vtdipole]
    save_abs = [TenA*H_to_cm, TenB*H_to_cm, O_stA, O_stB]#, TdipA, TdipB] 
    V_list = [vi*H_to_cm for vi in V_Clist]

    return save_V, save_abs, V_list, RAB

def make_cluster_plots(df, max_comp, feat_list, dimer_labels, 
                       save_scatter_cluster, save_cat_cluster):
    """Function for generating plots

    Args:
        df (pandas dataframe): DataFrame with calculated features.
        max_comp (list): Features explaining PCA Variance
        feat_list (list): List of requested features
        dimer_labels (list): List with dimer labels
        save_scatter_cluster (str): Path for saving scatter plot with cluster
        save_cat_cluster (str): Path for saving scatter plot with clusters collored by dimers

    Returns:
        None
    """
    import seaborn as sns

    # Defining different colors for each label
    def palette(label_list):
        labels = label_list
        unique_labels = list(set(labels))
        color_palette = sns.color_palette("Set1", len(unique_labels))
        label_to_color = {label: color for label, color in zip(unique_labels, color_palette)}
        colors = [label_to_color[label] for label in labels]
        return color_palette

    # Cluster scatter plot
    import matplotlib
    font = {'family' : 'helvetica',
            'weight' : 'normal',
            'size'   : 16}

    matplotlib.rc('font', **font)

    # Scatter plot of clusters and PCA
    color_palette = palette(list(df["Cluster"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.5,3.1))
    a=sns.scatterplot(ax=axes[0], data=df, x=feat_list[max_comp[0]], y=feat_list[max_comp[1]],
                    hue=df["Cluster"], palette=color_palette)

    b=sns.scatterplot(ax=axes[1], data=df, x="Component 1", y="Component 2",
                    hue=df["Cluster"], palette=color_palette)
    plt.tight_layout(pad=0.8)
    plt.savefig(save_scatter_cluster)

    # Scatter plot of clusters and categories 
    color_palette = palette(dimer_labels)
    fig, axes = plt.subplots(1, 2, figsize=(7.5,3.18))
    a=sns.scatterplot(ax=axes[0], data=df, x=feat_list[max_comp[0]], y=feat_list[max_comp[1]],
                    hue=df["dtype"], palette=color_palette)

    b=sns.scatterplot(ax=axes[1], data=df, x="Component 1", y="Component 2",
                    hue=df["dtype"], palette=color_palette)
    plt.tight_layout(pad=0.8)
    a.legend(fontsize=10)
    b.legend(fontsize=10)
    plt.savefig(save_cat_cluster)

    return