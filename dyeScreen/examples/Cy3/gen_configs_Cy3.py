import os
import numpy as np
import pandas as pd
from dyeScreen.MD.sampling import scan_DNAconfig, gen_leap_nolinker, modify_mol2

# -6) Scan DNA configs 

# Amber paths (change if different)
# If saved as an environment variable
amber_path = os.getenv("AMBERPATH")
# If installed in miniconda
# amber_path = "/Users/username/opt/miniconda3/envs/AmberTools23"
# If installed in Nersc 
#amber_path = "/global/common/software/nersc/pm-2021q4/sw/amber/20" 

path_parmchk2 = amber_path + "/bin/parmchk2"

# Define input and output files
home =  os.path.dirname(os.path.abspath('__file__')) 
path = f"{home}/dyeScreen/examples/Cy3/cy3_full"
examples_home = f"{home}/dyeScreen/examples"

samples = path + "/samples"

leap_out = path + "/tleap_screen.in"
dye_mol2 = path + "/cy3-link.mol2"
dye_mol2_rewrite = path + "/cy3-final.mol2"
dye_pdb = path + "/cy3-link.pdb"
DNA_pdb =  examples_home + "/dna_duplex.pdb"

# Info on the dye molecule (can be extracted from the pdb file)
dye_name = "CY3"
# Atom names for the OPO3H group in the linker
opo3_1 = ['O1','P1','O4','O8','O7','H37']
opo3_2 = ['O2','P2','O3','O5', 'O6', 'H36']

# The atom names that participate in the bonding between the dye and DNA 
# (could be a default and not given by the user? It's not supposed to change)
attach_cy3 = ['P', "O3'"]*2
attach_dna = ["O3'", "P"]*2


# Sample only the positions within 18.5A of separation and include a box of DNA of 20A
nsamples = scan_DNAconfig(dye_pdb, DNA_pdb, samples, resD=1, resL=None, 
                          chainDNA=None, dist_min=18.5, DNABox=20, DNASt=20,
                          attachment='double', attach_points=[opo3_1,opo3_2], 
                          box_type="doubleAtt") 

# -7) Delete and rename atoms from mol2 so they match the pdb file
modify_mol2([opo3_1, opo3_2], dye_mol2, mol2_out=dye_mol2_rewrite, 
            attachment='double', parmchk2=path_parmchk2)

# -8) Generate input files and run leap 
gen_leap_nolinker(samples, amber_path, attach_cy3, attach_dna,
                  dye_mol2_rewrite, dye_name=dye_name, wbox=10)

# -8) Prepare files for MD run

# Example of SLURM prefix inputs 
def prefix_cluster(nodes, max_time, mail):
    pref = f"""#!/bin/bash
#SBATCH -N {nodes}
#SBATCH --ntasks={int(32)*nodes}
#SBATCH --job-name="cy3-qm"
#SBATCH --mail-user={mail}
#SBATCH --mail-type=ALL
#SBATCH --output=qm_run.log
"""
    return pref

def prefix_nersc(nodes, max_time, mail):
    pref = f"""#!/bin/bash
#SBATCH --qos=debug    
#SBATCH -N {nodes}
#SBATCH -t 00:{max_time}:00
#SBATCH --constraint=cpu
#SBATCH --job-name="cy3-qm"
#SBATCH --mail-user={mail}
#SBATCH --mail-type=ALL
#SBATCH --output=qm_run.log
"""
    return pref

# Check which dimers we actually generated MD input files for
i_start, i_end = 0, 80
valid_dimers = []
for i in range(i_start, i_end+1):
    dfile = f"{samples}/dimer_{i}_clean.rst7"
    if os.path.exists(dfile):
        valid_dimers.append(i)
print(valid_dimers)       

''' 
# To be run on an HPC environment
for dimer_num in valid_dimers:
    print(dimer_num)
    slurm_prefix = prefix_debug(4, dimer_num) 
    md_run(dimer_num, samples, amber_path, sample_frefix='dimer_', pdb=None, param=None, coord=None, 
            input_prefix='DNA-dye', cutoff=12.0, edges_rest=10.0,
            min_cycles=[2000,2000], eq_runtime=[20,1000], prod_runtime=4000, dt=0.002,
            nodes=1, tasks=32, logfile='', 
            sander_path='srun $AMBERPATH/bin/sander.MPI', slurm_prefix=slurm_prefix)
'''

# -9) Evaluate features
import dyeScreen.QM.cluster_trajs as cl

feats_file = '/feats.txt'
traj_path = f"{samples}/prods"

dimers = [0, 1, 2, 8, 17, 54]#, 22, 23]
labels = [f'dimer {i}' for i in dimers]

params = [ f'{traj_path}/dimer_{d}_clean.prmtop' for d in dimers ]
trajs = [ f'{traj_path}/dimer_{d}_prod.nc' for d in dimers ] #+ [ f'{samples}/dimer_{d}_prod_2.nc' for d in dimers0 ]
feat_list = ['cofm', 'angle', 'disp-long', 'disp-short']
vec_atoms = [[],["C5","C17"],["C5","C17"], ["C4","C6"]]
print(params ,'\n',trajs, '\n', feat_list, vec_atoms)

#feat_matrix = cl.make_feat_matrix(trajs, feat_list, params, dye_res='CY3', dt=2, dt_write=10, num_sol=200, 
#                                  vec_atoms=vec_atoms, save_path=samples+"/geom_data",
#                                  feat_file=feats_file, print_tstep=True)

df= pd.read_csv(samples + "/geom_data" + feats_file, sep=" ", header=None)
df_pca_kmeans, kmeans_pca, scores, max_comp = cl.PCA_cluster(df, feat_list, dimers=dimers, print_tstep=True, 
                                                             label_path=samples + "/name_info.txt")
print(df_pca_kmeans.head())

# Save plots
save_scatter_cluster = samples + "/geom_data/clusters.eps"
save_cat_cluster = samples + "/geom_data/clusters_dyes.eps"

cl.make_cluster_plots(df_pca_kmeans, max_comp, feat_list, dimers, 
                      save_scatter_cluster, save_cat_cluster)

# Find representative samples

reps = cl.find_cluster_rep(kmeans_pca, scores)
nclusters = kmeans_pca.n_clusters
rep_labels = [df_pca_kmeans['dtype'].iloc[reps[k]] for k in range(nclusters)]
rep_samples = [df_pca_kmeans['traj'].iloc[reps[k]] for k in range(nclusters)]
print("****", rep_labels)

# Traj files of representative samples
params_rep = [ f'{traj_path}/dimer_{d}_clean.prmtop' for d in rep_samples ]
trajs_rep = [ f'{traj_path}/dimer_{d}_prod.nc' for d in rep_samples ]


# Samples at the limit of the clusters
num_lim = 3
lim = cl.find_cluster_limits(kmeans_pca, scores, num_max=num_lim)
nclusters = kmeans_pca.n_clusters
lim_labels = [np.unique([df_pca_kmeans['dtype'].iloc[lim[k][j]] for j in range(num_lim)])[0] for k in range(nclusters)]
lim_samples = [np.unique([df_pca_kmeans['traj'].iloc[lim[k][j]] for j in range(num_lim)])[0] for k in range(nclusters)]
print("**", lim_labels)

# QM run
for r, rep in enumerate(reps):
    cl.get_sample_pdb(df_pca_kmeans, rep, traj_path, samples+"/geom_data", 2, r+1, dye_res='CY3', tcol='traj')
    #save_V, save_abs, V_list = cl.QM_of_sample(df_pca_kmeans, rep, traj_path, samples+"/geom_data", 2, r+1, dye_res='CY3',
    #                                           tcol='traj', basis = "3-21g", xc = "b3lyp",
    #                                           ch=1, sp=1, scf_cycles=500, mode="Vexc", slurm_prefix=prefix_cluster(1, 1, "email@mit.edu"))
    #print("Final V is: ", save_V)
  
