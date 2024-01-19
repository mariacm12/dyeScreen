import numpy as np
import subprocess
import MDAnalysis as mda

from dyeScreen.commons.geom_utils import get_RAB, find_term_res_pdb

class md_samples():
    def __init__(self, amber_path, nres, file_prefix, cutoff=12.0, sander_path=None, edge_res=None, edge_rest=10.0) -> None:
        """MD samples class

        Args:
            amber_path (str): Path to AmberTools executables.
            nres (int): Number of residues excluding water and ions. 
            file_prefix (str): Dimer sample prefix including ID.
            cutoff (float, optional): Cutoff distance. Defaults to 12.0.
            sander_path (str, optional): Path to sander executable. Defaults to None for standard location.
            edge_res (list, optional): List with sel groups to be restrained. Defaults to None.
            edge_rest (float, optional): Restrain in kcal/mol-A**2. Defaults to 10.0.
        """
        self.amber = amber_path
        self.sander = sander_path
        if self.sander == None:
            self.sander = self.amber + "/bin/sander"
        self.f_prefix = file_prefix
        self.cutoff = cutoff
        self.nres = nres
        # Optionally constrain DNA edges
        res_str = ""
        if len(edge_res)>0:
            res_str = f'\n  restraint_wt = {edge_rest},     ! kcal/mol-A**2 force constant\n'
            res_list = ",".join(map(str, edge_res))
            res_str += f"  restraintmask = '(:{res_list})',"
        self.edgeRest = res_str
        # Standard harmonic restrains
        self.min_rest = 500.0
        self.eq_rest = 10.0
        # Standard temperature
        self.rtemp = 300.0

    def make_min_input(self, rest, edge_str, ncycles):
        """Sets up the Amber files for the constrained & unconstrained minimization

        Args:
            rest (int, float): Restrain to impose on system for constrained minimization.
            edge_str (str): String defining the system edges (in sel language).
            ncycles (list): Number of cycles for 2-step minimization as [int, int].
        """

        # Restrains
        ntr_i = 0
        if edge_str:
            ntr_i = 1

        min1_ncycles, min2_ncycles = ncycles[0], ncycles[1]
        file_min1 = self.f_prefix + "_min1.in"
        file_min2 = self.f_prefix + "_min2.in"
        clean_prefix = self.f_prefix.split("/")[-1]
        print(clean_prefix)

        with open(file_min1, 'w') as f:
            f.write(f"{clean_prefix}: Initial minimization of solvent + ions\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 1,
  maxcyc = {min1_ncycles+2000},
  ncyc   = {min1_ncycles},
  ntb    = 1,
  ntr    = 1,
  iwrap  = 1,
  cut    = {self.cutoff}
/
Hold the DNA fixed
{rest}
RES 1 {self.nres}''')
            f.write('\nEND\nEND')

        with open(file_min2, 'w') as f:
            f.write(f"{clean_prefix}: Initial minimization of entire system\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 1,
  maxcyc = {min2_ncycles+2000},
  ncyc   = {min2_ncycles},
  ntb    = 1,
  ntr    = {ntr_i},
  iwrap  = 1,{edge_str}
  cut    = {self.cutoff},\n/\n''')

        return file_min1.split("/")[-1], file_min2.split("/")[-1]

    def make_heat_input(self, tF, rest, total_ps, dt, twrite):
        """Sets up the Amber file for the constrained heating

        Args:
            tF (int, float): Final temperature.
            rest (int, float): Restrain to impose on system for constrained heating.
            total_ps (int): Total heating time in ps.
            dt (int): Time step of simulation in ps.
            twrite (int): Time step for writing trajectrories and output in ps.
        """
        nsteps = int(total_ps/dt) 
        nsteps_x = int(twrite/dt) # How often to print trajectory
        file_heat = self.f_prefix + "_eq1.in"
        clean_prefix = self.f_prefix.split("/")[-1]

        rest_str = ""
        with open(file_heat, 'w') as f:
            f.write(f"{clean_prefix}: {total_ps}ps MD equilibration with restraint on DNA\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 0,
  irest  = 0,
  ntx    = 1,
  ntb    = 1,
  iwrap  = 1,
  cut    = {self.cutoff}, !non-bonded cutoff in Ams
  ntr    = 1,
  ntc    = 2,
  ntf    = 2,
  tempi  = 0.0,
  temp0  = {tF},
  ntt    = 3,
  gamma_ln = 5.0,
  ig=-1,
  nstlim = {int(nsteps)},
  dt = {dt},
  ntpr = {nsteps_x}, !How often to write in out file
  ntwx = {nsteps_x}, !How often to write in traj file
  ntwr = {nsteps_x}, !How often to write in restart file
  ioutfm = 0
/
Hold the DNA fixed with weak restrains
{rest}
RES 1 {self.nres}''')
            f.write('\nEND\nEND')
        return file_heat.split("/")[-1]

    def make_eq_input(self, rest, temp, total_ps, dt, twrite):
        """Sets up the Amber file for the unrestrained equilibration

        Args:
            rest (int, float): Restrain to impose on system for constrained heating.
            temp (int, float): Reference temperature (final temp of heating)
            total_ps (int): Total equilibration time in ps.
            dt (int): Time step of simulation in ps.
            twrite (int): Time step for writing trajectrories and output in ps.
        """
        nsteps_x = int(twrite/dt) # How often to print trajectory
        rest_str = ""
        clean_prefix = self.f_prefix.split("/")[-1]
        def eq_file(steps, total_t, eqfile):
            with open(eqfile, 'w') as f:
                f.write(f"{clean_prefix}: {total_t}ps MD equilibration\n")
                f.write(" &cntrl\n")
                f.write(f'''  imin   = 0,
  irest  = 1,
  ntx    = 5,
  ntp    = 1,
  pres0  = 1.0, !Reference pressure
  taup   = 2.0, !Pressure relaxation time of 2ps
  ntc    = 2,     !SHAKE to constrain bonds with H
  cut    = {self.cutoff},  !non-bonded cutoff in Ams
  ntf    = 2, !Don't calculate forces for H-atoms
  ntt    = 3, gamma_ln = 5.0, !Langevin dynamics with collision frequency 5
  temp0  = {temp}, tempi  =  {temp}   !Reference (constant) temperature
  nstlim = {int(steps)},
  dt = {dt},
  ntpr = {nsteps_x}, !How often to write in out file
  ntwx = {nsteps_x}, !How often to write in traj file
  ntwr = {nsteps_x}, !How often to write in restart file
  iwrap  = 1,      !Coordinates written to restart & traj are "wrapped" into a primary box.
  ioutfm = 0,      !Formatted ASCII trajectory
  ntr    = 0,
  ig     = -1
/
Hold the DNA fixed with weak restrains
{int(rest)/2}
RES 1 {self.nres}''')
            f.write('\nEND\nEND')
                
            return eqfile
        
        # Base equilibration
        nsteps = total_ps / dt 
        file_eq = eq_file(nsteps, total_ps, self.f_prefix + "_eq2.in")
        #nsteps = total_ps[1] / dt 
        #file_eq2 = eq_file(nsteps, total_ps[1], self.f_prefix + "_eq-extra.in")
        return file_eq.split("/")[-1]

    def make_prod_input(self, edge_str, temp, total_ps, dt, twrite):
        """Sets up the Amber file for the production run

        Args:
            edge_str (_type_): Selection string with atom groups to be restrained. 
            rest (int, float): Restrain to impose on system for constrained heating.
            temp (int, float): Reference temperature (final temp of heating)
            total_ps (int): Total equilibration time in ps.
            dt (int): Time step of simulation in ps.
            twrite (int): Time step for writing trajectories and output in ps.
        """
        # Restrains
        ntr_i = 0
        if edge_str:
            ntr_i = 1

        nsteps = int(total_ps/dt)
        nsteps_x = int(twrite/dt) # How often to print trajectory
        prodfile = self.f_prefix + "_prod.in"
        clean_prefix = self.f_prefix.split("/")[-1]
        rest_str = ""
        with open(prodfile, 'w') as f:
            f.write(f"{clean_prefix}: {total_ps/1000}ns MD production\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 0,
  irest  = 1, !Restart simulation form saved restart file
  ntx    = 5, !Coordinates and velocities read from a NetCDF file
  ntp    = 1,   !constant pressure (isotropic scaling)
  pres0  = 1.0, !Reference pressure
  taup   = 2.0, !Pressure relaxation time of 2ps
  ntc    = 2,     !SHAKE to constrain bonds with H
  cut    = {self.cutoff},  !non-bonded cutoff of 12A
  ntf    = 2, !Don't calculate forces for H-atoms
  ntt    = 3, gamma_ln = 5.0, !Langevin dynamics with collision frequency 5
  temp0  = {temp}, tempi  =  {temp}            !Reference temperature
  nstlim = {nsteps},  !Number of MD steps
  dt     = {dt},  !2fs time-step
  ntpr   = {nsteps_x},    !How often to write in out file: Every 10ps
  ntwx   = {nsteps_x},   !How often to write in mdcrd file
  ntwr   = {nsteps_x*10},   !How often to write in restrt file
  iwrap  = 1,      !Coordinates written to restart & traj are "wrapped" into a primary box.
  ioutfm = 0,      !Formatted ASCII trajectory
  ntr    = {ntr_i},{edge_str}
  ig     = -1,\n/\n''')

        return prodfile.split("/")[-1]
    
    def run_md(self, path, sampleFile, param, coord, min_cycles, eq_time, prod_time, dt, save_time,
               nodes, tasks, logfile, slurm_prefix):
        """Write bash file and run MD with Amber

        Args:
            path (str): Location of param and coord files
            sampleFile (str): Dimer sample prefix including ID.
            param (str): Name of Amber parameter file.
            coord (str): Name of Amber coordinate file.
            min_cycles (list, optional): [constrained, unconstrained] minimization cycles. 
            eq_time (list, optional): Run time in ps for [heat, const eq, unconst eq].
            prod_time (int, optional): Run tume for production in ps. 
            dt (float, optional): Time step in ps. 
            save_time (int): Time step for writing trajectories and output in ps.
            nodes (int, optional): Number of nodes for HPC run. 
            tasks (int, optional): Number of tasks for HPC run. 
            logfile (str, optional): Custom name of log file for HPC run. 
            sander_path(str,optional): The path to sander executable. 
            slurm_prefix (str, optional): Custom prefix for SLURM file. Defaults to None for auto generated file.

        Returns:
            -Final production traj file
            -Final coordinate file
            -job ID for last run 
        """
        
        # Initialize bash script
        run_file = f"run_{sampleFile}.sh"
        f =  open(path+run_file, 'w')
        if slurm_prefix:
            f.write(slurm_prefix)
        else:
            f.write(f"#!/bin/sh\n#SBATCH --job-name={sampleFile}\n")
            f.write(f"#SBATCH --nodes={nodes}\n#SBATCH --ntasks={tasks}\n#SBATCH --output={logfile}\n\n")

        filemin1, filemin2 = self.make_min_input(edge_str=self.edgeRest, rest=self.min_rest, ncycles=min_cycles)

        ref = coord
        input1 = filemin1 
        summ1  = filemin1[:-2] + "out"
        coordi = coord
        coordf = sampleFile + "_min1.ncrst"
        

        f.write(f'{self.sander} -O -i {input1} -o {summ1} -p {param} '
                f'-c {coordi} -r {coordf} -ref {ref}\n'
                )
        
        ref = coordf
        input2 = filemin2
        summ2  = filemin2[:-2] + "out"
        coordi = coordf
        coordf = sampleFile + "_min2.ncrst"
                 
        f.write(f'{self.sander} -O -i {input2} -o {summ2} -p {param} '
                f'-c {coordi} -r {coordf} -ref {ref}\n'
                )
                
        fileheat = self.make_heat_input(tF=self.rtemp, 
                                        rest=self.eq_rest, total_ps=eq_time[0], dt=dt, twrite=0.2)
        fileeq   = self.make_eq_input(rest=self.eq_rest, temp=self.rtemp, 
                                        total_ps=eq_time[1], dt=dt, twrite=save_time)
        
        input1 = fileheat
        summ1  = fileheat[:-2] + "out"
        coordi = coordf
        ref    = coordf
        coordf = sampleFile + "_eq1.ncrst"
        traj   = sampleFile + "_eq1.nc"

        f.write(f'{self.sander} -O -i {input1} -o {summ1} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\n'
                )
        
        input2 = fileeq
        summ2  = fileeq[:-2] + "out"
        coordi = coordf
        coordf = sampleFile + f"_eq2.ncrst"
        traj   = sampleFile + f"_eq2.nc"
        
        f.write(f'{self.sander} -O -i {input2} -o {summ2} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\n'
                )
        
        fileprod = self.make_prod_input(edge_str=self.edgeRest, temp=self.rtemp, 
                                        total_ps=prod_time, dt=dt, twrite=save_time)
        input = fileprod
        summ  = fileprod[:-2] + "out"

        coordi = coordf
        coordf = sampleFile + "_prod.ncrst"
        traj   = sampleFile + "_prod.nc"

        f.write(f'{self.sander} -O -i {input} -o {summ} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\nwait\n'
                )
        f.close()
        
        jobID = subprocess.Popen(f"sbatch --parsable {run_file}", shell = True, cwd=path)
        return traj, coordf, jobID

    def run_extra_prod(self, path, sampleFile, param, coordf, prod_time, dt, save_time, iextra,
                       nodes, tasks, logfile, slurm_prefix):
        """ Run extra production 

        Args:
            path (str): Location of param and coord files
            sampleFile (str): Dimer sample prefix including ID.
            param (str): Name of Amber parameter file.
            coordf (str): Name of previously generated Amber coordinate file.
            prod_time (int, optional): Run tume for production in ps. 
            dt (float, optional): Time step in ps. 
            save_time (int): Time step for writing trajectories and output in ps.
            iextra (int): Starting index for coord and traj files (counting from previously saved files)
            nodes (int, optional): Number of nodes for HPC run. 
            tasks (int, optional): Number of tasks for HPC run. 
            logfile (str, optional): Custom name of log file for HPC run. 
            slurm_prefix (str, optional): Custom prefix for SLURM file. Defaults to None for auto generated file.

        Returns:
            -Final production traj file
            -Final coordinate file
            -job ID for last run 
        """
        
        # Initialize bash script
        run_file = f"run_{sampleFile}_extra.sh"
        f =  open(path+run_file, 'w')
        if slurm_prefix:
            f.write(slurm_prefix)
        else:
            f.write(f"#!/bin/sh\n#SBATCH --job-name={sampleFile}\n")
            f.write(f"#SBATCH --nodes={nodes}\n#SBATCH --ntasks={tasks}\n#SBATCH --output={logfile}\n\n")

             
        fileprod = self.make_prod_input(edge_str=self.edgeRest, temp=self.rtemp, 
                                        total_ps=prod_time, dt=dt, twrite=save_time)
        input_file = fileprod

        coordi = coordf
        ref = sampleFile + "_min2.ncrst"
        coordf = sampleFile + f"_prod_{iextra}.ncrst"
        traj   = sampleFile + f"_prod_{iextra}.nc"
        summ  = fileprod[:-3] + f"_{iextra}.out"

        f.write(f'{self.sander} -O -i {input_file} -o {summ} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\nwait\n'
                )
        f.close()
        
        jobID = subprocess.Popen(f"sbatch --parsable {run_file}", shell=True, cwd=path)
        return traj, coordf, jobID    
    
    def check_rmsd(self, param, trajs, step, rmsd_time, save_time, dt, dy, save_path=""):
        """Calculate RMSD in window [-dy, dy] along rmsd_time.

        Args:
            param (str): Parameter file name.
            trajs (list): List of trajectory files. 
            step (int): Dimer sample identifier (corresponding with file name).
            rmsd_time (int): Time defined to calculate RMSD in ps. 
            save_time (int): Time step for writing trajectories and output in ps.
            dt (float, optional): Time step in ps. Defaults to 0.002.
            dy (float, optional): Size of CofM window [-dy, dy]
            save_path (str): Directory path for saving output. 

        Returns:
            - Fraction (%) of data within window
            - Length of tested data
        """
        nsteps_x = save_time / dt
        nlast = rmsd_time / dt

        # Calculate rmsd of last trajectory with cpptraj
        script_cont=f"parm {param}"
        for traj in trajs:
            script_cont += f"\ntrajin {traj}"
        script_cont += f"\nrms ToFirst :1-{self.nres}&!@H= first out rmsd_{step}.txt mass\nrun\n"
        f = open(save_path + "get_rmsd.in", 'w')
        f.write(script_cont)
        f.close()
        cp = subprocess.Popen(f"source {self.amber}/amber.sh", shell = True)
        cp.wait()
        subprocess.Popen(f"{self.amber}/bin/cpptraj -i get_rmsd.in", shell = True, cwd=save_path).wait()

        # Test equilibration of last <rmsd_time>ps of rmsd (traj was generated every nwrite ps)
        rmsd = np.loadtxt(f"{save_path}rmsd_{step}.txt")[:,1]
        # window of time to test
        rmsd_last = rmsd[int(nlast/nsteps_x):]
        avg = np.mean(rmsd_last)

        # Test what % of the data stays inside the RMSD window [-dy, dy]
        avg_plus = avg + dy
        avg_minus = avg - dy
        within_wdw = (rmsd_last >= avg_minus) & (rmsd_last <= avg_plus)
        wdw_percent = (np.sum(within_wdw) / len(rmsd_last)) * 100

        return wdw_percent, len(rmsd_last)

    def calculate_cofm(self, param, trajs, step, sel_string, cofm_time, save_time, dt, dy, save_path):
        """Calculate Center of Mass in window [-dy, dy] along cofm_time.

        Args:
            param (str): Parameter file name.
            trajs (list): List of trajectory files. 
            step (int): Dimer sample identifier (corresponding with file name).
            sel_string (str): Residue name for dimer.
            cofm_time (int): Time defined to calculate cofm in ps. 
            save_time (int): Time step for writing trajectories and output in ps.
            dt (float, optional): Time step in ps. Defaults to 0.002.
            dy (float, optional): Size of CofM window [-dy, dy]
            save_path (str): Directory path for saving output. 

        Returns:
            - Fraction (%) of data within window
            - Length of tested data.
        """
        
        nsteps_x = save_time / dt
        nlast = cofm_time / dt

        # get resid of dyes
        u = mda.Universe(param, trajs[0], format="TRJ")
        res_list = u.select_atoms("resname " + sel_string).residues
        resid1 = res_list[0].resid
        resid2 = res_list[-1].resid

        # Calculate center of mass distance
        rabs = []
        for traj in trajs:
            rab = get_RAB(param, traj[:-3], '', ntrajs=1, dt=1, resnum1=str(resid1), resnum2=str(resid2))
            print(traj[:-3], rab.shape)
            rabs.append(rab)
        rabs = np.concatenate(rabs)
        np.savetxt(f"{save_path}cofm_{step}.txt", rabs, fmt='%.5f')

        # Test equilibration of last <cofm_time>ps of cofm (traj was generated every nwrite ps)

        # window of time to test
        rab_last = rabs[int(nlast/nsteps_x):]
        avg = np.mean(rab_last)

        # Test what % of the data stays inside the CofM window [-dy, dy]
        avg_plus = avg + dy
        avg_minus = avg - dy
        within_wdw = (rab_last >= avg_minus) & (rab_last <= avg_plus)
        wdw_percent = (np.sum(within_wdw) / len(rab_last)) * 100

        print(resid1, resid2)

        return wdw_percent, len(rab_last)
        
        
def md_run(sample, path, amber_path, sample_frefix='dimer_', pdb=None, param=None, coord=None, 
           cutoff=12.0, edges_rest=None, 
           min_cycles=[2000,2000], eq_runtime=[20,1000], prod_runtime=5000, dt=0.002,
           nodes=1, tasks=1, logfile="out.log", sander_path=None, slurm_prefix=None):
    """Running MD for the sample number <sample>: 2-step min, 2-step eq until rmsd convergence & production
        *** Note that this function is for running in a HPC SLURM environment

    Args:
        sample (idx): Dimer sample identifier (corresponding with file name).
        path (str): Directory location of pdb, param and traj files.
        amber_path (str): Directory location of AmberTools executables.
        sample_frefix (str, optional): File prefix of dimer sample files. Defaults to 'dimer'.
        pdb (str, optional): PDB file containing clean dimer sample if not auto generated.
        param (str, optional): Parameter file of dimer sample if not auto generated.
        coord (str, optional): Coordinate file of dimer sample if not auto generated.
        cutoff (float, optional): Amber simulation cutoff distance. Defaults to 12.0.
        edges_rest (float, optional): Restrain to be applied to edge residues in kcal/mol-A**2.
                                      Defaults to None for non-constained edges.
        min_cycles (list, optional): [constrained, unconstrained] minimization cycles. Defaults to [2000,2000].
        eq_runtime (list, optional): Run time in ps for [heat, const eq, unconst eq]. Defaults to [20,400,300].
        prod_runtime (int, optional): Run tume for production in ps. Defaults to 4000.
        dt (float, optional): Time step in ps. Defaults to 0.002.
        nodes (int, optional): Number of nodes for HPC run. Defaults to 1.
        tasks (int, optional): Number of tasks for HPC run. Defaults to 1.
        logfile (str, optional): Custom name of log file for HPC run. Defaults to "out.log".
        sander_path(str,optional): The path to sander executable. 
                                   Defaults to the executable in <amber_path>/bin if not provided.
        slurm_prefix (str, optional): Custom prefix for SLURM file. Defaults to None for auto generated file.
    """

    sampleF = sample_frefix + str(sample)
    if not param:
        param = sampleF + "_clean.prmtop"
    if not coord:
        coord = sampleF + "_clean.rst7"
    if not pdb:
        pdb = sampleF + "_clean.pdb"
    
    # Calculate number of non-solvent residues
    u = mda.Universe(path+pdb, format="PDB")
    nres = len(u.atoms.residues)

    # Find the edge residues
    ter_res = []
    if edges_rest:
        ter_res = find_term_res_pdb(path+pdb, nres, start_res=1)
        print(ter_res)

    md = md_samples(amber_path, nres, sampleF, cutoff, sander_path, ter_res, edges_rest)

    savetime = 10
    trajf, coordf, jobID = md.run_md(path, sampleF, param.split("/")[-1], coord.split("/")[-1], 
                                     min_cycles, eq_runtime, prod_runtime, dt, savetime,
                                     nodes=nodes, tasks=tasks, logfile=logfile, slurm_prefix=slurm_prefix)    
    return #trajf, coordf, jobID


def eq_check(sample, path, amber_path, sample_frefix='dimer_', pdb=None, trajs=None, dye_res='DYE',
             prod_runtime=5000, save_time=10, dt=0.002, rmsd_time=200, cutoff=12.0, edges_rest=None,
             metric='RMSD', iextra=2, wdw_min=10, pass_window=2,
             nodes=1, tasks=1, logfile='out.log', sander_path=None, slurm_prefix=None):

    """Once simulation is done, check if the trajectory is equilibrated and extend otherwise. 
        A trajectory is considered equilibrated when the % of data points inside {metric} window:
        [avg-{pass_window}, avg+{pass_window}], along {rmsd_time}ps, is larger than {wdw_min}%  

    Args:
        sample (idx): Dimer sample identifier (corresponding with file name).
        path (str): Directory location of pdb, param and traj files.
        amber_path (str): Directory location of AmberTools executables.
        sample_frefix (str, optional): File prefix of dimer sample files. Defaults to 'dimer'.
        pdb (str, optional): PDB file containing clean dimer sample if not auto generated.
        trajs (list, optional): List of trajectory files.
        dye_res (str, optional): Residue name for the dye. Defaults to 'DYE'.
        prod_runtime (int, optional): Run tume for production in ps. Defaults to 4000.
        save_time (int): Time step for writing trajectories and output in ps.
        dt (float, optional): Time step in ps. Defaults to 0.002.
        rmsd_time (int, optional): Length of time in ps used to calculate eq metric. Defaults to 200.
        cutoff (float, optional): Amber simulation cutoff distance. Defaults to 12.0.
        edges_rest (float, optional): Restrain to be applied to edge residues in kcal/mol-A**2.
                                      Defaults to None for non-constained edges.
        metric (str, optional): Metric to test eq {RMSD, CofMass}, case-insensitive. Defaults to 'RMSD'.
        iextra (int, optional): Starting index for addtional simulation. Defaults to 2.
        wdw_min (int, optional): Size of window in Ams used to test equilibration. Defaults to 10.
        pass_window (int, optional): The fraction (in %) for considering the . Defaults to 2.
        nodes (int, optional): Number of nodes for HPC run. Defaults to 1.
        tasks (int, optional): Number of tasks for HPC run. Defaults to 1.
        logfile (str, optional): Custom name of log file for HPC run. Defaults to "out.log".
        sander_path(str,optional): The path to sander executable. 
                                   Defaults to the executable in <amber_path>/bin if not provided.
        slurm_prefix (str, optional): Custom prefix for SLURM file. Defaults to None for auto generated file.


    Raises:
        NotImplementedError: Metric other than RMSD and CofM metrics is provided.

    Returns:
        Fraction (%) of samples within defined window. 
    """
    
    sampleF = sample_frefix + str(sample)
    if not trajs:
        traj1 = sampleF + "_eq2.nc"
        traj2 = sampleF + "_prod.nc"
        trajs = [traj1, traj2]
    param = sampleF + "_clean.prmtop"
    if not pdb:
        pdb = sampleF + "_clean.pdb"

    coordf = trajs[-1]+"rst"

    # Calculate number of non-solvent residues
    u = mda.Universe(pdb, format="PDB")
    nres = len(u.atoms.residues)

    # Find the edge residues
    ter_res = []
    if edges_rest:
        ter_res = find_term_res_pdb(path+pdb, dist_min=6)
        print(ter_res)

    md = md_samples(amber_path, nres, sampleF, cutoff, sander_path, ter_res, edges_rest)
    if metric.casefold() == "RMSD".casefold():
        wdw_percent, total = md.check_rmsd(param, trajs, sample, rmsd_time, save_time, dt, pass_window, save_path=path)
    elif metric.casefold() == "CofM".casefold():
        wdw_percent, total = md.calculate_cofm(param, trajs, sample, dye_res, rmsd_time, save_time, dt, pass_window, save_path=path)
    else:
        raise NotImplementedError("Only RMSD and CofM metrics are implemented")

    print(f'dimer_{sample} % of data points inside {metric} window [avg-{pass_window}, avg+{pass_window}], {rmsd_time}ps is {wdw_percent}%, with  min metric {wdw_min}%')
    if wdw_percent < wdw_min: # The data didn't passes enough times through the average
        # Run extra production if simulation not yet converged
        trajf, coordf, jobID = md.run_extra_prod(path, sampleF, param.split("/")[-1], coordf.split("/")[-1],
                                                 prod_runtime, dt, save_time, iextra=iextra,
                                                 nodes=nodes, tasks=tasks, logfile=logfile, slurm_prefix=slurm_prefix)
        print(f'starting production run #{iextra}')

    return wdw_percent


