#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for processing a plotting data

Created on Thu May 27 17:31:42 2021

@author: mariacm
"""

import numpy as np
import scipy.linalg
from pyscf import gto, scf, tdscf, lib, dft, lo, solvent

#MD analysis tools
import MDAnalysis

import matplotlib.pyplot as plt
import matplotlib as mpl


kJ_to_ha = 1/2625.5
ha_to_ev = 219474.6#27.211

#Getting data 

def get_data(path, name, rep, typ, ntrajs=6, dt=20, traji=1, 
             subpath="t_", file_fmt="/Dimer%s_%s_dft.txt"):
    """
    Gets trajectory data 

    Parameters
    ----------
    path : str
        location of QM from trajectories folders.
    name : str
       property name.
    rep : str
        name given to trajectory repetition.
    typ : str
        system's name: 'I' or 'II'
    ntrajs : int, optional
        Total number of trajectories to load. The default is 6.
    dt : int, optional
        time step. The default is 20.
    traji : int, optional
        Starting trajectory to load. The default is 1.
    subpath : string, optional
        Format of the trajectory segment folders. The default is "t_".
    file_fmt : TYPE, optional
        How the file names are formatted with (typ,name) order. The default is "/Dimer%s_%s_dft.txt".

    Returns
    -------
    prop : ndarray with the loaded QM data.
    t_array : time array.

    """
    # file_i = lambda i: path + subpath + rep + str(i) +file_fmt%(typ,name)
    file_i = lambda i: path + subpath + rep  +file_fmt%(name,str(i))
    cols = np.loadtxt(file_i(1)).shape[1]
    #print(file_i(1))
    prop = np.empty((0,cols),dtype=float)
    
    for ic in range(traji,ntrajs+1):

        prop_i = np.loadtxt(file_i(ic))
        len_i = len(prop_i)
        prop = np.concatenate((prop, prop_i), axis=0)    
    
    ntrajs -= traji-1    
    t_array = np.linspace(0,len_i*ntrajs*dt,len_i*ntrajs+1)[:-1]

    return prop, t_array


def plot_indiv(tarray, data, data_label='',labels_i=[], xlabel='t(ps)', ylabel='',
               xrange=(0,2000), yrange=(0,1), lw=1, mode='lines', save_path=None, format_save='eps', 
               size=[7.5, 4], colors=None, styles=None, marker=None, alphas=1):
    """
    Function for plotting multiple individual curves in a single plot

    Parameters
    ----------
    tarray : ndarray
        time array.
    data : list
        List with data to plot 
        (each element being an array with a single plot y axis data).
    data_label : str, optional
        Fixed part of plot labels (i.e. hared by all plots).
    labels_i : list, optional
        list with individual curve labels. 
    xlabel : str, optional
        x axis label. Use $text$ for latex equation-style text. The default is 't(ps)'.
    ylabel : str, optional
        y axis label. Use $text$ for latex equation-style text. The default is ''.
    xrange : tuple, optional
        x axis (min,max) limits. The default is (0,2000).
    yrange : tuple, optional
        y axis (min,max) limits. The default is (0,1).
    lw : list or int, optional
        list of line widths for plotting. The default is 1.
    mode : str, optional
        indicates whether the curves want to be plotted with varying line_style ('lines')
        or markers ('markers'). The default is 'lines'.
    save_path : str, optional
        Path to save plot files. The default is None, meaning plot is not saved.
    format_save : str, optional
        Format of the saved plot. The default is 'eps'.
    size : list, optional
        Size of the plot. The default is [7.5, 4].
    colors : list, optional
        If provided, the given colors of the lines are used. 
        If not provided, a set of colorblind-friendly colors are used.
    styles : list, optional
        If provided, the given linestyles are used instead of the default.
    marker : list, optional
        If provided, the given markers are used instead of the default.
    alphas : list or int, optional
        list of transparency of each line. The default is 1.

    """

    font_size = 22
    font_family = 'arial'  
    if colors is None:
        # line cyclers adapted to colourblind people
        colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
    if styles is None:
        styles = ["-", "--", "-.", ":", "-", "--", "-."]
    if marker is None:
        marker = ["4", "2", "3", "1", "+", "x", "."]
    cyc_len = len(data) # to make sure all have the same length
    
    
    from cycler import cycler
    line_cycler   = (cycler(color=colors[:cyc_len]) +
                     cycler(linestyle=styles[:cyc_len]))
    marker_cycler = (cycler(color=colors[:cyc_len]) +
                     cycler(linestyle=["none"]*cyc_len) +
                     cycler(marker=marker[:cyc_len]))
    
    # Set Font Parameters
    font = {'family': font_family, 'size': font_size}
    mpl.rc('font', **font)   

    
    fig = plt.figure(figsize=size)
    main_ax = fig.add_axes([0.12, 0.08, 0.705, 0.88])
    if mode == 'lines':
        main_ax.set_prop_cycle(line_cycler)
    elif mode =='markers':
        main_ax.set_prop_cycle(marker_cycler)
    else:
        raise NotImplementedError('Mode variable is incorrect')


    # Making sure style parameters are correct
    if isinstance(lw,list):
        if len(lw) != len(data):
            raise ValueError('Length of lw must be same as data or a single int')
    else:
        lw = [lw]*len(data)
        
    if isinstance(alphas,list):
        if len(alphas) != len(data):
            raise ValueError('Length of alphas must be same as data or a single int')
    else:
        alphas = [alphas]*len(data)
    
    for k,dat in enumerate(data):
        if len(labels_i) != 0:
            main_ax.plot(tarray, dat, linewidth=lw[k],markersize=lw[k],
                         label=r'%s%s'%(data_label,labels_i[k]),alpha=alphas[k])
        else:
            main_ax.plot(tarray, dat, linewidth=lw[k],markersize=lw[k],alpha=alphas[k])
            
        
    main_ax.set_xlabel(r'%s'%xlabel)
    main_ax.set_ylabel(r'%s'%ylabel)
    main_ax.grid(False)
    
    xmin,xmax = xrange
    ymin,ymax = yrange
    main_ax.set_xlim(xmin,xmax)
    main_ax.set_ylim(ymin,ymax)
    main_ax.tick_params(axis='both', direction='in')
    
    leg = main_ax.legend(loc='upper right',fontsize=16,framealpha=0.8,labelspacing=0.2) 
    leg.get_frame().set_linewidth(0.0)
    if save_path:
        plt.savefig(save_path, format=format_save,bbox_inches="tight")
    
    return None

def plot_scale(tarray, data_matrix, bs_min, bs_max, x_label, y_label, xrange=(0,1000), 
               yrange=(0,1), lw=0.8, size=[7., 3.5], save_path=None, format_save='eps'):
    
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm, Normalize
    
    colormap = 'viridis'

    if xrange:
        xmin, xmax = xrange
    else: 
        xmin, xmax = (0,len(data_matrix))
    ymin, ymax = yrange
    
    bs = np.linspace(bs_min, bs_max-1,bs_max)
    cmap = cm.get_cmap(colormap)
    norm = Normalize(vmin=bs_min, vmax=bs_max)
    z = [[0, 0], [0, 0]]
    cbar_object = plt.pcolor(z, cmap=cmap, norm=norm)
    plt.clf()   

    # Make Figure & Axes
    fig = plt.figure(figsize=size)
    main_ax = fig.add_axes([0.12, 0.08, 0.705, 0.88])
    c_ax = fig.add_axes([0.88, 0.08, 0.05, 0.88])

    # Generate Plot
    for b in bs:
        main_ax.plot(tarray, data_matrix[:,int(b)], color=cmap(norm(b)), linewidth=lw)
        
    main_ax.set_xlabel(r'%s'%x_label)
    main_ax.set_ylabel(r'%s'%y_label)
    
    main_ax.set_xlim(xmin,xmax)
    main_ax.set_ylim(ymin,ymax)

    # Make Color Bar
    c_bar = plt.colorbar(cbar_object, c_ax)
    c_ax.yaxis.set_label_position('left')
    c_ax.yaxis.set_ticks_position('right')
    
    fig.legend(loc='upper center',fontsize=12,framealpha=0.8,labelspacing=0.2)
    if save_path:
        plt.savefig(save_path, format=format_save,bbox_inches="tight")
    
    return None

def plot_map(data_mat,min_lim=None,max_lim=None,size=[7,4]):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.figure(figsize=size)
    ax = plt.gca()
    if not min_lim:
        im = ax.imshow(data_mat[:,:max_lim])
    else:
        if not max_lim:
            im = ax.imshow(data_mat[:,min_lim:])
        else:
            im = ax.imshow(data_mat[:,min_lim:max_lim])
            
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax=cax)
    
def plot_hist(data, binwidth=10, data_label='a', labels_i=[0],xlabel='data',ylabel='count',
              xrange=(0,2000),yrange=(0,1),lw=1,alphas=[0.5]*100,norm=False,save_path=None,
              colors=None,cmi=None,size=[7.5, 3.6], format_save='pdf'):
    
    font_size = 22
    font_family = 'arial'  
    
    from cycler import cycler
    if colors:
        color_cycler   = cycler(color=colors)
    else:
        # line cyclers adapted to colourblind people
        color_cycler   = cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"])
                     
    if cmi:
        color_cycler = cmi(np.linspace(0, 1, len(data)))

    # Set Font Parameters
    font = {'family': font_family, 'size': font_size}
    mpl.rc('font', **font)   

    
    fig = plt.figure(figsize=size)
    main_ax = fig.add_axes([0.12, 0.08, 0.705, 0.88])
    main_ax.set_prop_cycle(color_cycler)
    
    # Making sure style parameters are correct
    if isinstance(lw,list):
        if len(lw) != len(data):
            raise ValueError('Length of lw must be same as data or a single int')
    else:
        lw = [lw]*len(data)
          
    if isinstance(alphas,list):
        if len(alphas) != len(data):
            raise ValueError('Length of alphas must be same as data or a single int')
    else:
        alphas = [alphas]*len(data)
      
    labels_i = labels_i*len(data) #In case labels are not given 
    
    
    for k,dat in enumerate(data):
        if norm: #Normalized histogram aka prob distribution
            weights = np.ones_like(dat) / len(dat)
        else:
            weights = None
        main_ax.hist(dat, bins=np.arange(min(dat), max(dat) + binwidth, binwidth), linewidth=lw[k], stacked=True, 
                     alpha=alphas[k], label=r'$%s%s$'%(data_label,labels_i[k]), weights=weights)
        
    main_ax.set_xlabel('%s'%xlabel)
    main_ax.set_ylabel('%s'%ylabel)
    
    main_ax.tick_params(axis='both', direction='in')
    xmin,xmax = xrange
    ymin,ymax = yrange
    main_ax.set_xlim(xmin,xmax)
    main_ax.set_ylim(ymin,ymax)
    
    if labels_i[0] != 0:
        main_ax.legend(loc='upper right',fontsize=16,framealpha=0.8,labelspacing=0.2,frameon=False) 
    else:
        plt.plot()
    if save_path:
        plt.savefig(save_path, format=format_save, bbox_inches="tight")
    
    return None

def sel_dihedrals(sel_list, all_names, dih_names):
    
    atom1,atom2,atom3,atom4 = dih_names
    dih_names = np.vstack((atom1,atom2,atom3,atom4)).transpose()
    print(dih_names)
    idx_sel = []
    for i in sel_list: idx_sel.append(np.where(all_names==i)[0][0])

    names_left = np.delete(all_names, idx_sel)
    print(names_left)

    #Lopping over torsions
    dih_red = []
    red_names = []
    for idx,i in enumerate(dih_names):
        if not any(np.in1d(i, names_left)):
            dih_red.append(idx)
            red_names.append(i)
            #print(i) 
    
    return dih_red,red_names

def get_qm(path,rep, ntrajs=8,typ='',dt=10,traji=1):
    '''

    Parameters
    ----------
    path : str with file location (for type I or type II)
    rep : Whether the trajectories are rep "A" or "B"
    ntrajs : number of sub-trajectrory files

    Returns
    -------
    

    '''
    file_coup = lambda i: path+"t_"+rep+str(i)+"/Dimer%s_couplings_dft.txt"%typ
    file_tint = lambda i: path+"t_"+rep+str(i)+"/Dimer%s_tintegrals_dft.txt"%typ
    
    c_cols = np.loadtxt(file_coup(1)).shape[1]
    t_cols = np.loadtxt(file_tint(1)).shape[1]
    coup = np.empty((0,c_cols),dtype=float)
    tint = np.empty((0,t_cols),dtype=float)

    
    for ic in range(traji,ntrajs+1):

        coup_i = np.loadtxt(file_coup(ic))
        tint_i = np.loadtxt(file_tint(ic))
        len_i = len(coup_i)
        coup = np.concatenate((coup, coup_i), axis=0)
        tint = np.concatenate((tint, tint_i), axis=0)        
    
    ntrajs -= traji-1    
    t_array = np.linspace(0,len_i*ntrajs*dt,len_i*ntrajs+1)
    #Couplings
    V_coul = abs(coup[:,1])
    V_ct = coup[:,2] 
    RAB = coup[:,3] 

    #transfer integral
    te = tint[:,1] 
    th = tint[:,2] 
    #tes = tint[:,3]
    #ths = tint[:,4]

    return t_array[:-1],V_coul,V_ct,te,th,RAB


def VCT_calc(RAB, U, te, th, t_array,path_save=None):
    # Old code I used when the VCT in the out file was wrong
    
    RAB_1, RAB_2 = RAB
    te_1, te_2 = te
    th_1, th_2 = th

    #Constants
    J_to_cm = 5.034*(10**22)
    e = 1.60217662 * (10**(-19))
    den = 9 * 10**9 #1/4*pi*eps0
    er = 77.16600 #water at 301.65K and 1 bar
    
    VCTA = e**2 * den/RAB_1 *(10**10) /er * J_to_cm
    VCTB = e**2 * den/RAB_2 *(10**10) /er * J_to_cm
    
    #JCT
    JCTA = -2*te_1*th_1/(U-VCTA)
    JCTB = -2*te_2*th_2/(U-VCTB)
    
    if path_save:
        np.savetxt(path_save+'Dimer_same/VCT_correctedI.txt',np.vstack((t_array,te_1,th_1,VCTA,JCTA)).T)
        np.savetxt(path_save+'Dimer_opp/VCT_correctedII.txt',np.vstack((t_array,te_2,th_2,VCTB,JCTB)).T)
    
    return JCTA, JCTB
