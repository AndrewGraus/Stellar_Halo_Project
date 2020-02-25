import h5py, os
import numpy as np

#I'm doing something else here and grabbing the 
#from the merger tree files, there are less halos
#in the merger tree for a reason that I don't understand
#I'm hoping its because there very small halos are cut out

for data_file in os.listdir('./disk_files_full/'):
    halo_number = data_file.split('_')[2]

    print data_file
    
    f = h5py.File('./disk_files_full/'+data_file)
    print f.keys()
    snap = f['Snapshot00152']['HaloCatalog_RockstarMergerTree']
    Mvir = snap['Mvir'][:]
    Rvir = snap['Rvir'][:]
    cens = snap['Center'][:]
    Vmax = snap['Vmax'][:]
    Rmax = snap['Rmax'][:]
    Vpeak = snap['Vpeak_TotalMass'][:]
    ids_from_halo_cat = snap['Orig_halo_id'][:]
    Tree_id = snap['TreeNumber'][:]
    
    #now I need the most massive halo at z = 0
    host_index = np.argmax(Mvir)
    host_tree_id = Tree_id[host_index] #This is the id of the host tree

    #now grab the host
    host_tree = f['RockstarMergerTrees']['Tree_'+str(host_tree_id)]
    scale = host_tree['scale'][:]
    mmp = host_tree['mmp?'][:]
    Mvir_tree = host_tree['Mvir_all'][:]

    #now I want every unique snapshot in this merger tree
    scales_unique = np.unique(scale)
    #now at every scale factor I want the most massive halo in this stack

    MAH = []
    scales = []

    for snap_indicator in scales_unique:
        mask_at_snap = (scale==snap_indicator)
        masses_at_snap = Mvir_tree[mask_at_snap]
        MAH.append(np.max(masses_at_snap))
        scales.append(snap_indicator)

    data_matrix = np.zeros((len(MAH),2))
    
    data_matrix[:,0] = scales
    data_matrix[:,1] = MAH

    np.savetxt('./Mass_accretion_history/halo_'+str(halo_number)+'_MAH.txt',data_matrix,comments='# a Mvir (Msun/h)')
    
