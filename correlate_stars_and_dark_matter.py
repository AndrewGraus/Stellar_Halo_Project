#!/usr/bin/env python3

## job name
#PBS -N correlation
## queue: devel <= 2 hr, normal <= 8 hr, long <= 5 day
#PBS -q normal
#PBS -l select=1:ncpus=20:mpiprocs=1:ompthreads=1:model=ivy
#PBS -l walltime=8:00:00
## combine stderr & stdout into one file
#PBS -j oe
## output file name
#PBS -o correlation_job.txt
#PBS -M agraus@utexas.edu
#PBS -m bae
#PBS -V
#PBS -W group_list=s1542

#The purpose of this program is to take the stellar halo isolation test
#And apply it to the dark matter particles, giving a stellar mass to 
#halo mass ratio for the dark matter
#
# Step 1: Isolate the stellar halo using the info from the ipynb
# Step 2: Take the star particles position data and correlate that
#         With the nearest DM particle
#
# That second step is an interesting one and I might investigate a 
# few different ways to do that based how to pick the "closest"
# particle (maybe closest in phase space rather than just 
# physical space) and what algorithm to use
#
# Note: I have no idea how to integrate the actual star masses in this
# I had the original idea to just save the number of stars per halo
# but the mass per star particle is variable

#Note for July 7th 2020:
# beginning modification to add circularity cut to the program
# what do I need to do?
# 1) load j data
# 2) make an array fills in particles that don't have a j_value
# 3) use that data to make the cut and then correlate

import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
#step one: Isolate the stellar halo

h = 0.702

#First load up the stars
#NOTE I forgot to convert the particle parameters back to simulation units when saving
#the j file so coordinates are in kpc NOT kpc/h and masses are already in M_sun
f_part_star = h5py.File('../m12i_res_7100_cdm/output/snapshot_600.stars_with_j.hdf5')

#assign masses and coordinates

star_pos = f_part_star['PartType4']['Coordinates'][:]
star_mass = f_part_star['PartType4']['Masses'][:]
star_ids = f_part_star['PartType4']['ParticleIDs'][:]
star_epsilon = f_part_star['PartType4']['circularity'][:]

#load up the halo data
f_halo = h5py.File('../m12i_res_7100_cdm/halo/halo_600.hdf5')
pos_halo = f_halo['position'][:]
mass_halo = f_halo['mass'][:]
radius_halo = f_halo['radius'][:]
vel_halo = f_halo['velocity'][:]

#identify the host
host_id = np.argmax(mass_halo)
host_mass = mass_halo[host_id]
host_pos = pos_halo[host_id]
host_vel = vel_halo[host_id]

dist = np.linalg.norm(star_pos-host_pos,axis=1) #distance of all the stars from the host center
dist_halo = np.linalg.norm(pos_halo-host_pos,axis=1) #distance of all halos from the host center

print(np.sum(dist<50.0))
print(np.min(dist))

sat_mask = (dist_halo<300.0)&(dist_halo>0.0)&(mass_halo>1.0e7) #select only halos in "Rvir" and only sats
#                                                               #above a certain mass

#sat_mask = (dist_halo<300.0)&(dist_halo>0.0)&(mass_halo>1.0e10) #select only halos in "Rvir" and only sats

sat_pos = pos_halo[sat_mask]
sat_rad = radius_halo[sat_mask]

dwarf_dist_mask_tot = np.ones_like(dist)*True #initalize a mask for finding all stars in any satellite

#loop over all sats and mask out the stars inside a satellite
print("looping over satellites")
for jj in range(len(sat_rad)):
    dwarf_pos = sat_pos[jj]
    dwarf_rad = sat_rad[jj]
    
    dwarf_dist = np.linalg.norm(star_pos-dwarf_pos,axis=1)
    #False if it's IN the satellite
    dwarf_dist_mask = (dwarf_dist>dwarf_rad)
    
    dwarf_dist_mask_tot = dwarf_dist_mask_tot*dwarf_dist_mask

#convert this mask to booleans
dwarf_dist_mask_tot_bool = list(map(bool,dwarf_dist_mask_tot)) #in python3 this returns a map and NOT an array

stars_not_in_sats = star_pos[dwarf_dist_mask_tot_bool&(dist<300.0)&(dist>5.0)&(star_epsilon<0.5)]  
mass_of_stars_not_in_sats = star_mass[dwarf_dist_mask_tot_bool&(dist<300.0)&(dist>5.0)&(star_epsilon<0.5)]

#We can probably be alot more clever with this mask and select out stars that are in the actual disk
#by masking out stuff with significant rotation instead of anything within 10 kpc of the center
#and also to mask out things that are actually in the gravitational well of the satelltes instead of
#just in the virial radius of a satellite but we're gonna do this quick and dirty at first

#so now I have the positions of all the stars that are in the "halo" as I've defined it.

#now I need to load up the dark matter data and figure out a way to correlate every star particle to its
#corresponding dark matter particle.

#my intial thinking for doing this would be calculate the distance between every star particle and every
#dark matter particle (this would create an ENORMOUS matrix)

#I think I can actually use sklearn for this. The idea is to use
#NearestNeightbors run it on the dark matter particles (the training set)
#and then use the star particles as the data to get eatch star's Neighbor
#it also returns indicies I can then use those indicies along with a counting
#algorithim to see now many times each DM particles appears in the 
#nearest neighbors count then assign mass to the DM particle based on that
#algorithm

snap_num = 600

print("loading up DM data")
#gotta merge the snapshots for the folders with multiple files per snapshot                           
part_mass = np.empty((0))
part_pos = np.empty((0,3))
part_vel = np.empty((0,3))
part_ids = np.empty((0))

f_part = h5py.File('../m12i_res_7100_cdm/output/snapshot_'+str(snap_num).zfill(3)+'.0.hdf5')
scale = f_part['Header'].attrs['Time']

for block in range(4):
    f_part = h5py.File('../m12i_res_7100_cdm/output/snapshot_'+str(snap_num).zfill(3)+'.'+str(block)+'.hdf5')
    mass_block = f_part['PartType1']['Masses'][:]*10**10.0/h
    pos_block = f_part['PartType1']['Coordinates'][:]*scale/h
    vel_block = f_part['PartType1']['Velocities'][:]*np.sqrt(scale)
    ids_block = f_part['PartType1']['ParticleIDs'][:]

    part_mass = np.append(part_mass,mass_block)
    part_pos = np.append(part_pos,pos_block,axis=0)
    part_vel = np.append(part_vel,vel_block,axis=0)
    part_ids = np.append(part_ids,ids_block)

#"learn" the dm positions (relative to host)
print("learning dm positions")

########Nearest Neighbors Block#########
#Update May 7th, for some reason NN gives weird results, assigns things in 
#random chuncks throughout the simulation volume, I'm not sure what the issue is
#
#nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(part_pos-host_pos)
#now match all the stars in the halo
#NN_dist, NN_indicies = nbrs.kneighbors(stars_not_in_sats-host_pos)
#NN_indicies is the indexes of the dark matter particles
#I should be able to plug this into the particle IDs
#and get the ids of the DM particles each star
#is associated with
#NN_indicies = np.ndarray.flatten(NN_indicies)

#########KD tree block#######
#scipy's kd_tree seems to do a similar thing and works a similar way
#where you construct a tree with the data, and then query it to
#find the kth nearest neighbors
#
#returns d, i = tree.query(X, k = 1)
#d is the distance
#i is the index of the nearest neighbor in X

#construct the tree from the DM particles
tree = cKDTree(part_pos-host_pos)

NN_dist, NN_indicies = tree.query(stars_not_in_sats-host_pos, k = 1)

#I think the sorting is the issue
print("doing a bunch of array stuff")
ids_with_stars = part_ids[NN_indicies]

star_ref_mask = np.in1d(part_ids,ids_with_stars)
ids_without_stars = part_ids[~star_ref_mask]

###NEW CODE AS OF MARCH 30th HERE
#
#

#What I have no is the list of every star's closest dark matter particle
#I need to then grab all the duplicates and sum the masses for each duplicate
#id

#so search ids_with_stars for duplicates, and then sum mass_of_stars_not_in_sats
#based on those duplicates

#lets start with a loop because it's easy

parts_w_stars_mass = []

for select_id in np.unique(ids_with_stars):
    a_mask = (ids_with_stars==select_id)
    parts_w_stars_mass.append(np.sum(mass_of_stars_not_in_sats[a_mask]))

parts_w_stars_mass = np.array(parts_w_stars_mass)
parts_wo_stars_mass = np.zeros_like(ids_without_stars)

ids_with_stars_mask = np.in1d(part_ids,ids_with_stars)
ids_without_stars = part_ids[~ids_with_stars_mask]

pos_with_stars = part_pos[ids_with_stars_mask]
pos_without_stars = part_pos[~ids_with_stars_mask]

vel_with_stars = part_vel[ids_with_stars_mask]
vel_without_stars = part_vel[~ids_with_stars_mask]

mass_with_stars = part_mass[ids_with_stars_mask]
mass_without_stars = part_mass[~ids_with_stars_mask]

total_ids = np.append(ids_with_stars,ids_without_stars)
total_pos = np.append(pos_with_stars,pos_without_stars,axis=0)
total_vel = np.append(vel_with_stars,vel_without_stars,axis=0)
total_mass = np.append(mass_with_stars,mass_without_stars)
total_M_star = np.append(parts_w_stars_mass,parts_wo_stars_mass)

#Save it in a similar format to the particle data

f_write = h5py.File('DM_data_w_stars_training.hdf5','w')

f_write.create_dataset("PartType1/Coordinates",data=total_pos)
f_write.create_dataset("PartType1/Velocities",data=total_vel)
f_write.create_dataset("PartType1/ParticleIDs",data=total_ids)
f_write.create_dataset("PartType1/Masses",data=total_mass)
f_write.create_dataset("PartType1/Stellar_Masses",data=total_M_star)
f_write.close()


