#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import time
import HERA_hack_FG ##make sure it is in directory
import FG_pygsm
from scipy import signal

### also need to import the skies

comm = MPI.COMM_WORLD
myID = comm.rank
master = 0
numHelpers = comm.size - 1
freqs = np.arange(1420/(8+1),1420/(7.5+1),0.097) #central z = 7.75 # freqs 
numThingsToDo = len(freqs) 
numActiveHelpers = min(numHelpers,numThingsToDo)


#initialise HERA class
# dishes = np.array([[0,0],[150,75],[0,150],[0,-57.39295490174667],[30,0],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])
data1 = np.loadtxt('/Users/hannahfronenberg/Desktop/Grad School/HERA Noise/hera_positions_staged/antenna_positions_37.dat')
hera_bls_core = data1[:,:-1]

data2 = np.loadtxt('/Users/hannahfronenberg/Desktop/Grad School/HERA Noise/hera_positions_staged/excess_bls.dat')
hera_bls_outrigger = data2[::5,:-1]

hera_bls = np.vstack((hera_bls_core,hera_bls_outrigger))

npix_row,npix_col,npix_z = 50,50,len(freqs)

#import the naked sky 
HI = np.loadtxt("/Users/hannahfronenberg/Documents/GitHub/Hannah-Msc/SAZERAC/sky_sim_50/50_field_21cm_7.74.txt").reshape((npix_row*npix_col,len(freqs)))


pbeam = True 
norm = True 

# acorner = np.array([[120,280],[122,282],[120,282],[122,280]])

acorner = np.array([[119,274],[121,276],[119,276],[121,274]])

HERA = HERA_hack_FG.telescope(hera_bls, latitude=-30, channel_width=97800, beam_width=10, beam = 'gaussian')



if myID == master:
    obs_sky = np.zeros((npix_row*npix_col,npix_z)) ## obs array shape(npix_x,npix_y,npix_z) 
    numSent = 0
    print("Gonna send out the", numThingsToDo,"assignments now")

    for helperID in range(1,numActiveHelpers+1):
        #print "I asked ", helperID, "to square the  ", helperID, "number out of ", len(v), "total" #this gives each helper their initial task 
        comm.send(helperID-1, dest=helperID, tag=helperID)
        numSent += 1 #FROM HERE WE JUMP TO THE HELPERS THAT RECV

    for i in range(1,numThingsToDo+1):
        status = MPI.Status()
        temp = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) #this receives word form the helpes that they're done their first task
        sender = status.Get_source()
        tag = status.Get_tag()
        obs_sky[:,tag] = temp

        if numSent < numThingsToDo:
            comm.send(numSent, dest=sender, tag=1)
            numSent += 1
        else:
            comm.send(0,dest=sender,tag=0)


    obs_to_save = np.reshape(obs_sky,(npix_row*npix_col,len(freqs)))

    a_file = open("obs_sky_Hnoise_fg_774.txt", "w")
    print('made the file')
    for row in obs_to_save:
        np.savetxt(a_file, row)
    a_file.close() 

elif myID <= numActiveHelpers:
    complete = False
    while (complete == False):
        status = MPI.Status()
        assignment = comm.recv(source=master, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == 0:
            complete = True
        else:
            print('starting',assignment)
            obs_freq = freqs[assignment]
            obs = HERA_hack_FG.observation(telescope = HERA, n_days = 42, freq = obs_freq, delta_t = 0.01 ,corners = acorner, beam_sigma_cutoff=1, sky_shape = (npix_row,npix_col), norm = norm , pbeam = pbeam)
##FG stuff            
            obs.observable_coordinates()
            fg_21cm = FG_pygsm.foregrounds(obs,150) # initialise FG
            diffuse_fg = fg_21cm.diffuse_fg(100,True)#generate diffuse FG for sky
            sky = HI[:,assignment] + diffuse_fg # add cosmo signal

            obs.convolve_map(sky,None,None)
            obs.generate_map_noise(None,None)
            sky_map = np.real(obs.map + obs.noise/300)# observe the sky 
            comm.send(sky_map,dest=master, tag=assignment) #send map back and have the master append it somewhere


comm.Barrier()
