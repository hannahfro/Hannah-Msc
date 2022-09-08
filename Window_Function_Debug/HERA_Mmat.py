#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import time
import HERA_hack_FG ##make sure it is in directory
start = time.time()

### also need to import the skies

comm = MPI.COMM_WORLD
myID = comm.rank
master = 0
numHelpers = comm.size - 1
freqs = np.arange(1,10,1) # freqs 
numThingsToDo = len(freqs) 
numActiveHelpers = min(numHelpers,numThingsToDo)

#import the naked sky 
#initialise HERA class

dishes = np.array([[0,0],[150,75],[0,150],[0,-57.39295490174667],[30,0],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])
bls = np.loadtxt('/Users/hannahfronenberg/desktop/MSC1/HERA Noise/hera_positions_staged/antenna_positions_350.dat')
hera_bls = bls[:,:-1]

npix_row,npix_col,npix_z = 100,100,len(freqs)


pbeam = True 
norm = True 

# acorner = np.array([[120,280],[122,282],[120,282],[122,280]])

acorner = np.array([[119,274],[121,276],[119,276],[121,274]])

HERA = HERA_hack_FG.telescope(dishes, latitude=-30, channel_width=97800, beam_width=10, beam = 'gaussian')



if myID == master:
    M_matrix= np.zeros(npix_row,npix_col,npix_z) ## array to hold the M matrix in it

    numSent = 0
    print("Gonna send out the assignments now")
    for helperID in range(1,numActiveHelpers+1):
        #print "I asked ", helperID, "to square the  ", helperID, "number out of ", len(v), "total" #this gives each helper their initial task 
        comm.send(helperID-1, dest=helperID, tag=helperID)
        numSent += 1 #FROM HERE WE JUMP TO THE HELPERS THAT RECV

    for i in range(1,numThingsToDo+1):
        status = MPI.Status()
        temp = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) #this receives word form the helpes that they're done their first task
        sender = status.Get_source()
        tag = status.Get_tag()
        obs_sky[:,:,tag] = temp

        if numSent < numThingsToDo:
            comm.send(numSent, dest=sender, tag=1)
            numSent += 1
        else:
            comm.send(0,dest=sender,tag=0)


    end = time.time()
    print 'runtime(s)', (end-start)
    print  "dot product is", np.sum(squares) ## here write the obs to binary

elif myID <= numActiveHelpers:
    complete = False
    while (complete == False):
        status = MPI.Status()
        assignment = comm.recv(source=master, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == 0:
            complete = True
        else:
            obs_freq = freqs[assignment]
            obs = HERA_hack_FG.observation(telescope = HERA, n_days = 42, freq = obs_freq, delta_t = 0.01 ,corners = acorner, beam_sigma_cutoff=1, sky_shape = (npix_row,npix_col), norm = norm , pbeam = pbeam)
            obs.compute_M(None,None)
            M = self.Mmat
            comm.send(M,dest=master, tag=assignment) #send map back and have the master append it somewhere


comm.Barrier()
