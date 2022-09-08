#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import time
import CII_foregrounds
from CCAT_p import CCAT_p

### also need to import the skies

comm = MPI.COMM_WORLD
myID = comm.rank
master = 0
numHelpers = comm.size - 1

# freqs_HI = np.arange(1420/(6.5+1),1420/(6+1),0.097) #central z = 6.25  # freqs GHz
# freqs = np.linspace(1900/(6.5+1),1900/(6+1),len(freqs_HI))


# freqs_HI = np.arange(1420/(7+1),1420/(6.5+1),0.097) #central z = 6.75  # freqs GHz
# freqs = np.linspace(1900/(7+1),1900/(6.5+1),len(freqs_HI))

# freqs_HI = np.arange(1420/(7.5+1),1420/(7+1),0.097) #central z = 7.25  # freqs GHz
# freqs = np.linspace(1900/(7.5+1),1900/(7+1),len(freqs_HI))

freqs_HI = np.arange(1420/(8+1),1420/(7.5+1),0.097) #central z = 7.75  # freqs GHz
freqs = np.linspace(1900/(8+1),1900/(7.5+1),len(freqs_HI))

numThingsToDo = len(freqs) 
numActiveHelpers = min(numHelpers,numThingsToDo)

npix_row,npix_col = 50,50

#import the naked sky 
CII_field = np.loadtxt("/Users/hannahfronenberg/Documents/GitHub/Hannah-Msc/SAZERAC/sky_sim_50/50_field_CII_7.74.txt").reshape((npix_row,npix_col,len(freqs_HI)))


RA = np.linspace(0,0.0349066,npix_col) #rads (2 deg by 2 deg) 
DEC = np.linspace(0,0.0349066,npix_row) #rads (2 deg by 2 deg) 

delta_RA = (RA[1]-RA[0])
delta_DEC = (DEC[1]-DEC[0])
omega_pix = np.abs((RA[1]-RA[0])*(DEC[1]-DEC[0])) #in sr


if myID == master:
	obs_sky = np.zeros((npix_row,npix_col,len(freqs))) ## obs array shape(npix_x,npix_y,npix_z) 

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
		obs_sky[:,:,tag] = temp

		if numSent < numThingsToDo:
			comm.send(numSent, dest=sender, tag=1)
			numSent += 1
		else:
			comm.send(0,dest=sender,tag=0)


	obs_to_save = np.reshape(obs_sky,(npix_row*npix_col,len(freqs)))

	a_file = open("ccat_noise_fg_774.txt", "w")
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
			delta_channel = 0.0000975/2
			obs_freq = freqs[assignment]
			cii_fg = CII_foregrounds.CII_fg(obs_freq - delta_channel,obs_freq + delta_channel,omega_pix,2500)
			fgs = cii_fg.intensity()
			fg_tot = np.reshape(np.sum(cii_fg.I, axis =0) ,(50,50))
			sky = CII_field[:,:,assignment] + fg_tot
			noisy_convolve = CCAT_p(sky,RA,DEC,obs_freq,6,True, 0.86e6, 3*(60*60), 0.75) 
			comm.send(noisy_convolve,dest=master, tag=assignment) #send map back and have the master append it somewhere


comm.Barrier()
