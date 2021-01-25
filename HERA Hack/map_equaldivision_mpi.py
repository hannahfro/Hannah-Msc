
import numpy as np
import numpy.linalg as la
from scipy import signal
from mpi4py import MPI
import time
import HERA_hack_FG
start = time.time()

##################### HERA SETUP ###############################

norm = True 
dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])

#observable corners of the sky [lat,long]
acorner = np.array([[120,270],[122,280],[120,280],[122,270]])

HERA = HERA_hack_FG.telescope(dishes, latitude=-30, channel_width=1., Tsys=300, beam_width=3, beam = 'gaussian')

obs = HERA_hack_FG.observation(HERA, 100, 150, 0.01,acorner,1, 0.2, norm = norm, pbeam = True)

#################################################################

import time
start = time.time()

comm = MPI.COMM_WORLD
myID = comm.rank
master = 0
numHelpers = comm.size - 1
npix = len(obs.observable_coordinates())
vec = (signal.unit_impulse(npix,0) + signal.unit_impulse(npix, 'mid') + signal.unit_impulse(npix, 1234) +signal.unit_impulse(npix,193)+signal.unit_impulse(npix, 687)+signal.unit_impulse(npix, 1432)+ signal.unit_impulse(npix, 122 )+signal.unit_impulse(npix, 13)+signal.unit_impulse(npix, 1344)+signal.unit_impulse(npix, 45))
pixel = np.arange(0,npix,1)
elements_per_helper = npix//numHelpers 
remainder = npix%numHelpers



if myID == master:
	#print "I need to squrare",len(v),"numbers"
	#print "each helper will square", elements_per_helper
	#print "the remainder is", remainder
	sky_map = np.zeros(npix, dtype = complex)

	for i in range(1,npix+1):
			status = MPI.Status()
			temp = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
			sender = status.Get_source()
			tag = status.Get_tag()
			sky_map[tag] = temp

	end = time.time()
	print("runtime =", end-start)
	print(sky_map)


for helperID in range(1,comm.size):
	if myID == helperID:
		#print "I am ID", helperID, "here to square the numbers",(helperID-1)*elements_per_helper, "to", ((helperID-1)*elements_per_helper)+(elements_per_helper-1)
		for i in range((helperID-1)*elements_per_helper,((helperID-1)*elements_per_helper)+elements_per_helper): 
			map_pix = obs.single_pix_convolve_map(i,vec,None)
			#status = MPI.Status()
			comm.send(map_pix,dest=master, tag=i) 
			

if remainder != 0 :
	if myID == 1:
		print("I am ID 1 here to take care of remainders")
		for i in range(npix-remainder,npix):
			status = MPI.Status()
			map_pix = obs.single_pix_convolve_map(i,vec,None)
			tag = i
			comm.send(map_pix,dest=master, tag=i)

comm.Barrier()
