import numpy as np
import numpy.linalg as la
from scipy import signal
from mpi4py import MPI
import time
import HERA_hack_FG
start = time.time()

##################### HERA SETUP ###############################

norm = False 

dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])

#observable corners of the sky [lat,long]
acorner = np.array([[120,270],[122,280],[120,280],[122,270]])

HERA = HERA_hack_FG.telescope(dishes, latitude=-30, channel_width=1., Tsys=300, beam_width=3, beam = 'gaussian')

obs = HERA_hack_FG.observation(HERA, 100, 150, 0.01,acorner,1, 0.2, norm = norm, pbeam = True)

#################################################################



comm = MPI.COMM_WORLD
myID = comm.rank
master = 0
numHelpers = comm.size - 1
npix = len(obs.observable_coordinates())
vec = (signal.unit_impulse(npix,0) + signal.unit_impulse(npix, 'mid') + signal.unit_impulse(npix, 1234) +signal.unit_impulse(npix,193)+signal.unit_impulse(npix, 687)+signal.unit_impulse(npix, 1432)+ signal.unit_impulse(npix, 122 )+signal.unit_impulse(npix, 13)+signal.unit_impulse(npix, 1344)+signal.unit_impulse(npix, 45))
pixel = np.arange(0,npix,1)
numThingsToDo = npix
numActiveHelpers = min(numHelpers,numThingsToDo)



if myID == master:
    sky_map = np.zeros(npix, dtype = complex)
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
        sky_map[tag] = temp

    

        if numSent < numThingsToDo:
            comm.send(numSent, dest=sender, tag=1)
            #print "I asked ", sender, "to square the ", numSent+1, " number out of ", len(v), "total" #this sends the consequent tasks
            numSent += 1
        else:
            print("Everything is done, so I asked ", sender, "to pack up")
            comm.send(0,dest=sender,tag=0)

    if norm  == True: #this is how you have to deal with the new mixing norm
    	obs.compute_normalization(None)
    	sky_map = np.dot(obs.norm, sky_map)
    	        
    	end = time.time()
    	print('runtime(s)', (end-start))
    	print(sky_map)
    else: 
    	end = time.time()
    	print('runtime(s)', (end-start))
    	print(sky_map)

elif myID <= numActiveHelpers:
    complete = False
    while (complete == False):
        status = MPI.Status()
        assignment = comm.recv(source=master, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == 0:
            complete = True
        else: #here is where you put the task you want each node to do! 
            pix = pixel[assignment]
            map_pix = obs.single_pix_convolve_map(pix,vec,None)
            comm.send(map_pix,dest=master, tag=assignment) #try doing the squareing before the send


comm.Barrier()