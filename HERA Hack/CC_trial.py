#CC_test 
import numpy as np
import numpy.linalg as la
import numpy.random as ra
from scipy import signal



from astroquery.vizier import Vizier

import HERA_hack_FG
import FG_pygsm


##################### SETUP ###########################

freq_fid = 150

n_dim_sources = 100

dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])

#observable corners of the sky [lat,long]
acorner = np.array([[120,270],[122,280],[120,280],[122,270]])

HERA = HERA_hack_FG.telescope(dishes, latitude=-30, channel_width=1., Tsys=300, beam_width=3, beam = 'gaussian')

obs = HERA_hack_FG.observation(HERA, 100, 100, 0.01,acorner,1, 0.2, norm = True, pbeam = False)

fg = FG_pygsm.foregrounds(obs,150)

###############
mid_pixel = np.int(fg.Npix*0.5098507)

imp =  100000*signal.unit_impulse(fg.Npix+1, mid_pixel) #True sky we wanna see
foregrounds = fg.diffuse_fg(n_dim_sources,False) #diffuse fg 
psource_data = fg.bright_psources(16)

sky_map =  np.real(obs.convolve_map(imp,psource_data))



# np.savetxt('map_with_fg_ps.txt',sky_map)