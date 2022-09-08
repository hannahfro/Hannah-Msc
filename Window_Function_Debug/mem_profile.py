import numpy as np
import scipy.constants as sc
import HERA_hack_FG
import FG_pygsm 
from memory_profiler import profile



if __name__ == '__main__':

	dishes = np.array([[0,0],[150,75],[0,150],[0,-57.39295490174667],[30,0],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])
	data1 = np.loadtxt('/Users/hannahfronenberg/desktop/Grad School/HERA Noise/hera_positions_staged/antenna_positions_37.dat')
	hera_bls_core = data1[:,:-1]

	data2 = np.loadtxt('/Users/hannahfronenberg/desktop/Grad School/HERA Noise/hera_positions_staged/excess_bls.dat')
	hera_bls_outrigger = data2[::10,:-1]

	hera_bls = np.vstack((hera_bls_core,hera_bls_outrigger))


	npix_row,npix_col = 100,100


	pbeam = True 
	norm = True 


	acorner = np.array([[119,274],[121,276],[119,276],[121,274]])

	HERA = HERA_hack_FG.telescope(hera_bls, latitude=-30, channel_width=97800, beam_width=10, beam = 'gaussian')

	obs = HERA_hack_FG.observation(telescope = HERA, n_days = 3, freq = 182.54400000000044, delta_t = 0.01 ,corners = acorner, beam_sigma_cutoff=1, sky_shape = (npix_row,npix_col), norm = norm , pbeam = pbeam)

	obs.observable_coordinates()
	# obs.necessary_times()
	fg_21cm = FG_pygsm.foregrounds(obs,150)
	diffuse_fg = fg_21cm.diffuse_fg(100,True)



