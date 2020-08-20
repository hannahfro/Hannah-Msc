import numpy as np
import matplotlib.pyplot as plt 
import numpy.random as ra
import statistics 
from scipy import signal
import numpy.linalg as la 
import HERA_hack

#make sure that HERA hack is in the directory that you're running this code in! 
#Remember to SCP it over

########### DEFINE OBS AND NECESSARY VARS #########


freq_fid = 150

dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])

#observable corners of the sky [lat,long]
acorner = np.array([[120,270],[122,280],[120,280],[122,270]])

HERA = HERA_hack.telescope(dishes, latitude=-30, channel_width=1., Tsys=300, beam_width=3, beam = 'gaussian')

obs = HERA_hack.observation(HERA, 100, 100, 0.01,acorner,1, 0.2, norm = False, pbeam = False)

def generate_foregrounds():
############ SYNCHRO EMISSION ############

	alpha_0_syn = 2.8
	sigma_syn = 0.1
	Asyn = 335.4 #K

	pixel_flux_syn = []

	alpha_syn = np.random.normal(alpha_0_syn,sigma_syn,obs.Npix)

	for i in range(obs.Npix):
	    flux = Asyn*(obs.freq/freq_fid)**(-alpha_syn[i])
	    pixel_flux_syn.append(flux)




	########### FREE FREE EMISSION ##########

	alpha_0_ff = 2.15
	sigma_ff = 0.01
	Aff = 33.5 #K

	pixel_flux_ff = []

	alpha_ff = np.random.normal(alpha_0_ff,sigma_ff,obs.Npix)

	for i in range(obs.Npix):
	    flux = Aff*(obs.freq/freq_fid)**(-alpha_ff[i])
	    pixel_flux_ff.append(flux)

	########### UNRES POINT SOURCE ###########

	gamma = 1.75

	def dnds(s):
	    return 4.*(s/880)**(-gamma)

	s = np.arange(8,100,1) #maybe make this an argument 
	n_sources = 10

	pdf = np.asarray([s,dnds(s)]) #0 is s, 1 is dnds
	prob = pdf[1]/float(sum(pdf[1]))
	cum_prob = np.cumsum(prob)

	def gen_fluxes(N):
	    R = ra.uniform(0, 1, N)
	    #Here we first find the bin interval that random number lies in min(cum_prob[])
	    #then we find the flux who's index is that cum_prob
	    #repat for all r in R
	    return [int(s[np.argwhere(cum_prob == min(cum_prob[(cum_prob - r) > 0]))]) for r in R]

	alpha_0 = 2.5
	sigma = 0.5

	theta_res = np.abs(np.cos(obs.observable_coordinates()[1,0])-np.cos(obs.observable_coordinates()[0,0]))
	phi_res = obs.observable_coordinates()[30,1]- obs.observable_coordinates()[1,1]
	omega_pix = theta_res*phi_res
	factor = 1.4e-6*((obs.freq/freq_fid)**(-2))*(omega_pix**(-1))

	pixel_flux = []

	for i in range(obs.Npix):
	    alpha = np.random.normal(alpha_0,sigma,n_sources)
	    S_star = gen_fluxes(n_sources)
	    sum_fluxes = 0 

	    for i in range(n_sources-1):
	        sum_fluxes += factor*S_star[i]*(obs.freq/freq_fid)**(-alpha[i])
	    
	    pixel_flux.append(sum_fluxes/n_sources)


	########## TOTAL FG ################

	pixel_flux = np.asarray(pixel_flux)
	pixel_flux_ff = np.asarray(pixel_flux_ff)
	pixel_flux_syn = np.asarray(pixel_flux_syn)

	total_fg = pixel_flux + pixel_flux_ff + pixel_flux_syn

	return total_fg



############# X_CORR ############


nreal = 200

# corr_NFG = np.zeros(nreal)

# corr_SN = np.zeros(nreal)

# corr_NN = np.zeros(nreal)

# cov_NFG = np.zeros(nreal)
# cov_SN = np.zeros(nreal)
# cov_NN = np.zeros(nreal)

cov_FF = np.zeros(nreal)
corr_FF = np.zeros(nreal)

for i in range(nreal):
    noise_1 = np.real(obs.generate_map_noise())
    # noise_2 = np.real(obs.generate_map_noise())
    fg_1= generate_foregrounds()
    f_1 = fg_1 - np.mean(fg_1)
    fg_2 = generate_foregrounds()
    f_2 = fg_2 - np.mean(fg_2)
    # n1 = noise_1 - np.mean(noise_1)
    # n2 = noise_2 - np.mean(noise_2)
    # nfg_1 = n1 + (fg_1 - np.mean(fg_1))
    # nfg_2 = n2 + (fg_2 - np.mean(fg_2))

    # norm_1 = np.sqrt(np.dot(n1,n1))
    # norm_2 = np.sqrt(np.dot(n2,n2))
    # norm_nfg_1 = np.sqrt(np.dot(nfg_1,nfg_1))
    # norm_nfg_2 = np.sqrt(np.dot(nfg_2,nfg_2))

    norm_f1 = np.sqrt(np.dot(f_1,f_1))
    norm_f2 = np.sqrt(np.dot(f_2,f_2))


    cov_FF[i] = np.dot(f_1,f_2)
    corr_FF[i] = cov_FF[i] /(norm_f1*norm_f2)
    
    # cov_NFG[i] = np.dot(nfg_1,nfg_2)
    # corr_NFG[i] = cov_NFG[i] /(norm_nfg_1*norm_nfg_2)

    # cov_NN[i] = np.dot(n1,n2)
    # corr_NN[i] = cov_NN[i] /(norm_1*norm_2)

    # cov_SN[i]= np.dot(n1,n1)
    # corr_SN[i] = cov_SN[i]/(norm_1*norm_1)
    
  
    ######## SUBTRACTING MEANS IN CORR ########
    # cov_SN = np.dot(noise_1,noise_1)
    # mean_prods_SN = (np.mean(noise_1))*(np.mean(noise_1))
    # corr_SN[i] = (cov_SN - (mean_prods_SN))/(norm_1*norm_1)
   
    

stacked_corrs = np.reshape((np.dstack((corr_FF,cov_FF))),(nreal,2))
# stacked_covs= np.reshape((np.dstack((cov_NFG,cov_NN,cov_SN))),(nreal,3))

np.savetxt('fg_fg.txt', stacked_corrs)
#np.savetxt('covs_sameO.txt', stacked_covs)
