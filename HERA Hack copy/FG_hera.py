import numpy as np 
import numpy.linalg as la
import HERA_hack

####CODE FOR ADDING FOREGROUNDS####

#1. Galactic Synchrotron Portion
#2. Free-Free emission 
#3. Point sources 
#4. Unresolved point sources

#pseudocode for including foregrounds in the sky vector 


"""	The contiribution of the foreground will depend on the observation you want to make
(in the sky you are looking,at what freq, etc...) when in the universe's history, 

 """

class foregrounds(object):

	def __init__(self, obs, freq_fid):

		self.freq_fid = freq_fid

		self.freq = obs.freq
		self.Npix = obs.Npix

		#inherit observation properties like where in the sky/at what freq the observation is made. 


	def compute_synchro(self):

		
		alpha_0_syn = 2.8
		sigma_syn = 0.1
		Asyn = 335.4 #K

		pixel_flux_syn = []

		alpha_syn = np.random.normal(alpha_0_syn,sigma_syn,self.Npix)

		for i in range(self.Npix):
		    flux = Asyn*(self.freq/self.freq_fid)**(-alpha_syn[i])
		    pixel_flux_syn.append(flux)

		self.gal_emission = pixel_flux_syn

		return self.gal_emission

	def compute_bremsstrauhlung(self):

		""" compute thermal bremsstrauhlung at for given obs"""

		alpha_0_ff = 2.15
		sigma_ff = 0.01
		Aff = 33.5 #K

		pixel_flux_ff = []

		alpha_ff = np.random.normal(alpha_0_ff,sigma_ff,self.Npix)

		for i in range(self.Npix):
		    flux = Aff*(self.freq/self.freq_fid)**(-alpha_ff[i])
		    pixel_flux_ff.append(flux)

		self.free_free = pixel_flux_ff

		return self.free_free

	def unres_point_sources(self,n_sources): 

		""" Extrac relevant point sources in observed region. Consider including edge point
		sources that would leak into the field of view.   """
		#differential source count 
		gamma = 1.75

		def dnds(s):
		    return 4.*(s/880)**(-gamma)

		s = np.arange(8,100,1) #maybe make this an argument 

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
		factor = 1.4e-6*((self.freq/self.freq_fid)**(-2))*(omega_pix**(-1))

		pixel_flux = []

		for i in range(self.Npix):
		    alpha = np.random.normal(alpha_0,sigma,n_sources)
		    S_star = gen_fluxes(n_sources)
		    sum_fluxes = 0 

		    for i in range(n_sources-1):
		        sum_fluxes += factor*S_star[i]*(self.freq/self.freq_fid)**(-alpha[i])
		    
		    pixel_flux.append(sum_fluxes/n_sources)

		self.sources = pixel_flux

		return self.sources


	def foreground_map(self,n_sources):

		# if self.position is not None:
  #           pass
  #       else:
  #           self.position = self.observable_coordinates() 

		self.fg_map = self.gal_emission + self.free_free + self.sources

		##turn into visibilities here before reintegrading i think! 

