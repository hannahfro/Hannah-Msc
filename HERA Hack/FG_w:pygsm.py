import numpy as np 
import numpy.linalg as la
import numpy.random as ra
import astroquery
from astroquery.vizier import Vizier
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
		self.Npix = len(obs.observable_coordinates())
		obs.compute_beam()
		self.pbeam = obs.pbeam 
		self.observable_coordinates = obs.observable_coordinates()
		self.latitude = obs.latitude 
		self.times = obs.times 
		self.Nt = obs.Nt
		self.position  = obs.position
		self.beam_width = obs.beam_width


		#inheret observation properties like where in the sky/at what freq the observation is made. 


	def compute_synchro(self):
		nside = 1024
		df_gsm = pd.read_csv('pygsm_data.txt', sep=" ", header=None)
		df_gsm.columns = ["Temp (K)"]
		diffuse_synchrotron = df_gsm["Temp (K)"].to_numpy()

		wanted_pix = hp.pixelfunc.ang2pix(nside, self.observable_coordinates[:,0],self.observable_coordinates[:,1])

		print(len(wanted_pix))
		assert False
		
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

		self.free_free = np.asarray(pixel_flux_ff)

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
		##PROBLEM: the 30,1 is hardcoded but this depends on the number of pixels!!!
		theta_res = np.abs(np.cos(self.observable_coordinates[1,0])-np.cos(self.observable_coordinates[0,0]))
		phi_res = self.observable_coordinates[30,1]- self.observable_coordinates[1,1]
	
		omega_pix = float(theta_res*phi_res)

		factor = 1.4e-6*((self.freq/self.freq_fid)**(-2))*(omega_pix**(-1))

		pixel_flux = []

		for i in range(self.Npix):
			alpha = np.random.normal(alpha_0,sigma,n_sources)
			S_star = gen_fluxes(n_sources)
			sum_fluxes = 0 

			for i in range(n_sources-1):
				sum_fluxes += factor*S_star[i]*(self.freq/self.freq_fid)**(-alpha[i])
			
			pixel_flux.append(sum_fluxes/n_sources)

		self.sources = np.asarray(pixel_flux)

		return self.sources

	def bright_psource(self,nbins): #should also have preprocessed data  as input
	#### SCOOP THE DATA FROM VIZIER CATALOG #####
		Vizier.ROW_LIMIT = -1
		catalog_list = Vizier.find_catalogs('GLEAM')
		catalogs = Vizier.get_catalogs(catalog_list.keys())

		'''
		--------------------------------------------------------------------------------
		 FileName        Lrecl  Records   Explanations
		--------------------------------------------------------------------------------
		ReadMe              80        .   This file
		table1.dat          39       28   GLEAM first year observing parameters
		gleamegc.dat      3155   307455   GLEAM EGC catalog, version 2
		GLEAM_EGC_v2.fits 2880   137887   FITS version of the catalog
		--------------------------------------------------------------------------------
		'''

		#We will extract version 2 catalogue
		tt = catalogs['VIII/100/gleamegc'] 

		#List all the keys 
		#Details of the keys are available here: http://cdsarc.u-strasbg.fr/ftp/cats/VIII/100/ReadMe
		#And in more details here: https://heasarc.gsfc.nasa.gov/W3Browse/all/gleamegcat.html

		src_name = tt['GLEAM'] #Source ID
		RA       = tt['RAJ2000'] #RA
		DEC      = tt['DEJ2000'] #DEC
		flux     = tt['Fpwide'] #Peak flux in wide (170-231MHz) image

###########UNIT CONVERSIONS ################################


		#convert to equatorial coords (only DEC is changed)
		DEC_co_lat = 90-DEC 
		#convert to radians 
		RA_rads = RA*(np.pi/180.)
		DEC_rads = DEC_co_lat*(np.pi/180.)
		
	##########INITIAL CUT: Pick out only the sources bright enough to be seen at center of beam
		psource_flux = []
		psource_RA = []
		psource_DEC = []

		
		for i in range(len(flux)):
			if flux[i] >= 0.100: 
				psource_flux.append(flux[i])
				psource_RA.append(RA_rads[i])
				psource_DEC.append(DEC_rads[i])
				
		#convert to array for ease of use
		psource_flux = np.asarray(psource_flux)
		psource_RA = np.asarray(psource_RA)
		psource_DEC = np.asarray(psource_DEC)

	###### ORGANIZING THE DATA BY DIST TO CENTRE OF BEAM ############
		co_lat = np.pi / 2. - self.latitude

		#DEC distance to centre of the beam, may need to do actual distance...
		dist_from_centre = np.abs(psource_DEC-co_lat)

		data = np.stack([psource_flux,psource_RA,psource_DEC,dist_from_centre], axis = 1)

		psource_data = data[np.argsort(data[:, 3])] #sort by distance from centre of beam (becuase that way you do the brightest possiblesources first = less computing time)

		
	########## COMPUTE PBEAM #########################


		phis = (2. * np.pi * self.times) + self.position[0,1]

		primary = np.zeros((self.Nt, psource_data.shape[0]))

		for i in range(self.Nt): #compute the elements of pbeam
			primary[i] = np.exp(-((psource_data[:,1]-phis[i])**2 +(psource_data[:,2]-co_lat)**2) / float(self.beam_width**2))# 2D gaussian beam (N_position,2) 
			#this primary beam should now be in order of closest to furthest 

###########PICK OUT ALL THE BRIGHT BOISS ###################

		bin_index = np.int(len(primary[1])/nbins)

		psource_final = []
		for i in range(nbins):
			#find max pbeam of all time
			lower_bound = i*bin_index
			upper_bound = (i+1)*bin_index
			maxes = []
			for j in range(self.Nt): 
				maxes.append(max(primary[j,lower_bound:upper_bound]))
		  
			maxi = max(maxes) #This is now the max pbeam you use to check the fluxes
			
			for k in range(bin_index):
				if psource_data[((i+1)*k),0]* maxi >= .100: #find the bright guys
					psource_final.append(psource_data[((i+1)*k),:3]) #append bright guys to final list
				else:
					continue

		self.psource_final = np.asarray(psource_final)

		# convert flux to temperature brightness

		################### CONVERT TO 3D ##########################

		z_coord = np.zeros((self.psource_final.shape[0],1))

		self.psource_final = np.concatenate((self.psource_final, z_coord), axis = 1)

		for i in range(self.psource_final.shape[0]):                       
			self.psource_final[i,1] = np.sin(self.psource_final[i,1])*np.cos(self.psource_final[i,2])#X
			self.psource_final[i,2] = np.sin(self.psource_final[i,1])*np.sin(self.psource_final[i,2])#Y
			self.psource_final[i,3] = np.cos(self.psource_final[i,1])#Z     

		return self.psource_final

	def diffuse_fg(self,n_sources,nbins): #should also have data input

		self.bright_psource(nbins)

		extra_pixels = np.zeros(self.psource_final.shape[0])

		self.fg_map = self.compute_synchro() + self.compute_bremsstrauhlung() + self.unres_point_sources(n_sources) 
		self.fg_map =  np.concatenate((self.fg_map, extra_pixels), axis =0)

		#convert to temp brightness 

		return self.fg_map
		##turn into visibilities here before reintegrading i think! 



