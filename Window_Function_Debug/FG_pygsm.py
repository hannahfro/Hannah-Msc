import numpy as np 
import numpy.linalg as la
import numpy.random as ra
import pandas as pd
import healpy as hp
import astroquery
import scipy.constants as sc
from astroquery.vizier import Vizier
import HERA_hack_FG

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

		self.freq_fid = freq_fid #In MHz

		self.freq = obs.freq
		self.Npix = len(obs.observable_coordinates())
		self.observable_coordinates = obs.observable_coordinates()
		obs.necessary_times()
		self.latitude = obs.latitude 
		self.times = obs.times 
		self.Nt = obs.Nt
		self.position  = obs.position
		self.beam_width = obs.beam_width


		#inheret observation properties like where in the sky/at what freq the observation is made.

	def compute_synchro_pygsm(self):

		''' 
		This method computes galactic synchrotron emission by 
		findng the relevant pixels from the pygsm model
		based on the portion of observed sky, and the oberving frequency

		see Oliveira-Costa et. al., (2008) and Zheng et. al., (2016)
		'''

		## MAKE THIS READ FROM pygdsm.bin

		nside = 1024
		#Reads in the file for the frequecny in question
		diffuse_synchrotron = np.fromfile('pygdsm_%sMHz.bin'%self.freq, dtype=np.float32)

		obs_index = hp.pixelfunc.ang2pix(nside, self.observable_coordinates[:,0],self.observable_coordinates[:,1])

		self.gal_emission = []

		for i in range(len(obs_index)):
			self.gal_emission.append(diffuse_synchrotron[obs_index[i]])

		self.gal_emission = self.gal_emission - np.mean(self.gal_emission)

		return self.gal_emission## THIS IS IN KELVIN

	def compute_synchro(self):

		'''
		This method computes a statistically accurate 
		galactric synchrotron model published in Liu et. al., (2011)
		'''

		
		alpha_0_syn = 2.8
		sigma_syn = 0.1
		Asyn = 335.4 #K

		pixel_flux_syn = []

		alpha_syn = np.random.normal(alpha_0_syn,sigma_syn,self.Npix) 

		for i in range(self.Npix):
			flux = Asyn*(self.freq/self.freq_fid)**(-alpha_syn[i])
			pixel_flux_syn.append(flux)

		self.gal_emission = np.asarray(pixel_flux_syn)

		return self.gal_emission # in Kelvin


	def compute_bremsstrauhlung(self):

		'''
		This method computes a map of diffuse free-free 
		emission from a model published in Liu et. al., (2011)
		'''

		alpha_0_ff = 2.15
		sigma_ff = 0.01
		Aff = 33.5 #K

		pixel_flux_ff = []

		alpha_ff = np.random.normal(alpha_0_ff,sigma_ff,self.Npix)

		for i in range(self.Npix):
			flux = Aff*(self.freq/self.freq_fid)**(-alpha_ff[i])
			pixel_flux_ff.append(flux)

		self.free_free = np.asarray(pixel_flux_ff)- np.mean( np.asarray(pixel_flux_ff))

		return self.free_free # THIS IS IN KELVIN 

	def compute_omega(self): #this is for temp brightness conversion 

		phi = self.observable_coordinates[:,1]

		min_indices = np.where(phi == min(phi))

		upper_index = max(min_indices[0])+1

		theta_res = np.abs(np.cos(self.observable_coordinates[1,0])-np.cos(self.observable_coordinates[0,0]))
		phi_res = self.observable_coordinates[upper_index,1]- self.observable_coordinates[1,1]
	
		self.omega_pix = float(theta_res*phi_res) 


	def compute_unres_point_sources(self,n_sources): 




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
	
		self.compute_omega()

		factor = 1.4e-6*((self.freq/self.freq_fid)**(-2))*(self.omega_pix**(-1)) #[mjy^-1 sr^-1 K]

		pixel_flux = []

		for i in range(self.Npix):
			alpha = np.random.normal(alpha_0,sigma,n_sources) #pick alpha for each source in pix
			S_star = gen_fluxes(n_sources) #get fluxes for each source in pix
			sum_fluxes = 0 

			for j in range(n_sources-1):
				sum_fluxes += factor*S_star[j]*(self.freq/self.freq_fid)**(-alpha[j])
			
			pixel_flux.append(sum_fluxes/n_sources)

		self.sources = np.asarray(pixel_flux) - np.mean(np.asarray(pixel_flux))

		return self.sources # THIS IS IN KELVIN

	def bright_psources(self): #should also be preprocessed 
	#### SCOOP THE DATA FROM VIZIER CATALOG #####

		''' 
		This method selects, from the GLEAM catalogue, sources 
		brighter than 100 mJy

		'''


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
		psource_phi = []
		psource_theta = []

		
		for i in range(len(flux)):
			if flux[i] >= 0.100: 
				psource_flux.append(flux[i])
				psource_phi.append(RA_rads[i])
				psource_theta.append(DEC_rads[i])
				
		#convert to array for ease of use
		psource_flux = np.asarray(psource_flux)
		psource_phi = np.asarray(psource_phi)
		psource_theta = np.asarray(psource_theta)

	###### ORGANIZING THE DATA BY DIST TO CENTRE OF BEAM ############ I think that this is unecessary 
		co_lat = np.pi / 2. - self.latitude

		#DEC distance to centre of the beam, may need to do actual distance...
		dist_from_centre = np.abs(psource_theta-co_lat)


		data = np.stack([psource_flux,psource_phi,psource_theta,dist_from_centre], axis = 1) # check axis stack

		psource_data = data[np.argsort(data[:, 3])] #sort by distance from centre of beam (becuase that way you do the brightest possiblesources first = less computing time)
	

	########## COMPUTE PBEAM #########################


		phis = (2. * np.pi * self.times) + self.position[0,1]

		primary = np.zeros((self.Nt, psource_data.shape[0])) #Nt * n_sources

		for i in range(self.Nt): #compute the elements of pbeam
			primary[i] = np.exp(-((psource_data[:,1]-phis[i])**2 +(psource_data[:,2]-co_lat)**2)/float(self.beam_width**2), dtype = "float64")# 2D gaussian beam (N_position,2) 
		
		#this primary beam should now be in order of closest to furthest


########### PICK OUT ALL THE BRIGHT BOISS ###################
# ## REDO THIS SECTION 

# 		bin_index = np.int(len(primary[1])/nbins)

# 		psource_final = []
# 		psource_pbeam = []

# 		for i in range(nbins):
# 			#find max pbeam of all time
# 			lower_bound = i*bin_index
# 			upper_bound = (i+1)*bin_index
# 			maxes = []
# 			for j in range(self.Nt): 
# 				maxes.append(max(primary[j,lower_bound:upper_bound]))
		  
# 			maxi = max(maxes) #This is now the max pbeam you use to check the fluxes
			
# 			for k in range(bin_index):
# 				if psource_data[((i+1)*k),0]* maxi >= 0.100: #find the bright guys
# 					psource_final.append(psource_data[((i+1)*k),:3]) #append bright guys' to final list
# 					psource_pbeam.append(primary[:,((i+1)*k)]) # also make a list with that guys' primary beams 

# 				else:
# 					continue

		psource_final = []
		psource_pbeam = []

		#this will replace the binned selection technique
		#here we go through each source and check if it is bright enough to be seen even for 1 time, if so, append to list
		for i in range(psource_data.shape[0]):
			max_primary = max(primary[:,i]) #find the max primary value of all time for that source
			if psource_data[i,0]*max_primary >= 0.100: #if max primary doesn't dampen flux below threshold
				psource_final.append(psource_data[i,:3]) #append bright guys' to final list
				psource_pbeam.append(primary[:,i]) # also make a list with that guys' primary beams 

			else:
				continue


		self.psource_final = np.asarray(psource_final)
		self.psource_pbeam = np.asarray(psource_pbeam)
		self.psource_pbeam = self.psource_pbeam.T

		#write pbeam to file
		self.psource_pbeam = self.psource_pbeam.astype(np.float32)
		self.psource_pbeam.tofile('psource_beam.bin')

		self.compute_omega()

		wavelength_fid = sc.c/self.freq #meters

		#temp_conv = (sc.c**2)/(2*sc.k*((self.freq_fid*1e6)**2))

		temp_conv = (wavelength_fid**2)/(2*sc.k*self.omega_pix) #off by factor of 10^26 haha

		self.psource_final[:,0] = (self.psource_final[:,0]*(10**(-26)))*temp_conv #convert jy to watt and then convert to K 
		''' 
							DATA STRUCTUE
		--------------------------------------------------------
		flux(K)   phi(rads)   theta(rads)   dist_from_cent(rads)
		--------------------------------------------------------
		'''

		################### CONVERT TO 3D ##########################

		z_coord = np.zeros((self.psource_final.shape[0],1))

		print(self.psource_final.shape)

		self.psource_final = np.concatenate((self.psource_final, z_coord), axis = 1)

		print(self.psource_final.shape)

		for i in range(self.psource_final.shape[0]):                       
			self.psource_final[i,1] = np.sin(self.psource_final[i,1])*np.cos(self.psource_final[i,2])#X
			self.psource_final[i,2] = np.sin(self.psource_final[i,1])*np.sin(self.psource_final[i,2])#Y
			self.psource_final[i,3] = np.cos(self.psource_final[i,1])#Z   

		''' 
							DATA STRUCTUE
		--------------------------------------------------------
		flux(K)   phi(rads)   theta(rads)   z(rads)  
		--------------------------------------------------------
		'''  

		self.psource_final= self.psource_final.astype(np.float32)
		self.psource_final.tofile('psource_data.bin')#IN KELVIN

	def diffuse_fg(self,n_sources,pygsm): #should also have data input

		if pygsm == True: 

			self.fg_map = self.compute_synchro_pygsm() + self.compute_bremsstrauhlung() + self.compute_unres_point_sources(n_sources) 

			return self.fg_map

		elif pygsm == False: 


			self.fg_map = self.compute_synchro() + self.compute_bremsstrauhlung() + self.compute_unres_point_sources(n_sources) 

			return self.fg_map

		##turn into visibilities here before reintegrading i think! 



