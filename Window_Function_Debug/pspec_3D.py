import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from astropy.cosmology import WMAP9 as cosmo


class Power_Spectrum(object):
	"""docstring for Power_Spectrum

	This code computes the 2D and 1D power spectra from a 3D cube of data. These 3D cubes are assumed to be
	 within a small enough frequency range that they can be treated as coeval. We use the middle slice as the one which sets the 
	 comoving scales for the cube. 

"""
	def __init__(self, data, theta_x, theta_y , freqs, rest_freq, nbins,nbins_perp):
		
		self.data = data #- np.mean(data) #data cube in angles and frequency coordinates
		self.theta_x = theta_x# angular scale in the x direction RADS
		self.theta_y = theta_y #angular scale in the y direction RADS
		self.freqs = freqs*(1e6) #array of all the frequencies that were observed MHz --> Hz

		self.rest_freq = rest_freq*(1e6) #input in MHz convert to Hz
		self.nbins = nbins # number of bins, 
		self.nbins_perp = nbins_perp # number of bins, 


		self.y_npix = self.data.shape[0]
		self.x_npix = self.data.shape[1]
		self.freq_npix = self.data.shape[2]

		## get all the z info from the mid req 
		if self.freq_npix % 2 == 0 :
			self.mid_freq = self.freqs[(self.freq_npix//2)]
		else: 
			self.mid_freq = self.freqs[(self.freq_npix//2 + 1)]
	   
		self.z = (self.rest_freq/self.mid_freq) - 1
		#these two lines give you the physical dimensions of a pixel (inverse of sampling ratealong each axis)
		self.delta_thetay = (self.theta_y/self.data.shape[0]) # size of y pixel
		self.delta_thetax = (self.theta_x/self.data.shape[1]) # size of x pixel
		self.delta_freq = (max(self.freqs) - min(self.freqs))/self.data.shape[2]
		


	def cosmo_FFT3(self): 
		''' computes the fourier transform of a 2D field of mean 0'''

		self.volume_element = (self.delta_thetax*self.delta_thetay*self.delta_freq)*(((sc.c/1000) *((1+self.z)**2) * ((cosmo.comoving_distance(self.z).value)**2))/(cosmo.H0.value * cosmo.efunc(self.z) * self.rest_freq))
		self.fft_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.data*self.volume_element)))# [mK rads^2 Hz] This is "observer" fourier transform
		self.ps_data = (np.conj(self.fft_data))*self.fft_data # [mK rads^2 Hz]^2
		# self.ps_data *= (((2.99792e5) *((1+self.z)**2) * ((cosmo.comoving_distance(self.z).value)**2))/(cosmo.H0.value * cosmo.efunc(self.z) * self.rest_freq))**2 #[mk^2 Mpc^6] ## convert pspec data ###
	

	def compute_eta_nu(self):

		self.u_x = np.fft.fftshift(np.fft.fftfreq(self.x_npix, d = self.delta_thetax)) 
		self.u_y = np.fft.fftshift(np.fft.fftfreq(self.y_npix, d = self.delta_thetay))
		self.eta = np.fft.fftshift(np.fft.fftfreq(self.freq_npix, d = self.delta_freq))



		self.delta_ux = self.u_x[1]-self.u_x[0]
		self.delta_uy = self.u_y[1]-self.u_y[0]
		self.delta_eta = self.eta[1]-self.eta[0]

		U = []

		for i in range(len(self.u_y)): 
			for j in range(len(self.u_x)):
				U.append(np.sqrt(self.u_x[j]**2 + self.u_y[i]**2))

		self.U = np.asarray(U)


	def compute_Ubox(self):

		self.compute_eta_nu()

		self.U_box = np.reshape(self.U,(len(self.u_y),len(self.u_x)))

	def compute_kperp_kpar(self):

		self.compute_Ubox()

## compute k_par, k_perps
		self.k_par = self.eta * ((2*np.pi*self.rest_freq*cosmo.H0.value *1000* cosmo.efunc(self.z))/(sc.c*((1+self.z)**2)))
		# print(self.k_par)
		self.kx = self.u_x * ((2*np.pi)/cosmo.comoving_distance(self.z).value)
		self.ky = self.u_y * ((2*np.pi)/cosmo.comoving_distance(self.z).value)
		self.k_perp_mag = self.U * ((2*np.pi)/cosmo.comoving_distance(self.z).value)
		self.delta_kx = self.delta_ux * ((2*np.pi)/cosmo.comoving_distance(self.z).value)# the delta's are also now z-dependent
		self.delta_ky = self.delta_uy *((2*np.pi)/cosmo.comoving_distance(self.z).value)
		self.delta_kz = self.delta_eta * ((2*np.pi*self.rest_freq*cosmo.H0.value *1000* cosmo.efunc(self.z))/(sc.c*((1+self.z)**2)))

	def compute_volume(self):
	# convert find the approx volume of that chunk
		rx = cosmo.comoving_distance(self.z).value*self.theta_x
		ry = cosmo.comoving_distance(self.z).value*self.theta_y
		rz = ((sc.c*((1+self.z)**2))/(cosmo.H0.value*1000*self.rest_freq*cosmo.efunc(self.z)))*(max(self.freqs)-min(self.freqs))

		self.volume3D = rx*ry*rz

			
	def compute_2D_pspec(self):

		
		self.cosmo_FFT3() 
		self.compute_kperp_kpar()
		self.compute_volume()

		# so what we want to do here is bin each frequency chunk into a 1D vector and then ouput kperp_binned vs. k_par
		# This is a little bit tricky because this binning has to be done at every slice.
		# I think that this can be thought of as making a usual 2D power spectrum but just a bunch of times over. 


		self.pspec_2D = np.zeros((len(self.eta),self.nbins_perp))

		
		#make the modes
		bin_edges = np.histogram_bin_edges(self.k_perp_mag, bins = self.nbins_perp)
		self.k_perp_bin = (bin_edges[1:])
		

		# now the pspec is in (kx,ky,kz) so we want to go through each kz frouier mode and collapse the kxky into 1D to get a 2D pspec 
		for l in range(len(self.k_par)):
			a = np.zeros(len(bin_edges)-1) #holds real stuff..here you need to take the number of BINS not bin edges! # you alwaysneed an extra edge than you have bin!
			c = np.zeros_like(a) #c holds, in each element, the number of pixels 

			for i in range(self.data.shape[0]) : #theta_y direction
				for j in range(self.data.shape[1]): #theta_x direction
					kx = ((j-(self.data.shape[0]/2))*self.delta_kx)#need to multiply by kdelta to get your k units
					ky = ((i-(self.data.shape[1]/2))*self.delta_ky)
					kmag = np.sqrt((kx**2) + (ky**2))
					for k in range(len(bin_edges)-1): #make sure that you speed this up by not considering already binned ps's
						if bin_edges[k] < kmag <= bin_edges[k+1]: 
							a[k] += np.real(self.ps_data[i,j,l]) #[mk^2 Mpc^6]
							c[k] += 1
						else:
							pass

			arg = np.argwhere(np.isnan(a)), np.where(c == 0) # Make sure there are no nans! If there are make them zeros. Also make sure you never divide by 0!
			if len(arg) > 0:
				for i in range(len(arg)):
					a[arg[i]] = 0
					c[arg[i]]=1
			else:
				pass

			ps = (a/c)/(self.volume3D)# [mk^2 Mpc^3]	# not suer about this 2D 3D vol thing	
			self.pspec_2D[l,:] = ps
			
		self.k_par = np.asarray(self.k_par)
		self.k_perp_bin = np.asarray(self.k_perp_bin)

		# return self.k_par, self.k_perp_bin,self.pspec_2D #return k_par, k_perp_binned, pspec_2D


	def compute_kmag(self):

		self.compute_kperp_kpar()
		self.compute_2D_pspec()


		self.k_par = np.fft.fftshift(self.k_par)
	

		kmag = []
		for i in range(len(self.k_par)): 
			for j in range(len(self.k_perp_mag)):
				kmag.append(np.sqrt(self.k_perp_mag[j]**2 + self.k_par[i]**2))

		self.k_mag = np.asarray(kmag)


	def compute_1d_from_2d(self):

		self.compute_kmag()
		self.compute_2D_pspec()

		bin_edges = np.histogram_bin_edges(np.sort(self.k_mag), bins = self.nbins) # find the perp bins
		self.k_modes = bin_edges[1:]

		a = np.zeros(len(bin_edges)-1) #holds real stuff..here you need to take the number of BINS not bin edges! # you alwaysneed an extra edge than you have bin!
		c = np.zeros_like(a) #c holds, in each element, the number of pixels 

		for i in range(len(self.k_par)):
			for j in range(self.nbins_perp):
				kpar = (i - (self.data.shape[2]/2))*self.delta_kz
				kperp = self.k_perp_bin[j]
				kmag = np.sqrt((kpar**2)+(kperp**2))
				for k in range(len(bin_edges)-1): #make sure that you speed this up by not considering already binned ps's
						if bin_edges[k] < kmag <= bin_edges[k+1]: 
							a[k] += np.real(self.pspec_2D[i,j]) #[mk^2 Mpc^6]
							c[k] += 1
						else:
							pass

		arg = np.argwhere(np.isnan(a)), np.where(c == 0) # Make sure there are no nans! If there are make them zeros. Also make sure you never divide by 0!
		if len(arg) > 0:
			for i in range(len(arg)):
				a[arg[i]] = 0
				c[arg[i]]=1
		else:
			pass

		self.pspec_1D = (a/c)


	def compute_dimensionless_1D_pspec(self):

		self.compute_1D_pspec()

		self.dimensionless_pspec =  ((self.k_modes**3) * self.pspec_1D)/(2*np.pi)*2 #[mk^2]


	def compute_1D_pspec(self):# this is very confusing, i am not sure how to do this now that k_perp is not from - to +

		self.compute_volume()
		self.cosmo_FFT3() 
		self.compute_kperp_kpar()
		self.compute_kmag()
		# This is actually really easy since you are collapsing to 1D. You just want to use the check kmag thing to find all 
		# the P(k_perp,k_par) that are in that bin and then average them together. Literally just treat it like the old 2D case.

		
		#make the modes
		bin_edges = np.histogram_bin_edges(np.sort(self.k_mag), bins = self.nbins) # find the perp bins
		self.k_modes = bin_edges[1:]
		
		# now the pspec is in (kx,ky,kz) so we want to go through each kz frouier mode and collapse the kxky into 1D to get a 2D pspec 
		a = np.zeros(len(bin_edges)-1) #holds real stuff..here you need to take the number of BINS not bin edges! # you alwaysneed an extra edge than you have bin!
		c = np.zeros_like(a) #c holds, in each element, the number of pixels 

		for l in range(self.data.shape[2]):
			for i in range(self.data.shape[0]) : #theta_y direction
				for j in range(self.data.shape[1]): #theta_x direction
					kx = ((j-(self.data.shape[1]/2))*self.delta_kx) #need to multiply by kdelta to get your k units
					ky = ((i-(self.data.shape[0]/2))*self.delta_ky)
					kz = ((l - (self.data.shape[2]/2))*self.delta_kz)
					kmag = np.sqrt((kx**2) + (ky**2)+(kz**2))

					for k in range(len(bin_edges)-1): #make sure that you speed this up by not considering already binned ps's
						if bin_edges[k] < kmag <= bin_edges[k+1]:
							a[k] += np.real(self.ps_data[i,j,l]) #[mk^2 Mpc^6]
							c[k] += 1
						else:
							pass
		arg = np.argwhere(np.isnan(a)), np.where(c == 0) # Make sure there are no nans! If there are make them zeros. Also make sure you never divide by 0!
		if len(arg) > 0:
			for i in range(len(arg)):
				a[arg[i]] = 0
				c[arg[i]]=1
		else:
			pass

		ps = (a/c)*(1/self.volume3D)# [mk^2 Mpc^3]		
		self.ps_1D = ps
		
	# return self.k_modes, self.pspec_1D #k_modes and 3D spec 



