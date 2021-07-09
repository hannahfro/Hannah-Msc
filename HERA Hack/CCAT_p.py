import numpy as np 
import scipy.constants as sc
import matplotlib.pyplot as plt

 

def CCAT_p(sky, RA, DEC, freq, dish_diameter, noise, sigma_sys, t_obs, beam_FWHM):

	#Sky should be a 2D image
	#RA
	#DEC 
	#Freq in GHz
	# diam in m
	#noise - BOOL
	# Tsys in Jy/s^1/2 sr 
	#t_obs in s
	#survey size in deg^2
	# beam FWHM in arcmin
	# number of detectors 



	## ok ok so i need to star with physical units and do the convolution in pixel unitsare
	# 1) with the size of the box and the instrument res, find out how many pixels there 
	# 2) using the diff limited angular res, find the res in pixel units
	# 3) convolve the observed box with pixel unit gaussian
	#Convolve sky map with a 2D gaussian 


	def gauss(x,y,sigma_x,sigma_y,mu_x,mu_y):
	    return np.exp(-((((x-mu_x)**2)/(2*(sigma_x**2))) + (((y-mu_y)**2)/(2*(sigma_y**2)))))#* (1/(2*np.pi*sigma_x*sigma_y))

	res = (sc.c/(freq*(10**9)))/dish_diameter #diffaction limited angular resolution in radians 

	beam_FWHM *= np.pi/(60*180)
	

	delta_RA = (RA[1]-RA[0])
	delta_DEC = (DEC[1]-DEC[0])

	omega_pix = np.abs((RA[1]-RA[0])*(DEC[1]-DEC[0])) #in sr

	npix_x = len(RA)
	npix_y = len(DEC)


############ ADD THE NOISE ##########


	

############### CONVOLVE WITH GAUSSIAN ###########

	

	
	# x = np.linspace(-npix_x*delta_RA,npix_x*delta_RA, delta_RA) #[-n*delta_RA,...., n*delta_RA]

	# y = np.linspace(-npix_y*delta_DEC,npix_y*delta_DEC, delta_DEC)

	xx, yy = np.meshgrid(RA,DEC, sparse = True) #probs with the meshgrid

	
	mu_x = RA[(len(RA)//2)]
	mu_y = DEC[(len(DEC)//2)]


	gaussian = np.reshape(gauss(xx,yy,res,res,mu_x,mu_y),(sky.shape[0],sky.shape[1])) 

	# plt.imshow(gaussian)
	# plt.colorbar()



	sky_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky)))
	gauss_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian)))

	


	product = sky_fft*gauss_fft

	new_sky = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(product)))


	if noise == True:

		# calculate 

		omega_pix_ccat = 2*np.pi*(((beam_FWHM/ (2*np.sqrt(2*np.log(2)))))**2) 

		# 1: find Tsys for the given pixel size from sigma_big = sigma_small / root(n)

		npix_in_pix = omega_pix/omega_pix_ccat #pixel size is 0.61276sr from ccat-p specs


		sigma = (sigma_sys)/np.sqrt(npix_in_pix) # this is still in Jy/sr * root(s)

		# calculate t_pix = t_obs*(omega_pix/omega_surv)

		t_pix = (t_obs/np.sqrt(npix_in_pix))


		sigma *= ((10**(-26))*(sc.c**2))/(2*(freq*(10**(9)))**2 * sc.k) # K root(s)

		sigma_rms = sigma/(np.sqrt(t_pix)) 

		noise = np.random.normal(0,sigma_rms, (npix_y,npix_x))


		# plt.imshow(noise)
		# plt.colorbar() # unit issue here! 

		new_sky += noise
	else:
		pass 


	return np.real(new_sky)


# arr = np.zeros((30,30))

# RA = np.linspace(1,1.015,30)
# DEC = np.linspace(1,1.015,30)


# noisy_convolve = CCAT_p(arr,RA,DEC,274,6,True, 2.5e6, 3*(60*60), 0.75) 

                                            