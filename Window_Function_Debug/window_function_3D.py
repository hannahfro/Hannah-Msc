import numpy as np
from scipy import interpolate
import scipy.integrate as integrate
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt
import scipy.constants as sc

from timeit import default_timer as timer


class Window_Function(object):

	"""
	Object to perform computations for computing the window function and 
	estimated power sepctrum of a given experiment.

	One thing that is maybe useful is that if you are using the HERA_hack_FG code 
	to generate the M_matrix, there is a useful method to find out the npix_col and npix_row. 
	run the observation class method "sky_shape()". The first output is npix_col, the second is npix_row. 

	This computs the window function for a single 8 MHz chunk of data (i.e when it is appropriate to make a coeval approximation)

	Note to self: try to implement a function that auto-bins in bayesian block binning
	"""
	
	def __init__(self, M_matrix, npix_row, npix_col,theta_x,theta_y,freqs,rest_freq, nbins):
		
		self.Npix = M_matrix.shape[1]
		self.nfreqs = M_matrix.shape[2]
		self.freqs = freqs
		self.npix_row = npix_row
		self.npix_col= npix_col
		self.nbins = nbins
		self.rest_freq = rest_freq #in MHz
		
		## get all the z info from the mid req 
		if self.nfreqs % 2 == 0 :

			self.mid_freq = self.freqs[(self.nfreqs//2)]
		else: 
			self.mid_freq = self.freqs[(self.nfreqs//2 + 0.5)]
	   
		self.z_mid = (self.rest_freq/self.mid_freq) - 1

		self.Lx = theta_x * cosmo.comoving_distance(self.z_mid).value #Mpc
		self.Ly = theta_y * cosmo.comoving_distance(self.z_mid).value #Mpc 
		self.Lz = (max(self.freqs) - min(self.freqs)) * (((2.99792e5) *((1+self.z_mid)**2))/(cosmo.H0.value * cosmo.efunc(self.z_mid) * self.rest_freq))
		self.delta_y = (self.Ly/self.npix_row) # size of y pixel
		self.delta_x = (self.Lx/self.npix_col) # size of x pixel
		self.delta_z = (self.Lz/self.nfreqs) # size of z pixel | this may be too crude 

		self.delta_freq = (max(self.freqs) - min(self.freqs))/self.nfreqs


		self.M_matrix = np.real(M_matrix) 


#_____________________ METHODS FOR THE WINDOW FUNCTION_____________________________

	def compute_k_perp(self):

		''' Method to compute the perpendicular K mode using the middle frequency field size'''
		
		self.k_col = np.fft.fftshift(np.fft.fftfreq(self.npix_col, d = self.delta_x)) #k's labelling the columns
		self.k_row = np.fft.fftshift(np.fft.fftfreq(self.npix_row, d = self.delta_y)) #k's labelling the rows 
		self.k_col *= 2*np.pi #change fourier convention
		self.k_row *= 2*np.pi #change fourier convention
		self.delta_ky = np.abs(self.k_row[1]-self.k_row[0])
		self.delta_kx = np.abs(self.k_col[1]-self.k_col[0])

		#should have a line here that has all of the k values of all the entries. call is self.k
		k_perp = []
		#you MUST loop through the row first then the col in order to match m reshape
		for i in range(len(self.k_row)): 
			for j in range(len(self.k_col)):
				k_perp.append(np.sqrt(self.k_col[j]**2 + self.k_row[i]**2))

		self.k_perp = np.asarray(k_perp)

		self.delta_k_perp =  np.abs(self.k_perp[1]-self.k_perp[0])

	def compute_k_par(self):

		# 
		eta = np.fft.fftshift(np.fft.fftfreq(self.nfreqs, d = self.delta_freq)) 

		self.k_par_minus_prime = eta * ((2*np.pi*(self.rest_freq)*cosmo.H0.value *1000* cosmo.efunc(self.z_mid))/(sc.c*((1+self.z_mid)**2)))

		self.delta_k_par = np.abs(self.k_par_minus_prime[1] - self.k_par_minus_prime[0])

	def compute_2D_FFT(self): 

		''' Here we compute the 3D fft and also find the correspinding k's. To do this
		we need to first work in the observers convetion and then convert back'''
		#first 2d fft where we fold along each row, 2d fft it, and unfold it. 

		# self.stack_of_2d_Mtilde = np.zeros((self.Npix,self.Npix,self.nfreqs) ,dtype = complex)
		
		#define k_perps and delta_k_perps here. 

		self.Mbar  = np.zeros((self.Npix,self.Npix,self.nfreqs), dtype = complex)
	
		for freq in range(self.nfreqs): # for each slice of M for each map, compute the window function. #self.nfreqs
			M = self.M_matrix[:,:,freq]
			for i in range(self.Npix): 
				m = M[i] #fix a row (the elements are now labelled by k')
				m = np.reshape(m,(self.npix_col, self.npix_row)).T #this stacks rows of m 
				self.m_fft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m))) * (self.npix_col*self.npix_row) # multiply by N^2 to get fourier convention
		
				#unfold and append
				self.m_fft *= ((self.delta_x*self.delta_y)/((2*np.pi)**2)) # 1/mpc^2
				m_bar = np.reshape(self.m_fft,(self.Npix,))
				self.Mbar[i,:,freq] = m_bar

		self.M_tilde = np.zeros((self.Npix,self.Npix,self.nfreqs), dtype = complex)

			#second 2d fft where we fold each column, 2d fft it, and unfold it. 
		for freq in range(self.nfreqs):
			for i in range(self.Npix): 
				m = self.Mbar[:,i, freq]
				m = np.reshape(m,(self.npix_col,self.npix_row))#this stacks rows of m so m[50]=m[1,0]
				m_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m)))* (self.delta_x *self.delta_y)  #unitless
				#unfold the column and append it back to big array
				m_bar = np.reshape(m_fft,(self.Npix,))
				self.M_tilde[:,i,freq] = m_bar

			# unit Mpc^2 this is in fact not exactly M tilde, but has some elements swapped places along axis 1		
			# Test 1: should be the case that this times x_true_tilde = x_obs_tilde 
			#GOOD UNTIL M_TILDE !!!		

		print('perp FFT done')

	def compute_freq_FFT(self):

		self.compute_2D_FFT()

		self.full_Mtilde = np.zeros((self.Npix,self.Npix,self.nfreqs), dtype = complex)

		for i in range(self.Npix):
			for j in range(self.Npix):
				m_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.M_tilde[i,j,:]))) * self.delta_z #change this to the binned kperp #self.window_k_perp_binned_2[i,j,:])#self.window_perp_sorted[i,j,:]))
				m_fft /= (2*np.pi) #should have a total of 1/(2pi)^3 so we had two above and one here.
				self.full_Mtilde[i,j,:] = m_fft 

		print('parallel FFT done')

	def compute_square(self):

		
		self.compute_freq_FFT()

		self.window = np.real((self.full_Mtilde)*(np.conj(self.full_Mtilde)))# now we have the full npix x npix window matrix.

		print('square done')

	def sort_window_kperp(self):

		'''Here we sort the kperp direction so that the window function can be binned according to kx and ky. '''

		self.compute_square()

		indices = np.argsort(self.k_perp) #find the indices that sort self.k
		J = np.take(self.window,indices,axis = 0) #axis 0 reorders rows, 
		window_sorted = np.take(J,indices,axis =1)#axis 1 reorders columns 
		self.window_perp_sorted = np.asarray(window_sorted)
		self.k_perp_sorted = np.sort(self.k_perp) # now you can sort k's

		print('sorted in k_perp')

		#Now we have the full 3D window function but recall that the freq direction is labelled by (kpar - kpar')

# _________________________________________________________METHODS FOR THE 2D PSPEC_________________________________________________

	def compute_2D_pspec_estimate(self,pspec,kpspec,k_obs):

		''' This method takes in the covariance of the true sky as well as the observable k_obs = (kpar,kperp) given an experiment. 
		By defining the k_obs first it allows you to only calculate the k that you care about. '''

		
		self.compute_k_par()
		self.compute_k_perp()
		self.sort_window_kperp()

		self.interpolate_pspec_true(pspec, kpspec) ## you should be able to import the "functionized" true 2D spectrum (i.e. interpolate the input theory spectrum). 

		k_par_obs = k_obs# these are the k's we want to output and need to use k_par_obs to do the subtraction thingy. I think by construction your k_perp_obs are already defined from the window

		self.p_estimate = np.zeros((len(k_par_obs),self.Npix))

		for i in range(len(k_par_obs)): # for each row of the estimate spectrum
			for j in range(self.nfreqs): # sum over kpar_prime
				k_par_prime = -(self.k_par_minus_prime[j]-k_par_obs[i]) 
				p_true = np.reshape(self.f(k_par_prime,self.k_perp_sorted).T,(self.Npix,))
				self.p_estimate[i,:] += (self.window_perp_sorted[:,:,j].dot(p_true))*(self.delta_k_perp*self.delta_k_par) #sum over k_perp_prime


	def one_window(self):
		# self.compute_k_par()
		# self.compute_k_perp()
		# self.sort_window_kperp()


		k_to_plot = 0.9
		print(k_to_plot)
		
		self.k_par_prime = -(self.k_par_minus_prime-k_to_plot)

		print(self.k_par_prime)

		indices = np.argsort(self.k_par_prime) #find the indices that sort self.k

		self.k_prime_sorted = np.sort(self.k_par_prime) # now you can sort k's

		self.window_to_plot = np.take(self.window_perp_sorted[1,:,:],indices,axis = 1) #axis 0 reorders rows, 

		self.window_to_plot /= max(np.reshape(self.window_to_plot, (len(self.k_perp_sorted)*self.nfreqs)))


	def bin_p_estimate(self,pspec,kpspec,k_obs):
		''' here we are going to bin along the k unprimed direction which corresponds to binning rows together. '''
		
		# self.compute_2D_pspec_estimate()
		
		hist, self.bin_edges = np.histogram(self.k_perp_sorted, bins = self.nbins)

		# self.window_k_perp_binned = np.zeros((self.nbins, self.Npix, self.nfreqs))

		# for i in range(len(hist)):
		# 	if i == 0:
		# 		ave = np.sum( self.window_perp_sorted[0:hist[i]-1,:,:], axis  = 0)/(hist[i])
		# 		self.window_k_perp_binned[i] = ave
		# 	else: 
		# 		ave = np.sum( self.window_perp_sorted[np.sum(hist[:i]):np.sum(hist[:i+1])-1,:,:], axis  = 0)/hist[i]
		# 		self.window_k_perp_binned[i] = ave

		# ## can repeat this procedure on axis = 1 if you want to bin the k primed perps as well. 
		self.p_estimate_binned = np.zeros((self.nfreqs, self.nbins))

		for i in range(len(hist)):
			if i == 0:
				ave = np.sum(self.p_estimate[:,0:hist[i]-1], axis  = 1)/(hist[i])
				self.p_estimate_binned[:,i] = ave
			else: 
				ave = np.sum( self.p_estimate[:,np.sum(hist[:i]):np.sum(hist[:i+1])-1], axis  = 1)/hist[i]
				self.p_estimate_binned[:,i] = ave

		print('binned in k_perp')


	######################## METHODS RELATED TO POWER SPECTRUM ESTIMATION ################################

	def interpolate_pspec_true(self, pspec, kpspec):

		''' here you import in the true 2D power spectrum that was made using the pspec code from the realization of the unvierse. '''
		kpar, kperp = kpspec[0], kpspec[1]

		theory_spec_arr = pspec

		self.f = RectBivariateSpline(kpar,kperp, theory_spec_arr)#, bounds_error = False, fill_value = 0)

		# plt.pcolor(kperp,kpar,self.f(kperp,kpar),shading='auto')
		# plt.colorbar()






#_________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________



	def compute_cross(self):

		self.FFT()

		self.window = np.real((self.G_matrix)*(np.conj(self.G_matrix)))#


	def p_estimate_full(self,cov):

		self.compute_cross()

		'''compute the full k-vector estimate spectrum takes in the un-sorted covariance diagonal of the input theory field'''

		if isinstance(cov,np.ndarray) or isinstance(cov,tuple):

			ps = cov*((self.delta_phi*self.delta_theta)**2)/(self.L_x*self.L_y) #Maybe conside still using the covariance...

		elif callable(cov): 
			ps = cov(self.k)

		else:

			raise ValueError('Please give me a theoretical spectrum!')


		self.MM_cov_diag = (self.window).dot(ps)


	def sort_estimate_spec(self,cov):

		self.p_estimate_full(cov)

		indices = np.argsort(self.k) #find the indices that sort self.

		MM_cov_diag_sorted = np.take(self.MM_cov_diag,indices)#axis 1 sorts the columns into the right order
		self.MM_cov_diag_sorted = np.asarray(MM_cov_diag_sorted)#why is this transpose here??? I think because of reshaping but check
		self.k_sorted = np.sort(self.k)

	def spec_binning(self, cov):

		self.sort_estimate_spec(cov)

		idx = np.argwhere(self.k_sorted > 0.15)

		self.k_del = np.delete(self.k_sorted,idx) 

		indices = np.argsort(self.k_del)
		self.MM_cov_del = np.delete(self.MM_cov_diag_sorted, idx)
		self.MM_cov_del = np.take(self.MM_cov_del,indices)
		self.k_del = np.sort(self.k_del)

		hist, self.bin_edges = np.histogram(self.k_del, bins = self.nbins)
		self.pk_window_binned = np.zeros(self.nbins)

		min_index = 0
		for i in range(len(self.bin_edges)-1): #pick a bin!
			max_index = np.sum(hist[:i+1])#hist[i] + min_index 
			min_index = np.sum(hist[:i])
			a = np.sum(self.MM_cov_del[min_index:max_index]) #for row j, sum the columns from min to max index of the bin
			c = hist[i] #number of P_k values in that bin 
			self.pk_window_binned[i] = a/c #compute average W that bin


	def compute_pspec_estimate(self,cov): 

		self.spec_binning(cov)

		return  self.bin_edges[1:self.nbins], self.pk_window_binned[1:] # leave out the bottom and top bin edge, don't plot bottom bin cuz has k = 0 in it


######################################################################################################

############################# METHODS RELATED TO ERROR BAR ###########################################
	def sort_window(self):

		self.FFT()

		indices = np.argsort(self.k) #find the indices that sort self.k
		J = np.take(self.window,indices,axis = 0) #axis 0 reorders rows, 
		window_sorted = np.take(J,indices,axis =1)#axis 1 reorders columns 
		self.window_sorted = np.asarray(window_sorted)
		self.k_sorted = np.sort(self.k) # now you can sort k's 

	def bin_window(self):
		self.sort_window()

		hist, self.bin_edges = np.histogram(self.k_sorted, bins = self.nbins)

		self.W_collapse = np.zeros((self.nbins,self.Npix))

		for j in range(self.nbins): # pick a column, only 30 now!
			min_index = 0
			for i in range(len(self.bin_edges)-1): #pick a bin!
				max_index = np.sum(hist[:i+1])#hist[i] + min_index 
				min_index = np.sum(hist[:i])
				####### bin W values #####
				w_real = np.real(self.window_sorted)
				a = np.sum(w_real[min_index:max_index,j])
				c = hist[i] #number of entries in that bin 
			
				self.W_collapse[i,j] = a/c #compute average W in that bin

		print(self.W_collapse.shape)


		rounded_k = np.around(self.k_sorted, decimals = 3)
		self.reduced_k = np.sort(list(set(rounded_k)))

		binned = []

		for i in range(len(self.reduced_k)):# pick a bin
			a = 0
			c = 0
			for j in range(len(self.k_sorted)): # check which elements are in that bin
				
				if self.reduced_k[i] == np.around(self.k_sorted[j],decimals = 3):
					a += self.W_collapse[:,j]
					c += 1
					
				else:
					pass
			
			binned.append(a/c)

		self.window_binned = np.asarray(binned).T[1:,1:]

		self.reduced_k = self.reduced_k[1:]

	def normalize_window(self):

		'''normalize the rows of the binned window'''

		self.bin_window()

		window_norm = []

		#normalization
		for i in range(self.window_binned.shape[0]):
			norm = 1/(np.sum(self.window_binned[i])) #find the normalization factor which is 1/sum of the row
			window_norm.append(self.window_binned[i]*norm)

		self.window_norm = np.asarray(window_norm)

	def compute_error_bars(self):

	
		self.normalize_window()

		def find_nearest(array, value):
			array = np.asarray(array)
			idx = (np.abs(array - value)).argmin()
			return idx

		lower_bound = np.zeros(self.nbins-1)
		upper_bound = np.zeros(self.nbins-1)
		self.k_to_plot = np.zeros(self.nbins-1)

		for i in range(self.window_binned.shape[0]): # try it with the unbinned version maybe?

			pdf = self.window_binned[i]
			prob = pdf/float(sum(pdf))
			cum_prob = np.cumsum(prob)

			lower_idx = find_nearest(cum_prob,0.16)
			upper_idx = find_nearest(cum_prob,0.84)
			mid_idx = find_nearest(cum_prob, 0.5)

			lower_bound[i] = self.reduced_k[lower_idx]
			upper_bound[i] = self.reduced_k[upper_idx]
			mid_k = self.reduced_k[mid_idx]


			self.k_to_plot[i] = mid_k
			
		self.error_bars = np.asarray(list(zip(lower_bound, upper_bound))).T
		

