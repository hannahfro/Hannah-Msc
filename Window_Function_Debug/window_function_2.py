import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt


class Window_Function(object):

	"""
	Object to perform computations for computing the window function and 
	estimated power sepctrum of a given experiment.

	One thing that is maybe useful is that if you are using the HERA_hack_FG code 
	to generate the M_matrix, there is a useful method to find out the npix_col and npix_row. 
	run the observation class method "sky_shape()". The first output is npix_col, the second is npix_row. 

	Note to self: try to implement a function that auto-bins in bayesian block binning
	"""
	
	def __init__(self, M_matrix, npix_row, npix_col,Lx,Ly,freq, nbins):
		
		self.Npix = M_matrix.shape[1]
		self.npix_row = npix_row
		self.npix_col= npix_col
		self.nbins = nbins
		

		z = (1420/freq) - 1 ##freqs must be in MHz ### 
		#make sure this makes sense for mutliline window

		self.L_x = Lx
		self.L_y = Ly

		

		self.delta_phi = self.L_x/self.npix_col
		self.delta_theta = self.L_y/self.npix_row

		

		self.M_matrix = np.real(M_matrix)

		
	def FFT(self): 

		''' Here we compute the 2D fft and also find the correspinding k's'''
		#first 2d fft where we fold along each row, 2d fft it, and unfold it. 

		M_bar  = []
		k_col = 0
		k_row = 0 

		for i in range(self.Npix): 
			m = self.M_matrix[i] #fix a row (the elements are now labelled by k')
			m = np.reshape(m,(self.npix_col, self.npix_row)).T #this stacks rows of m 
			self.m_fft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m))) * (self.npix_col*self.npix_row) # multiply by N^2 to get fourier convention
			# Gotta shift both axes
			if i == 0:
				self.k_col = np.fft.fftshift(np.fft.fftfreq(self.m_fft.shape[1], d = self.delta_phi)) #k's labelling the columns
				self.k_row = np.fft.fftshift(np.fft.fftfreq(self.m_fft.shape[0], d = self.delta_theta)) #k's labelling the rows 
				self.k_col *= 2*np.pi #change fourier convention
				self.k_row *= 2*np.pi #change fourier convention
				self.delta_ky = np.abs(self.k_row[1]-self.k_row[0])
				self.delta_kx = np.abs(self.k_col[1]-self.k_col[0])
			else:
				pass
			# unfold and append
			self.m_fft *= ((self.delta_kx*self.delta_ky)/((2*np.pi)**2)) # 1/mpc^2
			m_bar = np.reshape(self.m_fft,(self.Npix,))
			M_bar.append(m_bar)

		self.Mbar = np.asarray(M_bar) #CORRECT UNITL HERE!
		self.M_tilde = []

		#second 2d fft where we fold each column, 2d fft it, and unfold it. 

		for i in range(self.Npix): 
			m = self.Mbar[:,i]
			m = np.reshape(m,(self.npix_col,self.npix_row))#this stacks rows of m so m[50]=m[1,0]
			m_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m) * (self.delta_theta *self.delta_phi)))  #unitless
			#unfold the column and append it back to big array
			m_bar = np.reshape(m_fft,(self.Npix,))
			self.M_tilde.append(m_bar)

		self.M_tilde = np.asarray(self.M_tilde).T# unit Mpc^2 this is in fact not exactly M tilde, but has some elements swapped places along axis 1		
		# should be the case that this times x_true_tilde = x_obs_tilde 

		#should have a line here that has all of the k values of all the entries. call is self.k
		k = []
		#you MUST loop through the row first then the col in order to match m reshape
		for i in range(len(self.k_row)): 
			for j in range(len(self.k_col)):
				k.append(np.sqrt(self.k_col[j]**2 + self.k_row[i]**2))

		self.k = np.asarray(k)

		self.window = np.real((self.M_tilde)*(np.conj(self.M_tilde)))# now we have the full npix x npix window matrix. 
	

######################## METHODS RELATED TO POWER SPECTRUM ESTIMATION ################################

	def p_estimate_full(self,cov):

		self.FFT()

		'''compute the full k-vector estimate spectrum takes in the un-sorted covariance diagonal of the input theory field'''

		if isinstance(cov,np.ndarray) or isinstance(cov,tuple):

			ps = cov*((self.delta_phi*self.delta_theta)**2)/(self.L_x*self.L_y)

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

		hist, self.bin_edges = np.histogram(self.k_sorted, bins = self.nbins)
		self.pk_window_binned = np.zeros(self.nbins)

		min_index = 0
		for i in range(len(self.bin_edges)-1): #pick a bin!
		    max_index = np.sum(hist[:i+1])#hist[i] + min_index 
		    min_index = np.sum(hist[:i])
		    a = np.sum(self.MM_cov_diag_sorted[min_index:max_index]) #for row j, sum the columns from min to max index of the bin
		    c = hist[i] #number of P_k values in that bin 
		    self.pk_window_binned[i] = a/c #compute average W that bin


	def compute_pspec_estimate(self,cov): 

		self.spec_binning(cov)

		return  self.bin_edges[1:], self.pk_window_binned


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

		self.W_col_collapse = np.zeros((self.Npix,self.nbins))

		for j in range(self.Npix): # pick a row!
			min_index = 0
			for i in range(len(self.bin_edges)-1): #pick a bin!
				max_index = np.sum(hist[:i+1])#hist[i] + min_index 
				min_index = np.sum(hist[:i])
				####### bin W values #####
				w_real = np.real(self.window_sorted)
				a = np.sum(w_real[j,min_index:max_index]) #for row j, sum the columns from min to max index of the bin
				c = hist[i] #number of P_k values in that bin 
				self.W_col_collapse[j,i] = a/c #compute average W that bin 
								
		self.W_collapse = np.zeros((self.nbins,self.nbins)) 

		for j in range(self.nbins): # pick a column, only 30 now!
			min_index = 0
			for i in range(len(self.bin_edges)-1): #pick a bin!
				max_index = np.sum(hist[:i+1])#hist[i] + min_index 
				min_index = np.sum(hist[:i])
				####### bin W values #####
				w_real = np.real(self.W_col_collapse)
				a = np.sum(w_real[min_index:max_index,j])
				c = hist[i] #number of entries in that bin 
			
				self.W_collapse[i,j] = a/c #compute average W in that bin

		self.window_binned = self.W_collapse


	def normalize_window(self):

		'''normalize the rows of the binned window'''

		self.bin_window()

		window_norm = []

		#normalization
		for i in range(self.window_binned.shape[0]):
			norm = 1/(np.sum(self.window_binned[i])) #find the normalization factor which is 1/sum of the row
			window_norm.append(self.window_binned[i]*norm)

		self.window_norm = np.asarray(window_norm)

	def compute_sigma(self):

		self.normalize_window()

		self.sigma_k = np.std(self.window_norm, axis = 0)

		return self.sigma_k 




# 1) normalized window function binning 
# 2) FWHM error bars
# 3) 













