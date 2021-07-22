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
	
	def __init__(self, M_matrix, npix_row, npix_col,delta_phi, delta_theta,freq, nbins, norm, linear_bin):
		
		self.Npix = M_matrix.shape[1]
		self.npix_row = npix_row
		self.npix_col= npix_col
		self.nbins = nbins
		self.norm = norm
		self.linear = linear_bin

		z = (1420/freq) - 1 ##freqs must be in MHz

		
		

		self.delta_phi = cosmo.comoving_distance(z).value * delta_phi
		self.delta_theta = cosmo.comoving_distance(z).value * delta_theta

		self.L_x = self.delta_phi*self.npix_col
		self.L_y = self.delta_theta * self.npix_row


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

		self.window = np.real((self.M_tilde)*(np.conj(self.M_tilde).T))# now we have the full npix x npix window matrix. 
		## CORRECT UNTIL HERE!

		factor = (self.L_x*self.L_y)


		# self.window /= (self.L_x*self.L_y) # unitless 


		#so now, it should be the case that if i reshape k, k[30] = k_reshape[1,0] --> can be a test
		#k_reshape = np.reshape(k,(50,30)) #make sure same reshape as m in the loop.

	def circular_bin(self): 

		self.FFT()

		hist, self.bin_edges = np.histogram(self.k, bins = self.nbins)

	


		############ COLLAPSE ALONG COLUMNS #################
		

		self.W_col_collapse = np.zeros((self.Npix,self.nbins))

		for h in range(self.window.shape[1]):

			a = np.zeros(len(self.bin_edges)-1) #holds real stuff..here you need to take the number of BINS not bin edges! # you alwaysneed an extra edge than you have bin!

			#c holds, in each element, the number of pixels 
			c = np.zeros_like(a)

			w = np.reshape(self.window[h],(self.npix_col, self.npix_row))


			for i in range(w.shape[0]) : 
				for j in range(w.shape[1]):
					kx = ((i-(w.shape[0]/2))*self.delta_kx)#need to multiply by kdelta to get your k units
					ky = ((j-(w.shape[1]/2))*self.delta_ky)
					kmag = np.sqrt((kx**2) + (ky**2))
					for k in range(len(self.bin_edges)-1): #make sure that you speed this up by not considering already binned ps's
						if self.bin_edges[k] < kmag <= self.bin_edges[k+1]:

							a[k] += w[i,j]
							c[k] += 1
							break

			arg = np.argwhere(np.isnan(a)), np.where(c == 0) # Make sure there are no nans! If there are make them zeros. Also make sure you never divide by 0!
			if len(arg) > 0:
				for i in range(len(arg)):
					a[arg[i]] = 0
					c[arg[i]]=1
			else:
				pass


			col_collapse = a/c 

			self.W_col_collapse[h] = np.asarray(col_collapse)


		################ COLLAPSE ALONG ROWS ################


		self.W_collapse = np.zeros((self.nbins,self.nbins))

		for h in range(self.W_col_collapse.shape[1]):

			a = np.zeros(len(self.bin_edges)-1) #holds real stuff..here you need to take the number of BINS not bin edges! # you alwaysneed an extra edge than you have bin!

			#c holds, in each element, the number of pixels 
			c = np.zeros_like(a)

			w = np.reshape(self.W_col_collapse[:,h],(self.npix_col, self.npix_row))

			for i in range(w.shape[0]) : 
				for j in range(w.shape[1]):
					kx = ((i-(w.shape[0]/2))*self.delta_kx)#need to multiply by kdelta to get your k units
					ky = ((j-(w.shape[1]/2))*self.delta_ky)
					kmag = np.sqrt((kx**2) + (ky**2))
					for k in range(len(self.bin_edges)-1): #make sure that you speed this up by not considering already binned ps's
						if self.bin_edges[k] < kmag <= self.bin_edges[k+1]: 
							a[k] += w[i,j]
							c[k] += 1

							
							break

			arg = np.argwhere(np.isnan(a)), np.where(c == 0) # Make sure there are no nans! If there are make them zeros. Also make sure you never divide by 0!
			if len(arg) > 0:
				for i in range(len(arg)):
					a[arg[i]] = 0
					c[arg[i]]=1
			else:
				pass

	
			row_collapse = a/c

			self.W_collapse[h] = np.asarray(row_collapse)


			self.window_binned = self.W_collapse.T

	def sort_by_k(self):

		### I think there is something wrong with the sorting...####

		self.FFT()

		indices = np.argsort(self.k) #find the indices that sort self.k
		J = np.take(self.window,indices,axis = 0) #axis 0 reorders rows, this used to be M_tilde
		window_sorted = np.take(J,indices,axis =1)#axis 1 sorts the columns into the right order
		self.window_sorted = np.asarray(window_sorted) #why is this transpose here??? I think because of reshaping but check
		self.k_sorted = np.sort(self.k) # now you can sort k's 

	def bin_window(self):

		''' Here we bin the window function along both axes. First we collapse the columns, then the rows!'''

		self.sort_by_k()

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

	def normalize_bins(self):
		'''Here, each row of W is normalized'''
		if self.linear == True:

			self.bin_window()
		else:
			self.circular_bin()
		window_binned = []

		#normalization
		for i in range(self.window_binned.shape[0]):
			norm = 1/(np.sum(self.window_binned[i])) #find the normalization factor which is 1/sum of the row
			window_binned.append(self.window_binned[i]*norm)

		self.window_binned = np.asarray(window_binned)

	def evaluate_pspec_theory(self,pspec): 

		''' Here we evaluate the theoretical power spectrum.
		There are two cases:
		1) P(k) analystic and we can just plug in k's 
		2) P(k) is an array and so we interpolate and evaluate at the correct k

		NOTE: if you are using an array as your theory spec, it must be a 2D array
		where the first row is k and the second row is P(k).'''

		if self.linear == True:

			self.bin_window()
		else:
			self.circular_bin()

		if isinstance(pspec,np.ndarray) or isinstance(pspec,tuple):
			#here you interpolate


			k_theory = pspec[0]
			p_theory = pspec[1] 


			f = interp1d(k_theory, p_theory, fill_value="extrapolate")

			self.pspec_binned = f(self.bin_edges[1:])
  
			arg = np.argwhere(np.isnan(self.pspec_binned)) # Make sure there are no nans! If there are make them zeros. 
			if len(arg) > 0:
				self.pspec_binned[arg] = 0
			else:
				pass

		elif callable(pspec): 
			# this is the case wehre psepc is a callable function

			self.pspec_binned = []

			for i in range(len(self.bin_edges)-1):
				delta_bin = np.abs(self.bin_edges[i+1] - self.bin_edges[i])
				ps_bin = (1/delta_bin) * integrate.quad(pspec,self.bin_edges[i],self.bin_edges[i+1])[0]
				self.pspec_binned.append(ps_bin)

		else: 
			raise ValueError('Please give me a theoretical spectrum!')

	def k_plotting(self):

		'''Take window binned and find the k at 50th percentile'''

		if self.norm == True:
			self.normalize_bins()
		else:
			pass

		self.k_to_plot = []

		for i in range(self.window_binned.shape[0]): #pick a row!
			riemann_sums = []
			for j in range(self.window_binned.shape[0]):
				integral = np.trapz(self.window_binned[i,:j+1]) # find the riemann sum up to the index
				riemann_sums.append(integral) #append it and repeat

			areas = np.asarray(riemann_sums/max(riemann_sums))
			# before interpolating make sure that both areas and k are STRICTLY INCREASING!


			if np.all(np.diff(areas) > 0) == True:
				pass
			else:
				pass
				# print('areas fail')

			if np.all(np.diff(self.bin_edges) > 0) == True: 
				pass
			else:
				pass
				# print('k fail')	

			f = interp1d(areas,self.bin_edges[:self.nbins]) # interpolate the inverse function integral  = f(k) (remove the rightmost bin edge!)
			self.k_to_plot.append(float(f(0.5))) # append the k where the area under the curve is 50% of the total

		self.k_to_plot = np.asarray(self.k_to_plot)
		self.delta_k = self.k_to_plot[1]-self.k_to_plot[0]

	def compute_pspec_estimate(self,pspec): 
		if self.norm == True:	
			self.normalize_bins()
		else:
			pass

		self.evaluate_pspec_theory(pspec)
		self.k_plotting()

		self.pspec_estimate = np.dot(self.window_binned,self.pspec_binned) #unit of mk^2 Mpc^2 since window is unitless


		return  self.k_to_plot, self.bin_edges, self.pspec_estimate



# # #gotta carefully reorder things so we have -k flip both axes

			# if i == 0:
			# 	plt.imshow(np.real(m_shift2))
				
			# else:
			# 	pass

			# M_flip_col = np.zeros_like(m_shift2)


			# ################## -kx' ################################
			
			# if m_shift2.shape[1] % 2 == 0: #check if the shape is even
			# ## EVEN CASE ##

			# 	for i in range(m_shift2.shape[1]): 
			# 		if self.k_col[i] == min(self.k_col): #most negative number case
			# 			M_flip_col[:,i] = m_shift2[:,i]
						
			# 		elif self.k_col[i] == 0: #0 case
			# 			M_flip_col[:,i] = m_shift2[:,i]
					
			# 		else: #others flip
			# 			j = np.where(self.k_col == -self.k_col[i])[0][0]
			# 			M_flip_col[:,i] = m_shift2[:,j]

			# else: 
			# ## ODD CASE ##
			# 	for i in range(m_shift2.shape[1]): 
						
			# 		if self.k_col[i] == 0: #0 case
			# 			M_flip_col[:,i] = m_shift2[:,i]
					
			# 		else: #others flip
			# 			j = np.where(self.k_col == -self.k_col[i])[0][0]
			# 			M_flip_col[:,i] = m_shift2[:,j]


			# ########### -ky' #########
			
			# self.M_flip_tot = np.zeros_like(m_shift2)

			# if m_shift2.shape[1] % 2 == 0:
			# ### EVEN CASE ##

			
			# 	for i in range(m_shift2.shape[0]):
			# 		if self.k_row[i]==min(self.k_row): #most negative number case
			# 			self.M_flip_tot[i,:] = M_flip_col[i,:]
						
			# 		elif self.k_row[i] == 0: #0 case
			# 			self.M_flip_tot[i,:] = M_flip_col[i,:]
					
			# 		else: #others flip
			# 			j = np.where(self.k_row == -self.k_row[i])[0][0]
			# 			self.M_flip_tot[i,:] = M_flip_col[j,:]

			# ## ODD CASE ##
			# else: 
			# 	for i in range(m_shift2.shape[0]): 

			# 		if self.k_row[i] == 0: #0 case
			# 			self.M_flip_tot[i,:] = M_flip_col[i,:]
					
			# 		else: #others flip
			# 			j = np.where(self.k_row == -self.k_row[i])[0][0]
			# 			self.M_flip_tot[i,:] = M_flip_col[j,:]






