import numpy as np
from scipy.interpolate import interp1d

class Window_Function(object):

	"""
	Object to perform computations for computing the window function and 
	estimated power sepctrum of a given experiment.

	One thing that is maybe useful is that if you are using the HERA_hack_FG code 
	to generate the M_matrix, there is a useful method to find out the npix_col and npix_row. 
	run the observation class method "sky_shape()". The first output is npix_col, the second is npix_row. 

	Note to self: try to implement a function that auto-bins in bayesian block binning
	"""
	
	def __init__(self, M_matrix, npix_row, npix_col, L_y,L_x, nbins):
		self.M_matrix = M_matrix
		self.Npix = M_matrix.shape[1]
		self.npix_row = npix_row
		self.npix_col= npix_col
		self.nbins = nbins
		self.delta_krow = L_y/self.npix_row
		self.delta_kcol = L_x/self.npix_col

	def FFT(self): 

		''' Here we compute the 2D fft and also find the correspinding k's'''
		#first 2d fft where we fold along each row, 2d fft it, and unfold it. 

		M_bar  = []
		k_col = 0
		k_row = 0 

		for i in range(self.Npix): 
			m = self.M_matrix[i]
			m = np.reshape(m,(self.npix_col, self.npix_row))#this stacks rows of m 
			m_bar = np.fft.fft2(m) 
			if i == 1:
				k_col = np.fft.fftfreq(m_bar.shape[0],d = self.delta_kcol) #need to add 2pis
				k_row = np.fft.fftfreq(m_bar.shape[1], d = self.delta_krow) # need to add 2pis
			else:
				pass
			m_bar = np.reshape(m_bar,(self.Npix,))
			M_bar.append(m_bar)

		Mbar = np.asarray(M_bar)
		self.M_tilde = []

		#second 2d fft where we fold each column, 2d fft it, and unfold it. 
		for i in range(self.Npix): 
			m = Mbar[:,i]
			m = np.reshape(m,(self.npix_col,self.npix_row))#this stacks rows of m so m[50]=m[1,0]
			m_bar = np.fft.fft2(m)
			m_bar = np.reshape(m_bar,(self.Npix,))
			self.M_tilde.append(m_bar)

		self.M_tilde = np.asarray(self.M_tilde).T

		#should have a line here that has all of the k values of all the entries. call is self.k
		k = []
		#you UST loMop through the row first then the col in order to match m reshape
		for i in range(len(k_row)): 
			for j in range(len(k_col)):
				k.append(np.sqrt(k_col[j]**2 + k_row[i]**2))

		self.k = np.asarray(k)

		#so now, it should be the case that if i reshape k, k[30] = k_reshape[1,0] --> can be a test
		#k_reshape = np.reshape(k,(50,30)) #make sure same reshape as m in the loop.

	
	def negative_index(self):

		self.FFT()

		''' It turns out that the Window function is what i like to call the 
		"almost" fourier transform of M. The difference is that some of the poisitons of entries
		of the second index are swapped. This method does the entry swapping as well as the mod-suqaring.'''

		Wmat = np.zeros_like(self.M_matrix)

		for i in range(Wmat.shape[0]):
			for j in range(Wmat.shape[1]):
				Wmat[i,j] = self.M_tilde[i,-j] #this used to be M_tilde_sorted

		self.window = ((np.conj(Wmat)))*(Wmat) # now we have the full npix x npix window matrix.

	def sort_by_k(self):
		self.negative_index()

		s=np.argsort(self.k)
		J = np.take(self.window,s,axis = 0) #axis 0 reorders rows, this used to be M_tilde
		window_sorted = np.take(J,s,axis =1)#axis 1 sorts the columns into the right order
		self.window_sorted = np.asarray(window_sorted) #why is this transpose here??? I think because of reshaping but check
		self.k_sorted = np.sort(self.k) # now you can sort k's 


	def bin_window(self):

		''' Here we bin the window function along both axes. First we collapse the columns, then the rows!'''

		self.sort_by_k()

		hist, self.bin_edges = np.histogram(self.k_sorted, bins = self.nbins)

		W_col_collapse = np.zeros((self.Npix,self.nbins))

		for j in range(self.Npix): # pick a row!
			min_index = 0
			for i in range(len(self.bin_edges)-1): #pick a bin!
				max_index = np.sum(hist[:i+1])#hist[i] + min_index 
				min_index = np.sum(hist[:i])
				####### bin W values #####
				w_real = np.real(self.window_sorted)
				a = np.sum(w_real[j,min_index:max_index]) #for row j, sum the columns from min to max index of the bin
				c = hist[i] #number of P_k values in that bin 
				W_col_collapse[j,i] = a/c #compute average W that bin 
								
		self.W_collapse = np.zeros((self.nbins,self.nbins)) 

		for j in range(self.nbins): # pick a column, only 30 now!
			min_index = 0
			for i in range(len(self.bin_edges)-1): #pick a bin!
				max_index = np.sum(hist[:i+1])#hist[i] + min_index 
				min_index = np.sum(hist[:i])
				####### bin W values #####
				w_real = np.real(W_col_collapse)
				a = np.sum(w_real[min_index:max_index,j])
				c = hist[i] #number of entries in that bin 
				self.W_collapse[i,j] = a/c #compute average W in that bin

	def normalize_bins(self):
		'''Here, each row of W is normalized'''

		self.bin_window()

		window_binned = []

		#normalization
		for i in range(self.W_collapse.shape[0]):
			norm = 1/(np.sum(self.W_collapse[i])) #find the normalization factor which is 1/sum of the row
			window_binned.append(self.W_collapse[i]*norm)

		self.window_binned = np.asarray(window_binned)


	def evaluate_pspec_theory(self,pspec): 

		''' Here we evaluate the theoretical power spectrum.
		There are two cases:
		1) P(k) analystic and we can just plug in k's 
		2) P(k) is an array and so we interpolate and evaluate at the correct k

		NOTE: if you are using an array as your theory spec, it must be a 2D array
		where the first row is k and the second row is P(k).'''

		self.bin_window()

		if isinstance(pspec,np.ndarray) or isinstance(pspec,tuple):
			#here you interpolate


			k_theory = pspec[0]
			p_theory = pspec[1] 


			f = interp1d(k_theory, p_theory, fill_value="extrapolate")

			self.pspec_binned = f(self.bin_edges[1:])

			arg = np.argwhere(np.isnan(self.pspec_binned)) # Make sure there are no nans! If there are make them zeros. 
			if len(arg) > 0:
				self.pspec_binned[arg] = 0
				print(self.pspec_binned)
			else:
				pass

		elif callable(pspec): 
			# this is the case wehre psepc is a callable function
			self.pspec_binned = pspec(self.bin_edges[1:])

		else: 
			raise ValueError('Please give me a theoretical spectrum!')

	def k_plotting(self):

		'''Take window binned and find the k at 50th percentile'''
		self.normalize_bins()

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
				print('areas fail')

			if np.all(np.diff(self.bin_edges) > 0) == True: 
				pass
			else:
				print('k fail')	

			f = interp1d(areas,self.bin_edges[:self.nbins]) # interpolate the inverse function integral  = f(k) (remove the rightmost bin edge!)
			self.k_to_plot.append(float(f(0.5))) # append the k where the area under the curve is 50% of the total

		self.k_to_plot = np.asarray(self.k_to_plot)


	def compute_pspec_estimate(self,pspec): 

		self.normalize_bins()
		self.evaluate_pspec_theory(pspec)
		self.k_plotting()

		self.pspec_estimate = np.dot(self.window_binned,self.pspec_binned)

		print(self.pspec_binned)
		print(self.pspec_estimate)

		return  self.k_to_plot , self.bin_edges , self.pspec_estimate 






