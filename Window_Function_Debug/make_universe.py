
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math


class universe(object):
	"""docstring for ClassName"""
	def __init__(self, ps, row_npix,col_npix, Ly,Lx, mean):

		if isinstance(ps,np.ndarray) or isinstance(ps,tuple):
			#here you interpolate


			k_theory = ps[0]
			p_theory = ps[1] 


			self.ps = interp1d(k_theory, p_theory, fill_value="extrapolate")

		else:
			self.ps = ps #[mk^2*Mpc^2]

		self.row_npix = row_npix
		self.col_npix = col_npix
		self.Lx = Lx
		self.Ly = Ly
		self.mean = mean 

		self.delta_y = self.Ly/np.float(self.row_npix) #sampling rate
		self.delta_x = self.Lx/np.float(self.col_npix) #sampling rate 

		self.delta_ky = (2*np.pi)/np.float(self.Ly)
		self.delta_kx = (2*np.pi)/np.float(self.Lx)

	def compute_k_2D(self):

	
		self.kx = np.fft.fftshift(np.fft.fftfreq(self.col_npix, d = self.delta_x)) 
		self.ky = np.fft.fftshift(np.fft.fftfreq(self.row_npix, d = self.delta_y))

		self.ky *= -2*np.pi
		self.kx *= -2*np.pi


		k = []

		#Careful! need to fix a y and cycle through x to get the right concentric rings in your kbox!

		for i in range(len(self.ky)): 
			for j in range(len(self.kx)):
				k.append(np.sqrt(self.kx[j]**2 + self.ky[i]**2)) 


		self.k = np.asarray(k)
		
		self.ksorted= np.asarray(sorted(self.k))

		
	def compute_theory_spec(self):

		self.compute_k_2D()

		self.theory_spec =  self.ps(self.ksorted)

	def compute_array_populated_box(self): 

		'''  need to make a method for if the theory psepc in an array, you interpolate to find the k's'''

	def compute_kbox(self):
		self.compute_k_2D()

		self.kbox = np.reshape(self.k,(self.row_npix,self.col_npix)) 

	
	def make_2D_universe(self):

		#this box will hold all of the values of k for each pixel
		
		self.compute_kbox()
		#here's the box that will hold the random gaussian things

		self.fft_box = np.zeros_like(self.kbox, dtype = complex)


		self.fft_box = np.zeros_like(self.kbox, dtype = complex)
		a = np.random.normal(size = (self.row_npix,self.col_npix))
		b = np.random.normal(size = (self.row_npix,self.col_npix))

		self.fft_box = a +(1j*b)

		self.stdev_box = np.zeros_like(self.fft_box)
		
		for i in range(self.row_npix):
			for j in range(self.col_npix): 
				stdev = np.sqrt(self.ps(self.kbox[i,j])/2) # [mk Mpc]

				radius = np.sqrt((j-(self.col_npix/2))**2 + (i-(self.row_npix/2))**2)
				if  3 < radius < 8:
			
					self.stdev_box[i,j] = 1
				else:
					self.fft_box[i,j] = 0 
		
		self.fft_box_stdev = self.fft_box * self.stdev_box

 
	######### IMPOSE SYMMETRY CONDITIONS SO THAT IFT IS REAL #########

		for i in range(self.row_npix):
			for j in range(self.col_npix):

				if self.row_npix % 2 == 0: # if rows even 
					mirror_i = (self.row_npix - i ) % self.row_npix
				else: # if rows odd
					mirror_i = (self.row_npix-1)-i

				if self.col_npix % 2 == 0: #if cols even
					mirror_j = (self.col_npix - j ) % self.col_npix
				else: #if cols odd
					mirror_j = (self.col_npix-1)-j

				if mirror_i == i and mirror_j == j: 
					# must be real
					self.fft_box[i,j] = np.real(self.fft_box[i,j]) #only number equal to its own complex conjugate are real numbers
				else:
				# copy complex conjugate from mirror coordinates to current coord
					self.fft_box[i,j] = np.conjugate(self.fft_box[mirror_i,mirror_j])

	########################################################################


		######### IMPOSE SYMMETRY CONDITIONS SO THAT IFT IS REAL #########

		for i in range(self.row_npix):
			for j in range(self.col_npix):

				if self.row_npix % 2 == 0: # if rows even 
					mirror_i = (self.row_npix - i ) % self.row_npix
				else: # if rows odd
					mirror_i = (self.row_npix-1)-i

				if self.col_npix % 2 == 0: #if cols even
					mirror_j = (self.col_npix - j ) % self.col_npix
				else: #if cols odd
					mirror_j = (self.col_npix-1)-j

				if mirror_i == i and mirror_j == j: 
					# must be real
					self.fft_box_stdev[i,j] = np.real(self.fft_box_stdev[i,j]) #only number equal to its own complex conjugate are real numbers
				else:
				# copy complex conjugate from mirror coordinates to current coord
					self.fft_box_stdev[i,j] = np.conjugate(self.fft_box_stdev[mirror_i,mirror_j])

	########################################################################


		self.fft_box *= np.sqrt((self.Lx*self.Ly)) # [mk Mpc^2]

		self.fft_box_stdev *= np.sqrt((self.Lx*self.Ly)) # [mk Mpc^2]
		self.u = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.fft_box * (self.delta_ky*self.delta_kx)))) #[mk]
		self.u_stdev = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.fft_box_stdev * (self.delta_ky*self.delta_kx)))) #[mk]


		self.u /= (2.*np.pi)**2
		self.u_stdev /= (2.*np.pi)**2
		
		if self.mean is not None: 

			self.mean = np.zeros((self.row_npix,self.col_npix)) + self.mean

			self.universe = (np.real(self.u))+self.mean
			

		else: 
			self.universe = (np.real(self.u))
			self.universe_stdev = (np.real(self.u_stdev))
		

		return self.universe , self.universe_stdev









		