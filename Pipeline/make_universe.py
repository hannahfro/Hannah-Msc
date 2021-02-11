
import numpy as np
import matplotlib.pyplot as plt
import math


class universe(object):
	"""docstring for ClassName"""
	def __init__(self, ps, row_npix,col_npix, Ly,Lx, mean):

		self.ps = ps #1d pspec
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

		self.ky *= 2*np.pi
		self.kx *= 2*np.pi


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

		fft_box = np.zeros_like(self.kbox, dtype = complex)


		
		for i in range(self.row_npix):
			for j in range(self.col_npix): 
				stdev = np.sqrt(self.ps(self.kbox[i,j])/2) # mk Mpc^2
				a = float(np.random.normal(0, stdev)) #for discrete, need to interp 
				b = float(np.random.normal(0, stdev)) #for disctrete need to interp
				fft_box[i,j] = complex(a,b) 
		
		for i in range(self.row_npix):
			for j in range(self.col_npix):

				#if our k coordinate is located on the edge of the space, there is no -k equivalent due to even number
				#of pixels. In these cases, we will make sure they are real
				if i <= self.row_npix/2:
					mirror_i = (self.row_npix - i ) % self.row_npix
					mirror_j = (self.col_npix - j ) % self.col_npix

					if mirror_i == i and mirror_j == j: 

					# must be real
						fft_box[i,j] = np.real(fft_box[i,j])
					else:
					# copy complex conjugate from mirror coordinates to current coord
						fft_box[i,j] = np.conjugate(fft_box[mirror_i,mirror_j])

		fft_box[self.row_npix//2,self.col_npix//2] = np.real(fft_box[self.row_npix//2,self.col_npix//2])

		fft_box *= np.sqrt((self.Lx*self.Ly))

		self.u = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(fft_box * (self.delta_ky*self.delta_kx)))) #[mk]

		
		if self.mean is not None: 

			self.mean = np.zeros((self.row_npix,self.col_npix)) + self.mean

			self.universe = (np.real(self.u))+self.mean
			

		else: 
			self.universe = (np.real(self.u))

		return self.universe






		