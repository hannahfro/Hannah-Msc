
import numpy as np
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

		self.delta_ky = 1/np.float(self.Ly)
		self.delta_kx = 1/np.float(self.Lx)

	def compute_k_2D(self):

		#COLUMNS

		if self.col_npix%2 == 0: # checking if num samples are even 
			
			self.kx_pos = np.arange(0,1/(self.delta_x*2), (self.delta_kx))
			self.kx_neg = np.arange(-1/(self.delta_x*2),0,(self.delta_kx))
			self.kx = np.concatenate((self.kx_neg,self.kx_pos))
		else: 
			self.kx = np.arange(-1/(self.delta_x*2),1/(self.delta_x*2), (self.delta_kx))

		#ROWS	

		if self.row_npix%2 == 0: # checking if num samples are even 
			
			self.ky_pos = np.arange(0,1/(self.delta_y*2), (self.delta_ky))
			self.ky_neg = np.arange(-1/(self.delta_y*2),0,(self.delta_ky))
			self.ky = np.concatenate((self.ky_neg,self.ky_pos))
		else: 
			self.ky = np.arange(-1/(self.delta_y*2),1/(self.delta_y*2),(self.delta_ky))


		k = []

		#Careful! need to fix a y and cycle through x to get the right concentric rings in your kbox!

		for i in range(len(self.ky)): 
			for j in range(len(self.kx)):
				k.append(np.sqrt(self.kx[j]**2 + self.ky[i]**2)) 


		self.k = np.asarray(k)
		
		self.ksorted= np.asarray(sorted(self.k))

		
	def compute_theory_spec(self):

		self.compute_k_2D()

		self.theory_spec = 	self.ps(self.ksorted)

		

	def compute_array_populated_box(self): 

		'''  need to make a method for if the theory psepc in an array. 
		need to soft pspec array in increasing kbin order. then when populating fft_box, 
		you need to match populate k pixel with pspec value that has the same index as that k mode. 

		''' 
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
				a = float(np.random.normal(0,np.sqrt(self.ps(self.kbox[i,j])))) #for discrete, need to interp 
				b = float(np.random.normal(0, np.sqrt(self.ps(self.kbox[i,j])))) #for disctrete need to interp
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

		self.u = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(fft_box))) 

		
		if self.mean is not None: 

			self.mean = np.zeros((self.row_npix,self.col_npix)) + self.mean

			self.universe = (np.real(self.u))+self.mean
			

		else: 
			self.universe = (np.real(self.u)) 
		
		return self.universe

		

	# def make_3D_universe():        
	#     if dim ==3:
		
	#     #here's our space which will hold the k values for each pixel
	#         kbox = np.zeros((npix,npix,npix))

	#     #here's the box that will hold the random gaussian things

	#         fft_box = np.zeros_like(kbox, dtype = complex)
	#     # kdelta = bins[1]-bins[0]
	#     #how to i figure out what kdelta is?
	#     #we're going through the pixels and giving them each a k 
	#         for i in range(npix): 
	#             for j in range(npix): 
	#                 for h in range(npix):
	#                     kbox[i,j,h] = (np.sqrt((i-(npix/2))**2 + (j-(npix/2))**2+ (h-(npix/2))**2)) #there's a kdelta unit missing here which we got from the fftfreq

	#         for i in range(npix):
	#             for j in range(npix): 
	#                 for h in range(npix):
	#                     a = float(np.random.normal(0,np.sqrt(ps(kbox[i,j,h]))))
	#                     b = float(np.random.normal(0, np.sqrt(ps(kbox[i,j,h]))))
	#                     fft_box[i,j,h] = complex(a,b)   
		
		
	#         for i in range(npix):
	#             for j in range(npix):
	#                 for h in range(npix):
					
	#                 #if our k coordinate is located on the edge of the space, there is no -k equivalent due to even number
	#                 #of pixels. In these cases, we will make sure they are real
	#                     if i <= npix/2:
	#                         mirror_i = (npix - i ) % npix
	#                         mirror_j = (npix - j ) % npix
	#                         mirror_h = (npix - h ) % npix
	#                         if mirror_i == i and mirror_j == j and mirror_h == h: 
		
	#                     # must be real
	#                             fft_box[i,j,h] = np.real(fft_box[i,j,h])
	#                         else:
	#                     # copy complex conjugate from mirror coordinates to current coord
	#                             fft_box[i,j,h] = np.conjugate(fft_box[mirror_i,mirror_j,mirror_h])
	#         fft_box[npix//2,npix//2] = np.real(fft_box[npix//2,npix//2])
		
	#         u = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(fft_box)))
	#         universe = (np.real(u))

	# old kbox method self.kbox = np.zeros((self.row_npix,self.col_npix))

		# self.kx = np.zeros(self.row_npix)
		# self.ky = np.zeros(self.col_npix)

		# for i in range(self.row_npix): 
		# 	for j in range(self.col_npix): 
		# 		self.kx[i] = (i-(self.row_npix/2))*(self.delta_kx)
		# 		self.ky[j] = (j-(self.col_npix/2))*(self.delta_ky) 
		# 		# I think there should be a factor of 2pi here but it makes the map look really ugly
		# 		self.kbox[i,j] = (np.sqrt(self.kx[i]**2 + self.ky[j]**2))

	'''- The transistor - Derivation of black-scholes option pricing - PV cell and basic efficiency considerations - Google page rank - long term recurrent neural nets - GMR and applications (e.g. hard drives) - Classical error correction (LDPC, circulants). Either Dan or Tomo for this one. - reinforcement learning - Shor code - Proof of error threshold in toric code - arnoldi and lanczos recursions - considerations that enter into de laval rocket nozzle design - bitcoin - the shape of a wake made by a duck'''
			