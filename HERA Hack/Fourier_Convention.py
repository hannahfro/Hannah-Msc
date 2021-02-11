import numpy as np 


class Fourier_Convention(object):

	''' This class contains various methods for converting the numpy.fft.fft outputs to any fourier convention of your choosing!
	For now this only deals with the forward transform and its frequencies but 
	hoepfully it will also work for inverse transforms, convolution theorem, fourier inversions, shifting, etc... very soon!


	Th notation if the following: 

	f~(k) = 1/sqrt(m) ∫ f(x) e^(sqikx) dx 

	s = [-1, +1] forawrd sign
	q = ['2pi', 1] exponent 
	m = ['2pi', 1] overall_factor

	Here I label the convention as follows F_sqm
	
	By default, numpy uses the convention 

	F_minus_2pi_1 --> f~(k) =  ∫ f(x) e^(-2pi ikx) dx 

	The notation used here was found on this useful site! https://www.johndcook.com/blog/fourier-theorems/ 

	'''
	
	def __init__(self, forward_sign , exponent , over all_factor):

		self.forward_sign = forward_sign
		self.exponent = exponent 
		self.overall_factor = overall_factor


	def F_minus11(self,FFT, fft_freq):

		''' This is the cosmology convention we usually use

		f~ (k)= ∫ f(x) e^(-ikx) dx '''

		self.fft_conv = FFT

		self.fft_freq_conv = fft_freq/(2*np.pi)

	def F_minus_1_2pi(self,FFT, fft_freq):

		''' f~ (k)= 1/root(2pi) ∫ f(x) e^(-ikx) dx '''
		
		self.fft_conv = FFT * (1/(np.sqrt(2*np.pi)))

		self.fft_freq_conv = fft_freq/(2*np.pi)


	def F_minus_2pi_2pi(self,FFT, fft_freq):

		''' f~ (k)= 1/root(2pi) ∫ f(x) e^(-2pi ikx) dx '''
		
		self.fft_conv = FFT * (1/(np.sqrt(2*np.pi)))

		self.fft_freq_conv = fft_freq

	def F_plus_2pi_1(self,FFT, fft_freq):

		''' f~ (k)=  ∫ f(x) e^(2pi ikx) dx '''
		
		self.fft_conv = FFT

		self.fft_freq_conv = - fft_freq


	def F_plus_1_1(self,FFT, fft_freq):

		''' f~ (k)=  ∫ f(x) e^(ikx) dx '''
		
		self.fft_conv = FFT

		self.fft_freq_conv = - (fft_freq/(2*np.pi))

	def F_plus_1_2pi(self,FFT, fft_freq):

		''' f~ (k)= 1/root(2pi) ∫ f(x) e^(ikx) dx '''
		
		self.fft_conv = FFT * (1/(np.sqrt(2*np.pi)))

		self.fft_freq_conv = - (fft_freq/(2*np.pi))

	def F_plus_2pi_2pi(self,FFT, fft_freq):

		''' f~ (k)= 1/root(2pi) ∫ f(x) e^(2pi ikx) dx '''
		
		self.fft_conv = FFT * (1/(np.sqrt(2*np.pi)))

		self.fft_freq_conv = - fft_freq


	def convert_my_fft(self,FFT, fft_freq):

		if self.forward_sign == -1 and self.exponent == '2pi' and self.overall_factor == 1:

			raise ValueError("numpy already uses this convention! No conversion necessary.")

		elif self.forward_sign == -1 and self.exponent == 1 and self.overall_factor == 1:

			self.F_minus11(FFT, fft_freq)

			return self.fft_freq_conv , self.fft_conv

		elif self.forward_sign == -1 and self.exponent == 1 and self.overall_factor == '2pi':

			self.F_minus_1_2pi(FFT, fft_freq)

			return self.fft_freq_conv , self.fft_conv

		elif self.forward_sign == -1 and self.exponent == '2pi' and self.overall_factor == '2pi':

			self.F_minus_2pi_2pi(FFT, fft_freq)

			return self.fft_freq_conv , self.fft_conv

		elif self.forward_sign == 1 and self.exponent == '2pi' and self.overall_factor == 1:

			self.F_plus_2pi_1(FFT, fft_freq)

			return self.fft_freq_conv , self.fft_conv

		elif self.forward_sign == 1 and self.exponent == 1 and self.overall_factor == 1:

			self.F_plus_1_1(FFT, fft_freq)

			return self.fft_freq_conv , self.fft_conv

		elif self.forward_sign == 1 and self.exponent == 1 and self.overall_factor == '2pi':

			self.F_plus_1_2pi(FFT, fft_freq)

			return self.fft_freq_conv , self.fft_conv

		elif self.forward_sign == 1 and self.exponent == '2pi' and self.overall_factor == '2pi':

			self.F_plus_2pi_2pi(FFT, fft_freq)

			return self.fft_freq_conv , self.fft_conv


		else:
			raise ValueError('That is not a convention defined here!')





