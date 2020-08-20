## Observations Occur between 190-315 GHz (z=5-9)
## This bandwidth coverage is apparently crucial for foreground removal
## apparently foregrounds are mostly due to CO lines at intermediate redshifts, spaced by 115.27 GHZ/(1+z)
## these can apparently be refected by anticorrelation ---> where is there CO? 
## there are also other much fainter fine structure lines... what are they?/
## Ideal obs has 1 arcmin reso


## SOUCRES OF CONTAMINATION

#1. Far-IR background 
#2. OI line 
#3. two NII lines
#4. a handful (uh oh) of CO rotational transition lines 


################ THIS IS PSEUDOCODE RN ####################
import numpy as np 
import scipy.constants as sc 
import pandas as pd
from scipy.interpolate import interp1d
from astropy.cosmology import WMAP9 as cosmo



class CII_fg(object):

	"""
	This module generates foregrounds for CII observations for a single channel at a time
	(this is under the assumption that the computation for all freqs will be parallelized).

	Parameters
	----------

	freq_min: float 
		This is the lower edge of your frequency bin in (GHz)

	freq_max: float 
		This is the upper edge of your frequency bin in (GHz)

	Recall that you need to define your bins in the following way:

		freq_bins = np.arange(190,315,3)

	Where 190 and 315 correspone to the upper and lower bound of the observing band 
	and 3 correspons to the channel width all in units of GHz. 

	"""

	def __init__(self, freq_min, freq_max):

		self.freq_min = freq_min
		self.freq_max = freq_max


	def CO_data(self):

		co_rest = {'line':['CO 1-0','CO 2-1','CO 3-2','CO 4-3','CO 5-4','CO 6-5','CO 7-6','CO 8-7'],'rest freq (GHZ)': [115.27,230.54,345.80,416.04,576.27,691.47,806.65,921.80],}
		self.df_co = pd.DataFrame(data=co_rest)

		self.schecter_params = [['CO 1-0',[0,1,2,4,6],[-1.36,-1.49,-1.52,-1.71,-1.94],[6.97,7.25,7.30,7.26,6.99],[-2.85,-2.73,-2.63,-2.94,-3.46]],['CO 2-1',[0,1,2,4,6],[-1.35,-1.47,-1.52,-1.75,-2.00],[7.54,7.84,7.92,7.89,7.62],[-2.85,-2.72,-2.66,-3.00,-3.56]],['CO 3-2',[0,1,2,4,6],[-1.29,-1.47,-1.53,-1.76,-2.00],[7.83,8.23,8.36,8.26,7.95],[-2.81,-2.79,-2.78,-3.11,-3.60]],['CO 4-3',[0,1,2,4,6],[-1.29,-1.45,-1.51,-1.80,-2.03],[8.16,8.50,8.64,8.70,8.23],[-2.93,-2.84,-2.85,-3.45,-3.78]],['CO 5-4',[0,1,2,4,6],[-1.20,-1.47,-1.45,-1.76,-1.95],[8.37,8.80,8.74,8.73,8.30],[-2.94,-3.03,-2.80,-3.34,-3.67]],['CO 6-5',[0,1,2,4,6],[-1.15,-1.41,-1.43,-1.73,-1.93],[8.38,8.74,8.77,8.84,8.38],[-2.92,-2.92,-2.80,-3.40,-3.72]]]

		self.ll_star = np.logspace(-3,1,100) # log space bins
		self.delta_l = self.ll_star[1]-self.ll_star[0]

	def find_lines_in_bin(self):

		self.CO_data()

		def find_z(f_emit,f_obs):
				return(f_emit/f_obs)-1 

		data = [] #this list holds the names of the lines that contaminate a redshift bin and their emitted redshift

		for i in range(self.df_co.shape[0]):
			z_emit_min = find_z(self.df_co['rest freq (GHZ)'][i],self.freq_max)
			z_emit_max = find_z(self.df_co['rest freq (GHZ)'][i],self.freq_min)
			delta_z = z_emit_max-z_emit_min
			z_mid = (z_emit_min + (delta_z/2))
			if z_emit_min >= 0:
				#print(df_co['line'][i])
				for j in range(len(self.schecter_params)):
					if self.df_co['line'][i] == self.schecter_params[j][0]:
						# pull our the schecter params
						zs = self.schecter_params[j][1]
						alphas = self.schecter_params[j][2]
						Ls = self.schecter_params[j][3]
						phis = self.schecter_params[j][4]
						
						#interpolate to find a function
						alpha_interp = interp1d(zs,alphas, kind = 'cubic')
						L_interp = interp1d(zs,Ls, kind = 'cubic')
						phi_interp = interp1d(zs,phis, kind = 'cubic')
						#compute function at z_mid and append 
						data.append([self.df_co['line'][i],z_mid,delta_z,float(alpha_interp(z_mid)),float(L_interp(z_mid)),float(phi_interp(z_mid))])
						break
						
		# now i will go and may maps, line by line, and then add them together to make the frequency map
		
		self.data = np.asarray(data)
		


	def compute_fg(self,npix):


		self.find_lines_in_bin()


		def schecter(ll_star,log_phi,alpha):
			phi_star =10**(log_phi)
			phi = phi_star*(ll_star)**(alpha) * np.exp(-ll_star)
			return phi



		def comov_vol(z,delta_z,omega_pix):
			return (omega_pix*((cosmo.angular_diameter_distance(z).value)**2)*(sc.c/cosmo.H(z).value)*delta_z) ###how to deal with delta z...



		nlines = self.data.shape[0]

		self.L_tot = np.zeros(npix)

		ave_Ns = np.zeros(len(self.ll_star))

		for h in range(nlines): #here you find the average Ns for each lum bin for each line

			#ave_Ns = np.zeros(len(self.ll_star))
			
			
			for i in range(len(self.ll_star)):
				central_l = self.ll_star[i]+(self.delta_l/2)
				ave_Ns[i] = schecter(central_l,float(self.data[h,5]),float(self.data[h,3]))*(self.delta_l)*comov_vol(float(self.data[h,1]),float(self.data[h,2]),(0.44**2))
   
		for j in range(npix): # this is when you populate all the pixels with sources. 
			for h in range(nlines):
				for i in range(len(self.ll_star)): #pick a luminosity pls
					numsources =  np.random.poisson(ave_Ns[i]) #find the number of sources in that (L,z)
					central_l = (self.ll_star[i])+(self.delta_l/2) # central llstar of that bin
					L_s = 10**(float(self.data[h,4]))
					self.L_tot[j] += (central_l*L_s) * numsources #luminosity of one (L,z) bin

		return self.L_tot 

# #OLD DATA :
# d = {'z': [0,1,1,2], 'obs freq': [230.54,230.52,288.13,230.49],'alpha':[-1.35,-1.45,-1.47,-1.43],'log_Ls':[7.54,8.50,8.80,8.77],'log_phi':[-2.85,-2.84,-3.03,-2.80]}
# df = pd.DataFrame(data=d)
