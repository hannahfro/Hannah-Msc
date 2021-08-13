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
from scipy.interpolate import interp1d, barycentric_interpolate
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt



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
		
		freq_bins = np.arange(185,310,0.4) 
		test--> len(freq_bins) = 313 since the number of spectral channels is 312

	Where 185 and 310 correspone to the upper and lower bound of the observing band 
	and 0.4 correspons to the frequency resolution all in units of GHz. This means that ccat-p has 
	312 spectral channels. 

	This code computes the foregrounds for ONE frequnecy channel at a time. The reason why I made it this was is because
	it when I go to parallelize the code, I will likely have each node compute diff freq bins since they are independent of one another. 

	All of the Schechter parameters and rest frequencies of the CO lines were taken from Popping et al. 2016 

	Inspiration for this algorithm was takedn from Cheng, Chang, and Bock 2020 

	"""

	def __init__(self, freq_min, freq_max, omega_pix,npix):

		self.freq_min = freq_min #GHz
		self.freq_max = freq_max #GHz 
		self.delta_freq = (self.freq_max - self.freq_min) #GHz
		self.npix = npix
		self.omega_pix = omega_pix # in steradians

		co_rest = {'line':['CO 1-0','CO 2-1','CO 3-2','CO 4-3','CO 5-4','CO 6-5'],'rest freq (GHZ)': [115.27,230.54,345.80,416.04,576.27,691.47]}#,'CO 7-6','CO 8-7',806.65,921.80]}
		self.df_co = pd.DataFrame(data=co_rest)

		'''This data is organizes in an nlines x nparams x nredshifts array in the following way
		
		Line name| redshifts | alpha| log(L*) | log(phi*) | 

		'''

		self.schechter_params = [[['CO 1-0'],[0,1,2,4,6],[-1.36,-1.49,-1.52,-1.71,-1.94],[6.97,7.25,7.30,7.26,6.99],[-2.85,-2.73,-2.63,-2.94,-3.46]],[['CO 2-1'],[0,1,2,4,6],[-1.35,-1.47,-1.52,-1.75,-2.00],[7.54,7.84,7.92,7.89,7.62],[-2.85,-2.72,-2.66,-3.00,-3.56]],[['CO 3-2'],[0,1,2,4,6],[-1.29,-1.47,-1.53,-1.76,-2.00],[7.83,8.23,8.36,8.26,7.95],[-2.81,-2.79,-2.78,-3.11,-3.60]],[['CO 4-3'],[0,1,2,4,6],[-1.29,-1.45,-1.51,-1.80,-2.03],[8.16,8.50,8.64,8.70,8.23],[-2.93,-2.84,-2.85,-3.45,-3.78]],[['CO 5-4'],[0,1,2,4,6],[-1.20,-1.47,-1.45,-1.76,-1.95],[8.37,8.80,8.74,8.73,8.30],[-2.94,-3.03,-2.80,-3.34,-3.67]],[['CO 6-5'],[0,1,2,4,6],[-1.15,-1.41,-1.43,-1.73,-1.93],[8.38,8.74,8.77,8.84,8.38],[-2.92,-2.92,-2.80,-3.40,-3.72]]]

		self.v_0 = self.freq_min + (self.delta_freq/2) #in GHZ

		for i in range(6):
			self.schechter_params[i][3]  = self.schechter_params[i][3] + np.log10((10**((-26))*(self.v_0 * (10**(9))))/(3*(10**5))) #change units CHECK THIS AGAIN!!! should be jk km/s* 10^(-26)*nu_0[Hz]/3*10^5 [km/s]

		self.log_ll_star = np.linspace(-3,1,100) # L/L* bins in log space!!! 

		self.delta_log_llstar = self.log_ll_star[1]-self.log_ll_star[0]

	def find_lines_in_bin(self):

		''' this method picks out the lines that contaminate each frequency bin. The schecter parameters at the precise frequency are found by interpolating the schechter_params for each line'''

		def find_z(f_emit,f_obs):
				return(f_emit/f_obs)-1 

		data = [] #this list holds the names of the lines that contaminate a redshift bin and their emitted redshift


		for i in range(self.df_co.shape[0]):
			z_emit_min = find_z(self.df_co['rest freq (GHZ)'][i],self.freq_max)
			z_emit_max = find_z(self.df_co['rest freq (GHZ)'][i],self.freq_min)
			delta_z = np.abs(z_emit_max-z_emit_min)
			z_mid = (z_emit_min + (delta_z/2))

			if z_emit_min >= 0:
				# pull our the schecter params
				zs = self.schechter_params[i][1]
				alphas = self.schechter_params[i][2]
				Ls = self.schechter_params[i][3]
				phis = self.schechter_params[i][4]
				
				#interpolate to find a function
				alpha_interp = interp1d(zs,alphas, kind = 'cubic')
				L_interp = interp1d(zs,Ls, kind = 'cubic')
				phi_interp = interp1d(zs,phis, kind = 'cubic')
				#compute function at z_mid and append 
		
				data.append([self.df_co['line'][i],z_mid,delta_z,float(alpha_interp(z_mid)),float(L_interp(z_mid)),float(phi_interp(z_mid))])
				
						
		# now i will go and make maps, line by line, and then add them together to make the frequency map

		##DATA LABELS

			# data[h,0] = line name
			# data[h,1] = zmid
			# data[h,2] = delta z
			# data[h,3] = alpha @ zmid
			# data[h,4] = log(L*) @ zmid
			# data[h,5] = log(phi) @ mid
		
		self.data = np.asarray(data)

	def compute_fg(self):


		self.find_lines_in_bin()


		#this function is actualy phi(l)dL i.e. the number of sources in the luminosity bin centred on L
		def number_density(log_ll_star, log_phi, alpha):
			phi_star = 10**(log_phi)
			phi = np.log(10)* phi_star * np.exp(-(10**(log_ll_star)))* ((10**log_ll_star)**(alpha + 1)) * self.delta_log_llstar 
			return phi 
		
		def comov_vol(z,delta_z,omega_pix):
			return (omega_pix*((cosmo.angular_diameter_distance(z).value)**2)*(sc.c/cosmo.H(0).value)*(1/np.sqrt((0.31*((1+z)**3))+ (0.69)))*((1+z)**2)*delta_z) ###how to deal with delta z...

		nlines = self.data.shape[0]

		

		self.L_tot = np.zeros((nlines,self.npix))

		self.ave_Ns = np.zeros((nlines,len(self.log_ll_star)))


		for h in range(nlines): #here you find the average Ns for each lum bin for each line
			for i in range(len(self.log_ll_star)):
				central_l = self.log_ll_star[i]+(self.delta_log_llstar/2) # check math here 
				self.ave_Ns[h,i] = number_density(central_l,float(self.data[h,5]),float(self.data[h,3])) * (cosmo.differential_comoving_volume(float(self.data[h,1])).value*self.omega_pix)#comov_vol(float(self.data[h,1]),float(self.data[h,2]),self.omega_pix) # this should not be negative.... uh oh 

	

   		# here we are populating every pixel with every line in every luminosity bin.
		
		for h in range(nlines):
			for j in range(self.npix): # this is when you populate all the pixels with sources. 
				for i in range(len(self.log_ll_star)): #pick a luminosity pls
					numsources =  np.random.poisson(self.ave_Ns[h,i]) #find the number of sources in that (L,z)
					# central_l = (self.ll_star[i])+(self.delta_l/2) # central llstar of that bin
					# L_s = 10**(float(self.data[h,4]))
					self.L_tot[h,j] += (10**(float(self.log_ll_star[i])+float(self.data[h,4]))) * float(numsources)#luminosity of one (L,z) bin here taking L to be (L/L*)(L*)


		return self.L_tot 

	def intensity(self):

		""" this method is for converting the unit from W m^-2 Mpc^2 to Jy/sr """

		self.compute_fg()

		nlines = self.data.shape[0]


		self.I = np.zeros((nlines, self.npix))

		for i in range(self.L_tot.shape[0]): 
			delta_nu = self.delta_freq * (10**9)
			factor = 4*np.pi*(cosmo.luminosity_distance(float(self.data[i,1])).value**2)*delta_nu*self.omega_pix #[Mpc^2 Hz sr]
			self.I[i] = self.L_tot[i]* (1/(factor))* (10**(26)) #[10^(-26) W m^-2 hz^-1 sr^-1 = 1 Jy/sr ] CONVERT TO JY/SR PROPERLY 


		self.T = (self.I * (sc.c**2))/(2*(self.v_0*(10**(9)))**2 * sc.k) #Also could get intensity in K
	



#################### PLOTTING FOR A_mat #####################

# freq_bins = np.arange(200,350,1)

# A_mat = np.zeros((6,len(freq_bins)))

# redshift = np.zeros((6,len(freq_bins)))

# line = 5




# for j in range(A_mat.shape[0]):
#     for i in range(len(freq_bins)-1):
#         f_obs_min = freq_bins[i]
#         f_obs_max = freq_bins[i+1]

#         z_emit_max = find_z(df_co['rest freq (GHZ)'][j],f_obs_min)
#         z_emit_min = find_z(df_co['rest freq (GHZ)'][j],f_obs_max)

#         delta_z = z_emit_max-z_emit_min
#         z_mid = (z_emit_min + (delta_z/2))
#         print(z_mid)

#         if z_emit_min >= 0:
#             # pull our the schecter params
#             zs = schecter_params[j][1]
#             alphas = schecter_params[j][2]
#             Ls = schecter_params[j][3]
#             phis = schecter_params[j][4]

#             #interpolate to find a function
#             alpha_interp = interp1d(zs,alphas, kind = 'cubic',fill_value=(x[0],x[-1]))
#             L_interp = interp1d(zs,Ls, kind = 'cubic',fill_value=(x[0],x[-1]))
#             phi_interp = interp1d(zs,phis, kind = 'cubic',fill_value=(x[0],x[-1]))
#             #compute function at z_mid and append

#             redshift[j,i] = z_mid
#             A_mat[j,i] = float((10**L_interp(z_mid)))*(1/(4*np.pi*((cosmo.luminosity_distance(z_mid).value)**2)*(0.5*10**9)))
#         else:
#             A_mat[j,i] = 0
#             continue


# cII_rest = 1901.03 

# cII_intensity = np.zeros(len(freq_bins))
# cII_z = np.zeros(len(freq_bins))


# cII_schecter_params = [['CII',[0,1,2,3,4,6],[-1.25,-1.43,-1.52,-1.41,-1.53,-1.77],[7.47,7.66,7.81,7.80,7.85,7.80],[-2.33,-2.15,-2.20,-2.12,-2.37,-2.95]]]





# for i in range(len(freq_bins)-1):
#     f_obs_min = freq_bins[i]
#     f_obs_max = freq_bins[i+1]

#     z_emit_max = find_z(cII_rest,f_obs_min)
#     z_emit_min = find_z(cII_rest,f_obs_max)

#     delta_z = z_emit_max-z_emit_min
#     z_mid = (z_emit_min + (delta_z/2))
#     print(z_mid)
    

#     Ls = cII_schecter_params[0][3]
#     zs = cII_schecter_params[0][1]
#     L_interp = interp1d(zs,Ls, kind = 'cubic',fill_value = "extrapolate")
#     cII_intensity[i] = float((10**L_interp(z_mid)))*(1/(4*np.pi*((cosmo.luminosity_distance(z_mid).value)**2)*(0.5*10**9)))
#     cII_z[i] = z_mid



# fig, ax = plt.subplots(1, figsize = (20,7))
# for i in (1,2,3,4,5):
#     ax.scatter(redshift[i], freq_bins, c=np.log(A_mat[i]), label = df_co['line'][i])#, cmap='Greens')
# im = ax.scatter(cII_z,freq_bins, c = np.log(cII_intensity), label = 'CII')
# ax.set_xlabel('z Emit',fontsize = 24)
# ax.set_ylabel('Observed Freq (GHz)',fontsize = 24)
# ax.grid(color='gray', linestyle='--', linewidth=0.5)

# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('log(I [some unit i need to discern])', fontsize = 24)
# ax.legend()
