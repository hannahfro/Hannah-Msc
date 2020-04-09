import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy import signal

class telescope(object):
    """
    Object to store the properties of the telescope, including:
    ---Telescope location
    ---Array configuration
    ---Primary beam
    """
    def __init__(self, ant_locs, latitude, channel_width, Tsys, beam_width, beam='gaussian'):
        self.ant_locs = ant_locs # (Nants,2) sized array # relative position in meters 
        self.latitude = latitude # degrees this is the central latitute of the HERA strip 
        self.latitude *= np.pi / 180. # radians
        self.channel_width = channel_width # assume Hz
        self.Tsys = Tsys # assume Kelvin
        if beam != 'gaussian':
            raise NotImplementedError()
        else:
            self.beam = beam
            self.beam_width = beam_width # FWHM in degrees
            self.beam_width *= np.pi / 180. # FWHM in radians


    def compute_2D_bls(self):
        """
        Computes the 2D ("uv plane") baselines for a local coordinate
        system when the observations started.
        """
        N_ants = self.ant_locs.shape[0]
        N_bls = int(N_ants * (N_ants - 1) / 2)
        self.bls = np.zeros((N_bls,2)) # initialize a list holding the length of all the baselines 
        k = 0 #initialize this k variable
        for i in range(N_ants):
            ant_i = self.ant_locs[i]
            for j in range(i+1,N_ants):
                ant_j = self.ant_locs[j]
                self.bls[k] = ant_i - ant_j # this subtracts each coordinate from the other [0,0]-[1,1]
                k += 1 #add k every time you identify a baseline 
                
                
    def remove_redundant_bls(self):
        """
        Removes redundant baselines. Information is preserved by
        storing a multiplicity array.
        """
        self.compute_2D_bls()
        self.bls, self.ucounts = np.unique(self.bls, axis=0, return_counts=True)
        self.N_bls= len(self.bls) # number of unique pairs of baselines
        return(self.N_bls) #this reutrns the number of unique baselines

    def compute_celestial_bls(self):
        """
        Computes a 3D distribution of baselines in a coordinate
        system fixed to the celestial sphere.
        """
        # First define the rotation matrix that gets baselines
        # to the correct latitude for the array
        #rotate about the z axis 
        self.remove_redundant_bls()
        
        co_lat = (np.pi / 2.) - self.latitude 
        cos_co_lat = np.cos(co_lat)
        sin_co_lat = np.sin(co_lat)
        lat_rot_matrix = np.array([[1., 0., 0.], #x
                                  [0., cos_co_lat, -sin_co_lat], #Y
                                  [0., sin_co_lat, cos_co_lat]]) #Z


        # Add a z coordinate to the 2D baseline vectors 
        self.bls_celestial = np.vstack((self.bls.T, np.zeros(self.N_bls)))
       
        # Rotate them! 
        self.bls_celestial = np.dot(lat_rot_matrix, self.bls_celestial)
        
        return self.bls_celestial

class observation(object):
    """
    Object that stores all the sky information 
    """

    def __init__(self, telescope, n_days, freq, delta_t, corners, effarea, beam_sigma_cutoff, resol):
        self.times = None # in days
        self.position = None# observable corners (theta, phi)
        self.bl_times = None
        self.norm = None
        self.corners = corners# corners has to be the four corner of the sky (theta, phi) [4,2]
        self.beam_sigma_cutoff = beam_sigma_cutoff
        self.telescope = telescope
        self.n_days = n_days # Number of cycles of the observation
        self.delta_t = delta_t
        self.effarea = effarea # Effective area of an antenna
        
        self.beam_width = telescope.beam_width
        self.beam = telescope.beam
        self.ant_locs = telescope.ant_locs
        self.latitude = telescope.latitude
        self.channel_width = telescope.channel_width
        self.Tsys = telescope.Tsys
        self.bls_celestial = telescope.compute_celestial_bls()
        self.ucounts = telescope.ucounts # number of unique baselines
        
        self.freq = freq #MHz
        self.resol = resol # deg
        self.resol *= np.pi/180
        
        self.Nbl = self.bls_celestial.shape[1]
        

    def observable_coordinates(self):
        """
        Find the observable coordinates given the four corners of a patch
        """
        s = self.corners.shape
        if s[0] != 4:
            raise ValueError('Four coordinates should be provided')
        elif s[1] != 2:
            raise ValueError('Input has to be RA & Dec')
            
        beam_minus, beam_plus = np.array([-1, +1])*self.beam_width*self.beam_sigma_cutoff
        min_corner = np.min(self.corners, axis=0)
        max_corner = np.max(self.corners, axis=0)
        min_obsbound = self.latitude + beam_minus #Here we're computing the size of the observing window
        max_obsbound = self.latitude + beam_plus #bigger beam_width or sigma_cutoff means a larger observing window 

        
        if max_corner[0]*(np.pi/180) < min_obsbound or min_corner[0]*(np.pi/180) > max_obsbound:
            raise ValueError('Requested Region is not observable')
        else:
            thetas = np.arange(min_obsbound, max_obsbound, self.resol) 
            phis = np.arange((min_corner[1]*(np.pi/180)), (max_corner[1]*(np.pi/180)), self.resol)
            self.position = np.concatenate(np.dstack(np.meshgrid(thetas, phis)))
            self.Npix = self.position.shape[0]
            
            
        return self.position
        

    def necessary_times(self):
        """
        Assuming Phi = 0h at phi coordinate of observable region(self.potion), 
        Figure out time when the telescope scanning and observable region overlaps
        """

        if self.position is not None:
            pass
        else:
            self.position = self.observable_coordinates() 

        time_length = (np.max(self.position[:,1]) - np.min(self.position[:,1])) / (2*np.pi)
        #fraction of roation you've completed in observing window
        self.times = np.arange(0., time_length, self.delta_t)#this is the array of zeros that is messing up the primary
        self.Nt = len(self.times)

        return self.times
    

    def rotate_bls(self):
        """
        Rotates the baselines to their correct positions given
        the observation times. Results are stored in a bl-time array.
        """

        if self.times is not None:
            pass
        else:
            self.times = self.necessary_times()
        phis = ((2. * np.pi * self.times) + np.min(self.position[:,1]))# Times are assumed to be in days
       

        # Define rotation matrices about z axis
        cos_phis = np.cos(phis)
        sin_phis = np.sin(phis)
        time_rot_matrices = np.zeros((self.times.shape[0], 3, 3))
        time_rot_matrices[:,-1,-1] = 1.
        for i in range(self.times.shape[0]):
            time_rot_matrices[i,0,0] = cos_phis[i]
            time_rot_matrices[i,0,1] = -sin_phis[i]
            time_rot_matrices[i,1,0] = sin_phis[i]
            time_rot_matrices[i,1,1] = cos_phis[i]

        # Multiply baselines by rotation matrices
        # Result is a N_times x 3 x N_bls matrix
        
        self.bl_times = np.dot(time_rot_matrices, self.bls_celestial)
        self.bl_times = np.moveaxis(self.bl_times, -1, 0) # now N_bls x N_t x 3
        self.bl_times = np.reshape(self.bl_times, (-1,3)) # now N_bl N_t x 3
        
        
    def convert_to_3d(self):
        """
        Project the patch of the sky (Equatorial coord) onto 
        the baseline coordinate (3D) for calibration
        """
        
        if self.position is not None:
            pass
        else:
            self.observable_coordinates()
        
        thetas = self.position[:,0]
        phis = self.position[:,1]
        
        
        
        transformed = np.zeros((self.Npix, 3))
        for i in range(self.Npix):
            transformed[i,0] = np.sin(thetas[i])*np.cos(phis[i])#X
            transformed[i,1] = np.sin(thetas[i])*np.sin(phis[i])#Y
            transformed[i,2] = np.cos(thetas[i])#Z
            
        
            
        return transformed
        
    
    def compute_bdotr(self):
        """
        Given an array of times and sky positions,
        computes an (N_times,N_pos) array
        containing b-dot r.
        """

        if self.bl_times is not None:#if self.bl_times
            pass
        else:
            self.rotate_bls()
        
        position3d = self.convert_to_3d() 

        # Result is a N_t N_bl x N_pix array
        self.bdotr = np.dot(self.bl_times, position3d.T) 


    def compute_beam(self):
        """
        Compute Primary Beam assuming all antenna has an identical beam with the fixed sky.
        """

        if self.times is not None:
            pass
        else:
            self.times = self.necessary_times()
        phis = ((2. * np.pi * self.times) + np.min(self.position[:,1])) #shift back to where in the sky you are. 
       

        if self.beam == 'gaussian':
            primary = np.zeros((self.Nt, self.Npix))
        
            for i in range(self.Nt): 
                primary[i] = np.exp(-((self.position[:,1]-phis[i])**2 +(self.position[:,0]-self.latitude)**2) / (self.beam_width**2))# 2D gaussian beam (N_position,2)
        
        else:
            raise NotImplementedError()
        
        ## assume primary beam is same for all baselines
        self.pbeam = np.tile(primary[np.newaxis].T,(1,self.Nbl))
        self.pbeam = np.reshape(self.pbeam, (-1,self.Npix))
        return self.pbeam
    
        
    def compute_Amat(self):
        """
        Compute A matrix
        """
        if self.times is not None:
            pass
        else:
            self.times = self.necessary_times()
        self.compute_beam()
        self.compute_bdotr()

        exponent = np.exp(-1j*self.bdotr * 2 * np.pi * (self.freq*1e6)/(3e8)) ## currently have a shape of Nt Nbl x N_pix
        pix_size = self.resol ** 2
        
        self.Amat =  exponent # self.pbeam *
       
        
        wavelength = (3e8)/(self.freq*1e6)
       
        self.noise_rms = (wavelength * self.Tsys) / (self.effarea * np.sqrt(self.n_days*self.delta_t * self.channel_width))
        self.invN = np.diag(np.repeat(self.ucounts, self.Nt)) * (1/(self.noise_rms)**2)
        # with a unit of variance, this makes an array of u counts nt times. 
        
        

    def compute_vis(self, vec):
        """
        Compute visibility from given vector
        """
        self.compute_Amat()
    
        self.Adotx = np.dot(self.Amat, vec) 
       
        noise_gauss = np.random.randn(len(self.Adotx))#add noise to Adotx
        self.Adotx = np.dot(self.Amat, vec) #+ noise_gauss 
      
        
    def compute_normalization(self):
        ## compute [A^t N^{-1} A]^{-1}
        self.compute_Amat()
        AtA = ((np.conj(self.Amat)).T).dot(self.invN).dot(self.Amat)
        
        
        #here we are setting all the non-diagonal elements of AtA to 0
        
        diagAtA = np.diag(AtA)
    
        matrix_diagAtA = np.diag(diagAtA)
    
        self.norm = la.inv(matrix_diagAtA) #take the diagonal and take the inverse of the diagonal elements only
        


    def generate_map_noise(self):
        """
        Draw Gaussian random white noise from noise rms
        Returns to normalized noise prediction
        """
        
        if self.norm is not None:
            pass
        else:
            self.compute_normalization()
                   
        white_noise = np.zeros((self.Nbl, self.Nt))

        for i in range(self.Nbl):
            count_norm= np.diag(np.repeat(self.ucounts[i], self.Nt))
            white_noise[i] = np.random.normal(0, self.noise_rms, size=self.Nt).dot(count_norm)## separate white noise for different baselines
        white_noise = white_noise.flatten()
        self.noise = np.dot(self.norm, ((np.conj(self.Amat)).T).dot(self.invN).dot(white_noise))
        
        
        

    def convolve_map(self, vec):
        
        """
        Normalized Sky Prediction
        """
        if self.norm is not None:
            pass
        else:
            self.compute_normalization()
        self.compute_vis(vec)

        self.map = np.dot(self.norm,((np.conj(self.Amat)).T).dot(self.invN).dot(self.Adotx)) #This is the map with norm (better for cosmology)
        #((np.conj(self.Amat)).T).dot(self.invN).dot(self.Adotx) # this is the map without norm (useful for point sources)
        
        return self.map 
        

       
    
    def W_matrix(self): 
        if self.norm is not None:
            pass
        else:
            self.compute_normalization()
            
        self.compute_Amat()
        
        self.W =  np.dot(self.norm,((np.conj(self.Amat)).T).dot(self.invN).dot(self.Amat))
        
        return self.W
        