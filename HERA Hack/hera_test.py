import unittest
import nose.tools as nt
import numpy as np
import numpy.linalg as la
from scipy import signal
import HERA_hack

#command to run in terminal: nosetests "HERA Hack"/hera_test.py

#Types of things to test: 
    #1. Array shapes 
    #2. Specific values of things 
    #3. Correct data type 
    #4. wrong type of data input nt.assert_raises(TypeError, tools.picky, 'hey')



class test_tools():

    def setUp(self): #this will be run before the test functions 
        # Create a telescope and observation        
        self.dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[134,65],[139,163]])
        self.acorner = np.array([[120,270],[122,280],[120,280],[122,270]])

        self.HERA = HERA_hack.telescope(self.dishes, latitude= -30, channel_width=1., Tsys=300, beam_width= 3 ,beam = 'gaussian')        
        self.obs = HERA_hack.observation(self.HERA, 100., 100., 0.01,self.acorner, 1, 0.3, norm = True, pbeam = False)

        n_pix = self.obs.observable_coordinates().shape[0]
        imp = signal.unit_impulse(n_pix, 'mid')

        self.obs.convolve_map(imp) # you have to run this in the setup so that all of the NONE variables get written!
        self.obs.generate_map_noise()

    def tearDown(self): #this will be run after the test functions 
        pass

    ######### TELESCOPE CLASS TESTS ##############

    def test_compute_2D_bls(self):

        nt.assert_equal(self.HERA.bls.shape,(self.obs.Nbl,2))

    def test_remove_redundant_bls(self):

        nt.assert_equal(len(self.HERA.bls),self.HERA.N_bls)

    def test_compute_celestial_bls(self):

        nt.assert_equal(self.HERA.bls_celestial.shape,(3,self.HERA.N_bls))

    ############ OBSERVATION CLASS TESTS ############
    def test_observable_coords(self): 

        #Check that the min and max thetas are correct
        nt.assert_almost_equal(self.obs.position[0,0], (np.pi/2.) - (self.obs.latitude + (self.obs.beam_width*self.obs.beam_sigma_cutoff)), places=2)
        nt.assert_almost_equal(self.obs.position[self.obs.Npix-1,0], (np.pi/2.) - (self.obs.latitude - (self.obs.beam_width*self.obs.beam_sigma_cutoff)), places=1)

        #check that the min and max phis are correct
        nt.assert_almost_equal(np.round(self.obs.position[0,1]*(180/np.pi)), 270, places = 2 )
        nt.assert_almost_equal(np.round(self.obs.position[self.obs.Npix-1,1]*(180/np.pi)), 280, places = 2 )

        #check the shape of the array 
        nt.assert_equal(self.obs.position.shape, (self.obs.Npix,2))

    def test_necessary_times(self): 

        #check that the spacing in times in correct
        nt.assert_almost_equal(self.obs.times[1]-self.obs.times[0], self.obs.delta_t)

        nt.assert_equal(np.dtype(self.obs.times[0]),float)

        phi_min = self.obs.position[0,1]

        phi_max = self.obs.position[self.obs.Npix-1,1]

        #print(np.dtype(max(self.obs.times)))

        nt.assert_almost_equal(max(self.obs.times)*2.*np.pi, np.abs(phi_max-phi_min), places = 1)

    def test_rotate_bls(self):
        # Test the rotate_bls to see if it
        # returns an array of the right size  
        output_arr = self.obs.bl_times
        nt.assert_equal(output_arr.shape, (self.obs.Nbl*self.obs.Nt, 3))

        phis = 2*np.pi*self.obs.times
        nt.assert_equal(phis[0],0)


    def test_convert_to_3d(self):

        transformed = self.obs.convert_to_3d()

        nt.assert_equal(transformed.shape, (self.obs.Npix,3))


    def test_bdotr(self):

        bdotr = self.obs.bdotr
        nt.assert_equal(bdotr.shape, (self.obs.Nt*self.obs.Nbl,self.obs.Npix))

    def test_compute_beam(self):

        phis = (2. * np.pi * self.obs.times) + self.obs.position[0,1] 


        every_226_phi = self.obs.observable_coordinates()[::226,1]


        for i in range(len(self.obs.times)):
            nt.assert_almost_equal(phis[i],every_226_phi[i], places = 1)

        nt.assert_equal(self.obs.pbeam.shape, (self.obs.Nt*self.obs.Nbl, self.obs.Npix))

        

    def test_compute_Amat(self):

        nt.assert_equal(self.obs.Amat.shape,(self.obs.Nt*self.obs.Nbl, self.obs.Npix))

        nt.assert_equal(np.dtype(self.obs.Amat[0,0]),complex)

        #maybe do a single baseline analytic solution here 
    
        nt.assert_equal(self.obs.invN.shape, (self.obs.Nt*self.obs.Nbl,self.obs.Nt*self.obs.Nbl))

    def test_compute_vis(self): 

        nt.assert_equal(self.obs.Adotx.shape,(self.obs.Nt*self.obs.Nbl,) )
        nt.assert_equal(np.dtype(self.obs.Adotx[0]),complex)


    def test_compute_normalization(self):
        
        for i in range(self.obs.norm.shape[0]-1): 
            for j in range(self.obs.norm.shape[0]-1):
                if i != j:
                    nt.assert_equal(self.obs.norm[i,j],0)
                else: 
                    nt.assert_equal(np.dtype(self.obs.norm[i,j]),complex)

    def test_generate_map_noise(self):

        nt.assert_equal(np.dtype(self.obs.my_noise[0]),complex)

        nt.assert_equal(self.obs.noise.shape, (self.obs.Npix,))

    def test_convolve_map(self):

        nt.assert_equal(np.dtype(self.obs.map[0]),complex)

        nt.assert_equal(self.obs.map.shape, (self.obs.Npix,))  

    
