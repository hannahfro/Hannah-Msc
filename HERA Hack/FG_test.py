import unittest
import nose.tools as nt
import numpy as np
import numpy.linalg as la
from scipy import signal
import HERA_hack
import FG_hera

###### NOTES ON TESTS#########

#note that the tests will fail if run on python 2. Python 2 has the 
#type output <type 'numpy.ndarray'> while python 3 has "<class 'numpy.ndarray'>"
#so if you're running in python 2 just look it over. STRONGLY RECOMMENT PYTHON 3 ppl. 

class test_tools():

    def setUp(self): #this will be run before the test functions 
        # Create a telescope and observation        
      #SETUP 

        self.freq_fid = 150

        self.n_sources = 10

        self.dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[47,2],[45,23],[56,21],[30,115],[48,52],[100,100],[0,200],[115,30],[33,31],[49,11],[21,24],[25,6],[56,9],[12,13],[16,17],[38,17],[60,14],[26,28],[6,45],[3,37],[12,55],[200,0],[145,13],[134,65],[139,163]])

        #observable corners of the sky [lat,long]
        self.acorner = np.array([[120,270],[122,280],[120,280],[122,270]])

        self.HERA = HERA_hack.telescope(self.dishes, latitude=-30, channel_width=1., Tsys=300, beam_width=3, beam = 'gaussian')

        self.obs = HERA_hack.observation(self.HERA, 100, 100, 0.01,self.acorner,1, 0.2, norm = False, pbeam = False)

        self.fg = FG_hera.foregrounds(self.obs,self.freq_fid)

    def tearDown(self): #this will be run after the test functions 
        pass

    ######### FOREGROUND CLASS TESTS ##############

    def test_synchro(self):

        synchro = self.fg.compute_synchro()
        type_synchro = type(synchro)

        nt.assert_equal("%s" %type_synchro, "<class 'numpy.ndarray'>")
        nt.assert_equal(len(synchro), self.obs.Npix)


    def test_bremsstrauhlung(self):

        ff = self.fg.compute_bremsstrauhlung()
        type_ff = type(ff)

        nt.assert_equal("%s" %type_ff, "<class 'numpy.ndarray'>")
        nt.assert_equal(len(ff), self.obs.Npix)

    def test_unres_point_sources(self):


        unres = self.fg.unres_point_sources(self.n_sources)
        type_unres = type(unres)

        nt.assert_equal("%s" %type_unres, "<class 'numpy.ndarray'>")

        nt.assert_equal(len(unres), self.obs.Npix)

    def test_diffuse_fg(self): 

        diffuse = self.fg.diffuse_fg(self.n_sources)
        type_diffuse = type(diffuse)

        nt.assert_equal("%s" %type_diffuse, "<class 'numpy.ndarray'>")
        nt.assert_equal(len(diffuse), (self.obs.Npix+self.fg.num_bright_guys)






