import unittest
import nose.tools as nt
import numpy as np
import hera_3

class test_tools():

    def setUp(self):
        # Create a telescope and observation
        self.dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[134,65],[139,163]])
        self.acorner = np.array([[-30,250],[-32,285],[-30,285],[-32,250]])
        self.HERA = hera_3.telescope(self.dishes, latitude=-30, channel_width=1., Tsys=300, beam_width=10, beam = 'gaussian')
        self.obs = hera_3.observation(self.HERA, 100, 100, 0.1,self.acorner, 93, 1, 0.3)
   
    def tearDown(self):
        pass

    def test_bls_celestial(self):

        # Test the rotate_bls to see if it
        # returns an array of the right size
        self.dishes = np.array([[0,0],[0,55],[30,30],[0,60],[2,55],[134,65],[139,163]])
        self.acorner = np.array([[-30,250],[-32,285],[-30,285],[-32,250]])
        self.HERA = hera_3.telescope(self.dishes, latitude=-30, channel_width=1., Tsys=300, beam_width=10, beam = 'gaussian')
        self.obs = hera_3.observation(self.HERA, 100, 100, 0.1,self.acorner, 93, 1, 0.3)
   
        output_arr = self.obs.bl_times
        print(self.obs.bl_times)
        assert False
        nt.assert_equal(output_arr.shape, (self.obs.Nbl*self.obs.Nt, 3))

    #     # Test the square function to make sure that squaring
    #     # a matrix containing only 1s and 0s returns the same thing
    #     test_arr = np.diag(np.ones(self.n))
    #     output_arr = tools.square(test_arr)
    #     for i in range(self.n):
    #         for j in range(self.n):
    #             nt.assert_equal(output_arr[i,j], test_arr[i,j])

    #     # Test the square function for a known input/output
    #     test_arr = np.array([1, 2])
    #     output_arr = tools.square(test_arr)
    #     nt.assert_equal(output_arr[0], 1)
    #     nt.assert_equal(output_arr[1], 4)


    #def test_self_position_size(self):
    #     # See if the get_pi function really returns pi
    #     alleged_pi = tools.get_pi()
    #     nt.assert_almost_equal(alleged_pi, 3.141592653589)

    # def test_picky(self):
    #     # Make sure that tools.picky raises an error if the
    #     # wrong type is inputted
    #     nt.assert_raises(TypeError, tools.picky, 'hey')
