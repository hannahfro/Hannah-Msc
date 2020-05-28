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

        self.n_sources = 10
        self.freq_fid = 150

        self.foregrounds = FG_hera.foregrounds(self.obs,self.freq_fid)

    def tearDown(self): #this will be run after the test functions 
        pass

    ######### FOREGROUND CLASS TESTS ##############

    def test_synchro(self):


    def test_bremsstrauhlung(self):

    def test_unres_point_sources(self):