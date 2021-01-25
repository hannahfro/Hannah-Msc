import numpy as np


class Power_Spectrum(object):
    """docstring for Power_Spectrum
"""
    def __init__(self, data, Lx, Ly, nbins, log):
        
        self.data = data #box/cube data set
        self.Lx = Lx # physical length scale of data.shape[0] side
        self.Ly = Ly # physical length scale of data.shape[1] side
        self.nbins = nbins # number of bins, 
        self.log = log #indicate whether to use log spacing
        #self.dim = len(self.data.shape)


        #these two lines give you the physical dimensions of a pixel (inverse of sampling ratealong each axis)
        self.delta_y = (self.Ly/self.data.shape[0]) # size of y pixel
        self.delta_x = (self.Lx/self.data.shape[1]) # size of x pixel

        self.delta_ky = 1/self.Ly
        self.delta_kx = 1/self.Lx


        #ADD if len shape is 3 then add delta z

    def cosmo_FFT3(self): 
        '''
        The cosmology convention is to use FFT with no 2pi, while np.FFT 
        uses 2pi in the exponent. To convert frmo exponent to none you divide your k's -->k/2pi
        '''
        
        mean = np.mean(self.data)

        mean_arr = np.repeat(mean,len(self.data))

        self.data = self.data - mean_arr

        fft_data = np.fft.fftn(self.data)
        ps_data = np.abs(np.fft.fftshift(fft_data))**2 #this has variance equal to p(k)
        
        self.npix = self.data.shape[0]*self.data.shape[1]*self.data.shape[2]
        kx = np.fft.fftfreq(npix,self.L)/(2*np.pi)
        ky = np.fft.fftfreq(npix,self.L)/(2*np.pi)
        kz = np.fft.fftfreq(npix,self.L)/(2*np.pi)
        self.kdeltax = kx[1]-kx[0]
        self.kdeltay = ky[1]-ky[0]
        self.kdeltaz = kz[1]-kz[0]


        k = []

        for i in range(len(kx)): 
            for j in range(len(ky)):
                for h in range(len(kz)):
                    k.append(np.sqrt(kx[i]**2 + ky[j]**2 + kz[h]**2))

        self.k = np.asarray(k)

    def cosmo_FFT2(self): 
        ''' computes the fourier transform of a 2D field of mean 0'''
        

        # mean = np.mean(self.data)
        # mean_arr = np.zeros_like(self.data)+mean
        # self.data = self.data - mean_arr

        fft_data = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.data)))
        fft_data = fft_data * (self.delta_y * self.delta_x) #scaling the fft 
        self.ps_data = np.abs(fft_data)**2

    def compute_k_2D(self):

        # #need to divis
        # kx = np.arange(0,self.data.shape[0],1)*self.delta_kx # Mpc^-1
        # ky = np.arange(0,self.data.shape[1],1)*self.delta_ky

        # k = []

        # for i in range(len(kx)): 
        #     for j in range(len(ky)):
        #         k.append(np.sqrt(kx[i]**2 + ky[j]**2)) 

        # self.k = np.asarray(k)#/(np.pi*2)

        # self.k = np.asarray(sorted(self.k))


        if self.data.shape[1]%2 == 0: # checking if num samples are even 
            
            self.kx_pos = np.arange(0,1/(self.delta_x*2), (self.delta_kx))
            self.kx_neg = np.arange(-1/(self.delta_x*2),0,(self.delta_kx))
            self.kx = np.concatenate((self.kx_neg,self.kx_pos))
        else: 
            self.kx = np.arange(-1/(self.delta_x*2),1/(self.delta_x*2), (self.delta_kx))

        if self.data.shape[0]%2 == 0: # checking if num samples are even 
            
            self.ky_pos = np.arange(0,1/(self.delta_y*2), (self.delta_ky))
            self.ky_neg = np.arange(-1/(self.delta_y*2),0,(self.delta_ky))
            self.ky = np.concatenate((self.ky_neg,self.ky_pos))
        else: 
            self.ky = np.arange(-1/(self.delta_y*2),1/(self.delta_y*2), (self.delta_ky))


        k = []

        for i in range(len(self.kx)): 
            for j in range(len(self.ky)):
                k.append(np.sqrt(self.kx[i]**2 + self.ky[j]**2)) 

        self.k = np.asarray(k)


    def compute_2D_pspec(self):

        self.cosmo_FFT2() 
        self.compute_k_2D()

        if self.log == True:
            bin_edges = np.logspace(0, max(self.k), self.nbins)
            print(bin_edges)

            self.kmodes = [] # here you have to find each central bin separately 

            for i in range(len(bin_edges)-1):
                half_bin = (bin_edges[i+1]-bin_edges[i])/2
                mid = bin_edges[i]+half_bin
                self.kmodes.append(mid)

            print(self.kmodes)


        else: 
            bin_edges = np.histogram_bin_edges(self.k, bins = self.nbins)

            half_delta_bin = (bin_edges[1]-bin_edges[0])/2

            self.kmodes = bin_edges[:self.nbins]+half_delta_bin 

        a = np.zeros(len(bin_edges)-1) #here you need to take the number of BINS not bin edges! # you alwaysneed an extra edge than you have bin!

        #c holds, in each element, the number of pixels 
        c = np.zeros_like(a)

        for i in range(self.data.shape[0]) : 
            for j in range(self.data.shape[1]):
                kx = ((i-(self.data.shape[0]/2))*self.delta_kx)#need to multiply by kdelta to get your k units
                ky = ((j-(self.data.shape[1]/2))*self.delta_ky)
                kmag = np.sqrt((kx**2) + (ky**2))#/(2*np.pi)
                for k in range(len(bin_edges)-1): #make sure that you speed this up by not considering already binned ps's
                    if bin_edges[k] < kmag <= bin_edges[k+1]: 
                        a[k] += self.ps_data[i,j]
                        c[k] += 1
                        break

        volume = self.Lx*self.Ly 

        print(c)
        print(volume)   

        self.pk = (a/c)/volume

        return self.kmodes[1:],self.pk[1:]

    def compute_dimensionless_2D_pspec(self):

        self.make_2Dpspec()

        self.power =  ((self.kmodes**2) * self.pk)/(2*np.pi)

        return self.kmodes, self.power


    # def make_power_spectrum(self): 

            
    #         hist, bin_edges = np.histogram(k, bins = self.nbins)
            
    #         a = np.zeros(len(bin_edges)-1) #here you need to take the number of BINS not bin edges!  
    #                                # you alwaysneed an extra edge than you have bin!

    #         #c holds, in each element, the number of pixels 
    #         c = np.zeros_like(a)

    #         #Here you sum all the pixels in each k bin. 
    #         for i in range(npix) : 
    #             for j in range(npix):
    #                 for h in range(npix):
    #                     kmag = kdelta*np.sqrt((i-npix/2)**2 + (j-npix/2)**2 + (h-npix/2)**2) #need to multiply by kdelta to get your k units
    #                     for k in range(len(bin_edges)):#make sure that you speed this up by not considering already binned ps's
    #                         if bin_edges[k] < kmag <= bin_edges[k+1]: 
    #                             a[k] += ps_data[i,j,h]
    #                             c[k] += 1
    #                             break 
                            
    #         pk = (a/c) /((self.L*npix)**2) #take average and divide by area to get P(k)
    #         kmodes = bin_edges[1:]
            
        
    #         fft_data = np.fft.fft2(np.fft.fftshift(data))
    #         ps_data = np.abs(np.fft.fftshift(fft_data))**2
        
    #         npix = len(data)
    #         kx = np.fft.fftfreq(npix,delta)
    #         ky = np.fft.fftfreq(npix,delta)
    #         kdelta = kx[1]-kx[0]

    #         k = []

    #         for i in range(len(kx)): 
    #             for j in range(len(ky)):
    #                 k.append(np.sqrt(kx[i]**2 + ky[j]**2)) 
                
    #         hist, bin_edges = np.histogram(k, bins = nbins)
            
    #         a = np.zeros(len(bin_edges)-1) #here you need to take the number of BINS not bin edges!  
    #                                # you alwaysneed an extra edge than you have bin!


    #         c = np.zeros_like(a)

    #         #Here you sum all the pixels in each k bin. 
    #         for i in range(npix) : 
    #             for j in range(npix):
    #                 kmag = kdelta*np.sqrt((i-npix/2)**2 + (j-npix/2)**2) #need to multiply by kdelta to get your k units
    #                 for k in range(len(bin_edges)): #make sure that you speed this up by not considering already binned ps's
    #                     if bin_edges[k] < kmag <= bin_edges[k+1]: 
    #                         a[k] += ps_data[i,j]
    #                         c[k] += 1
    #                         break
                        
                        
    #         pk = (a/c) /((delta*npix)**2)
    #         kmodes = bin_edges[1:]
            
        # return kmodes, pk





