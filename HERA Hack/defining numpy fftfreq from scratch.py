  
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
        


        if self.data.shape[1]%2 == 0: # checking if num samples are even 
            
            self.kx_pos = np.arange(0,1/(self.delta_x*2), (self.delta_kx))
            self.kx_neg = np.arange(-1/(self.delta_x*2),0,(self.delta_kx))
            self.kx_1 = np.concatenate((self.kx_neg,self.kx_pos))
        else: 
            self.kx_1 = np.arange(-1/(self.delta_x*2),1/(self.delta_x*2), (self.delta_kx))

        if self.data.shape[0]%2 == 0: # checking if num samples are even 
            
            self.ky_pos = np.arange(0,1/(self.delta_y*2), (self.delta_ky))
            self.ky_neg = np.arange(-1/(self.delta_y*2),0,(self.delta_ky))
            self.ky_1 = np.concatenate((self.ky_neg,self.ky_pos))
        else: 
            self.ky_1 = np.arange(-1/(self.delta_y*2),1/(self.delta_y*2), (self.delta_ky))
