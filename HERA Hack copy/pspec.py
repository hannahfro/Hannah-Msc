import numpy as np

def make_power_spectrum(data,npix,dim,delta,nbins): 
    
    if dim == 3: 
        fft_data = np.fft.fftn(np.fft.fftshift(data)) 
        ps_data = np.abs(np.fft.fftshift(fft_data))**2 #this has variance equal to p(k)
        
        kx = np.fft.fftfreq(npix,delta)
        ky = np.fft.fftfreq(npix,delta)
        kz = np.fft.fftfreq(npix,delta)
        kdelta = kx[1]-kx[0]

        k = []

        for i in range(len(kx)): 
            for j in range(len(ky)):
                for h in range(len(kz)):
                    k.append(np.sqrt(kx[i]**2 + ky[j]**2 + kz[h]**2))
        
        hist, bin_edges = np.histogram(k, bins = nbins)
        
        a = np.zeros(len(bin_edges)-1) #here you need to take the number of BINS not bin edges!  
                               # you alwaysneed an extra edge than you have bin!

        #c holds, in each element, the number of pixels 
        c = np.zeros_like(a)

        #Here you sum all the pixels in each k bin. 
        for i in range(npix) : 
            for j in range(npix):
                for h in range(npix):
                    kmag = kdelta*np.sqrt((i-npix/2)**2 + (j-npix/2)**2 + (h-npix/2)**2) #need to multiply by kdelta to get your k units
                    for k in range(len(bin_edges)):#make sure that you speed this up by not considering already binned ps's
                        if bin_edges[k] < kmag <= bin_edges[k+1]: 
                            a[k] += ps_data[i,j,h]
                            c[k] += 1
                            break 
                        
        pk = (a/c) /((delta*npix)**2) #take average and divide by area to get P(k)
        kmodes = bin_edges[1:]
        
    elif dim == 2: 
        fft_data = np.fft.fft2(np.fft.fftshift(data))
        ps_data = np.abs(np.fft.fftshift(fft_data))**2
    

        kx = np.fft.fftfreq(npix,delta)
        ky = np.fft.fftfreq(npix,delta)
        kdelta = kx[1]-kx[0]

        k = []

        for i in range(len(kx)): 
            for j in range(len(ky)):
                k.append(np.sqrt(kx[i]**2 + ky[j]**2)) 
            
        hist, bin_edges = np.histogram(k, bins = nbins)
        
        a = np.zeros(len(bin_edges)-1) #here you need to take the number of BINS not bin edges!  
                               # you alwaysneed an extra edge than you have bin!


        c = np.zeros_like(a)

        #Here you sum all the pixels in each k bin. 
        for i in range(npix) : 
            for j in range(npix):
                kmag = kdelta*np.sqrt((i-npix/2)**2 + (j-npix/2)**2) #need to multiply by kdelta to get your k units
                for k in range(len(bin_edges)): #make sure that you speed this up by not considering already binned ps's
                    if bin_edges[k] < kmag <= bin_edges[k+1]: 
                        a[k] += ps_data[i,j]
                        c[k] += 1
                        break
                    
                    
        pk = (a/c) /((delta*npix)**2)
        kmodes = bin_edges[1:]
        
    return kmodes, pk