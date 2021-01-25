import numpy as np
import os


#cosmology constants
c = 3*10**8
H0 = float(68)/float(3.086e19)
OMm = 0.31
OMl = 0.75
rho_crit = 8.62*10**-27
mp = 1.672e-27
baryon2DMfrac = 0.1
Hy2baryonfrac = 0.75
HyIGMfrac = 1
pc = 3.08*10**16 # pc in terms of m


def rand_choice(fduty, amount):
    choices = (0, 1)
    choices = np.array(choices)
    p = (1-fduty, fduty)
    p = np.array(p)
    #print('SHAPE of LIST TO FILER0', list_to_filter.shape[0])
    #DELETETHIS = np.random.choice(choices, list_to_filter.shape[0], p = p)
    #print("RETURNING " + str(DELETETHIS.shape))
    return np.random.choice(choices, amount, p = p)

#this code computes the dispersion measure of an FRB at location x,y in a box along the LoS from redshift
#z_i to z_end
def compute_DM(x,y, xH_lightcone, density_lightcone, lightcone_redshifts, Halolightcone , **kwargs):
    
        #random values which determine it will hit a halo or not
        #chi = rand_choice(0.0001, lightcone_redshifts.shape[0])
        
        #starting redshift
        #print(lightcone_redshifts.shape)
        z_start = lightcone_redshifts[0]
        
        #ending redshift
        z_end = lightcone_redshifts[-1]
        #print('z_end is ' , z_end)
        
        #initialize the DM
        DM = 0
        
        #the redshifts of each cell in the lightcone will tell us what the redshift spacing is
        #print(lightcone_redshifts[0], lightcone_redshifts[1])
        delta_z = lightcone_redshifts[0] - lightcone_redshifts[1]
        #print('delta z is ' , delta_z   )
        
        #as an approximation, we may not want to integrate through the entire lightcone
        #if not specified by the user then assume we are integrating through the entire lightcone
        if 'depth' in kwargs:
            depth = kwargs.get('depth')
        else:
            depth = lightcone_redshifts.shape[0]

        #loop through the entire box
        z_i = z_start
        #i = 0
        if np.random.uniform(-1, 1000) < 0:
            print('starting at ' + str(z_start) + ' and ending at ' + str(z_end) + ' in steps of ' + str(delta_z),' for a total of' , lightcone_redshifts.shape[0])

        for i in range(lightcone_redshifts.shape[0]):
            #the redshift at this location is
            #z_i = z_start - delta_z*i
            z_i = lightcone_redshifts[i]
            #print(z_i, i)
            #print('the redshift is ' , z_i)
            
            #H(z_i) is
            H = H0*np.sqrt(OMm*(1 + z_i)**3 + OMl)
            
            #compute the dispersion
            #print(density_lightcone[i][x][y], xH_lightcone[i][x][y] )
            insta_DM  = float(((1+z_i)**3)*c*delta_z*np.abs(float(1) + float(density_lightcone[i][x][y]))*(float(1) - float(xH_lightcone[i][x][y]))*rho_crit*OMm*baryon2DMfrac*HyIGMfrac)/float(mp*H*(1 + z_i)**2)
            
            #print('at i' + str(i) + ' we are at redshift ' + str(z_i) + ' and are adding ' + str(insta_DM) + ' to our DM')
            DM += ( ((1+z_i)**3)*c*delta_z*np.abs(float(1) + float(density_lightcone[i][x][y]))*(float(1) - float(xH_lightcone[i][x][y]))*rho_crit*OMm*baryon2DMfrac*HyIGMfrac/float(mp*H*(1 + z_i)**2)    )#+ 200*pc*chi[i]*Halolightcone[i][x][y]/float((0.01**3)) )

                #if 200*pc*chi[i]*Halolightcone[i][x][y]/float((0.01**3)) !=0 :
#  print('there are ' , Halolightcone[i][x][y], 'halos at', x, y, ' adding ',200*pc*chi[i]*Halolightcone[i][x][y]/float((0.01**3)),  )
#i +=1

        return DM

