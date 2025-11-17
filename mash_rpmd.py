import numpy as np
import utils
import sys
from scipy.linalg import expm
from scipy.special import erf
import math

#Define class for Mash-rpmd
#It's a child-class of the rpmd parent class
#There are two ways to implement the RP-MASH: normal way and centroid approximation
#Normal: nothing needs to be edit in the __init__;
#centroid approximation: only the centroid nuclear mode is coupled to electronic states, centroid_bool=True.

import map_rpmd
import integrator

class mash_rpmd( map_rpmd.map_rpmd ):

    def __init__( self, nstates, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, 
                 mapR=None, mapP=None, mapSx=None, mapSy=None, mapSz=None, nucR=None, nucP=None, 
                 spinmap_bool=False, centroid_bool=False, bead_bool=False, functional_param=None, langevin=None):

        super().__init__( 'RP-MASH', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP, langevin )
        
        self.spin_map = spinmap_bool # Boolean that decides if we use spin mapping variables
        self.centroid_bool = centroid_bool # Boolean that decides if we use the centroid of nuclei to be coupled with electronic states
        self.bead_bool = bead_bool #Boolean that decides if if we use the bead average potential of nuclei that is coupled with electronic states
        self.functional_param = functional_param # The function parameter if the Gaussian function is used for the momentum rescaling

        if (spinmap_bool == True):
            if (nstates != 2):
                print('ERROR: The spin mapping variable is only for 2-state systems')
                exit()
            self.mapSx = mapSx
            self.mapSy = mapSy
            self.mapSz = mapSz
            self.spin_map_error_check()

        if (bead_bool == True and centroid_bool==True):
            print('ERROR: bead_bool and centroid_bool cannot be True at the same time')
            exit()
        
    #####################################################################

    def run_dynamics( self, Nsteps=None, Nprint=100, delt=None, intype=None, init_time=0.0, small_dt_ratio=1 ):

        #Top-level routine to run dynamics specific to MASH

        #Number of decimal places when printing current time
        #modf splits the floating number into integer and decimal components
        #converting to string, taking the length, and subtracting 2 (for the 0.) gives us the length of just the decimal component
        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

        #Error checks REMINDER TO REWITE THIS GUY FOR MASH
        #self.dynam_error_check( Nsteps, delt, intype )

        #Initialize the integrator
        self.integ = integrator.integrator( self, delt, intype, small_dt_ratio )

        #Automatically initialize nuclear momentum from MB distribution if none have been specified
        if( self.nucP is None ):
            print('Automatically initializing nuclear momentum to Maxwell-Boltzmann distribution at beta_p = ', self.beta_p*self.nbds ,' / ',self.nbds)
            self.get_nucP_MB()

        print()
        print( '#########################################################' )
        print( 'Running', self.methodname, 'Dynamics for', Nsteps, 'Steps' )
        print( 'Spin-mapping:', self.spin_map)
        print( 'centroid approximation:', self.centroid_bool)
        print( 'bead approximation:', self.bead_bool)

        print( '#########################################################' )
        print()

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_nucP   = open( 'nucP.dat', 'w' )
        self.file_mapSx  = open( 'mapSx.dat','w' )
        self.file_mapSy  = open( 'mapSy.dat','w' )
        self.file_mapSz  = open( 'mapSz.dat','w' )
        #self.file_mapR   = open( 'mapR.dat','w' )
        #self.file_mapP   = open( 'mapP.dat', 'w' )
        #self.file_Q      = open( 'Q.dat', 'w')
        #self.file_phi    = open( 'phi.dat', 'w')
        #self.file_semi   = open( 'mvsq.dat', 'w')

        current_time = init_time
        step = 0
        for step in range( Nsteps ):

            #Print data starting with initial time
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at step', step, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for', self.methodname, 'Dynamics calculation')
                self.print_data( current_time )
                sys.stdout.flush()

            #Integrate EOM by one time-step
            self.integ.onestep( self, step )

            #Increase current time
            current_time = init_time + delt * (step+1)

        #Print data at final step regardless of Nprint
        print('Writing data at step', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for', self.methodname, 'Dynamics calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_nucP.close()
        self.file_mapSx.close()
        self.file_mapSy.close()
        self.file_mapSz.close()
        #self.file_mapR.close()
        #self.file_mapP.close()
        #self.file_Q.close()
        #self.file_phi.close()
        #self.file_semi.close()

        print()
        print( '#########################################################' )
        print( 'END', self.methodname, 'Dynamics' )
        print( '#########################################################' )
        print()
        
    #####################################################################

    def get_timederivs( self ):
        #subroutine to calculate the time-derivatives of all position/momenta

        #update electronic Hamiltonian matrix
        self.potential.calc_Hel( self.nucR )

        #Calculate time-derivative of nuclear position
        dnucR = self.get_timederiv_nucR()

        #Calculate time-derivative of nuclear momenta
        dnucP = self.get_timederiv_nucP()

        #Calculate time-derivative of mapping position and momentum
        if (self.spin_map == False):
            dmapR = self.get_timederiv_mapR()
            dmapP = self.get_timederiv_mapP()

            return dnucR, dnucP, dmapR, dmapP
        
        else:
            dmapSx = self.get_timederiv_mapSx()
            dmapSy, dmapSz = self.get_timederiv_mapSyz()

            return dnucR, dnucP, dmapSx, dmapSy, dmapSz

    #####################################################################

    def get_timederiv_nucR( self ):
        #Subroutine to calculate the time-derivative of the nuclear positions for each bead

        return self.nucP / self.mass

    #####################################################################

    def get_timederiv_nucP( self, intRP_bool=True ):
        #Subroutine to calculate the time-derivative of the nuclear momenta for each bead

        #Force associated with harmonic springs between beads and the state-independent portion of the potential
        #This is dealt with in the parent class  
        #If intRP_bool is False it does not calculate the contribution from the harmonic ring polymer springs

        if (self.nbds > 1):
            d_nucP = super().get_timederiv_nucP( intRP_bool )
        else:
            d_nucP = super().get_timederiv_nucP( intRP_bool = False )

        if (self.spin_map==False):
            #Calculate contribution from MMST term
            #XXX could maybe make this faster getting rid of double index in einsum
            d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapR, self.potential.d_Hel, self.mapR )
            d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapP, self.potential.d_Hel, self.mapP )

            #add the state-average potential
            if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
                d_nucP +=  0.5 * np.einsum( 'ijnn -> ij', self.potential.d_Hel )
        
        else:
            #The MASH nuclear force, note that Hel here are adiabatic surfaces
            dE_bo = self.potential.get_bopes_derivs(self.nucR)
            d_Vz = (dE_bo[:,:,1] - dE_bo[:,:,0])/2
            #Calculate delta function force as functional limit or with adaptive timestep
            if (self.functional_param != None):
                E_bo = self.potential.get_bopes(self.nucR)
                Vz = (E_bo[:,1] - E_bo[:,0])/2
                NAC = self.potential.calc_NAC(self.nucR)
                d_nucP += -d_Vz * erf(self.mapSz[:,np.newaxis] / self.functional_param) + 4 * Vz[:,np.newaxis] * NAC * self.mapSx[:,np.newaxis] * np.exp(-(self.mapSz[:,np.newaxis] / self.functional_param)**2) / (self.functional_param * np.sqrt(np.pi))
                #d_nucP += -d_Vz * np.sign(self.mapSz[:,np.newaxis])
            else:
                d_nucP += -d_Vz * np.sign(self.mapSz[:,np.newaxis]) - (dE_bo[:,:,1] + dE_bo[:,:,0])/2

        return d_nucP

    #####################################################################

    def get_timederiv_mapR( self ):
        #Subroutine to calculate the time-derivative of just the mapping position for each bead

        d_mapR =  np.einsum( 'inm,im->in', self.potential.Hel, self.mapP )

        return d_mapR

    #####################################################################

    def get_timederiv_mapP( self ):
        #Subroutine to calculate the time-derivative of just the mapping momentum for each bead

        d_mapP = -np.einsum( 'inm,im->in', self.potential.Hel, self.mapR )

        return d_mapP

    #####################################################################

    def get_2nd_timederiv_mapR( self, d_mapP ):
        #Subroutine to calculate the second time-derivative of just the mapping positions for each bead
        #This assumes that the nuclei are fixed - used in vv style integrators

        d2_mapR = np.einsum( 'inm,im->in', self.potential.Hel, d_mapP )

        return d2_mapR

    #####################################################################

    def get_timederiv_mapSx( self ):

        if (self.centroid_bool==False and self.bead_bool==False):

            E_bo = self.potential.get_bopes(self.nucR)
            Vz = (E_bo[:,1] - E_bo[:,0])/2
            NAC = self.potential.calc_NAC(self.nucR)

            d_mapSx = 2 * np.sum( NAC * self.nucP / self.mass, axis = 1 ) * self.mapSz - 2 * Vz * self.mapSy #axis = 1 corresponds to the dimension of nuclear DOFs
                    
        elif self.centroid_bool==True:

            R_bar = np.mean(self.nucR, axis = 0)
            Rbar_arr = np.tile(R_bar, (self.nbds, 1))
            E_bo = self.potential.get_bopes(Rbar_arr)
            Vz = (E_bo[:,1] - E_bo[:,0])/2
            NAC = self.potential.calc_NAC(Rbar_arr)

            velo_bar = np.mean(self.nucP/self.mass, axis = 0)
            vbar_arr = np.tile(velo_bar, (self.nbds, 1))

            d_mapSx = 2 * np.sum(NAC * vbar_arr, axis = 1) * self.mapSz - 2 * Vz * self.mapSy

        elif self.bead_bool==True:
            E_bo = self.potential.get_bopes(self.nucR)
            Vz = (E_bo[:,1] - E_bo[:,0])/2
            NAC = self.potential.calc_NAC(self.nucR)

            d_mapSx = np.mean(2 * np.sum( NAC * self.nucP / self.mass, axis = 1 ) * self.mapSz - 2 * Vz * self.mapSy) * np.ones(self.nbds)

        return d_mapSx

    #####################################################################

    def get_timederiv_mapSyz(self):

        if (self.centroid_bool==False and self.bead_bool==False ):

            E_bo = self.potential.get_bopes(self.nucR)
            Vz = (E_bo[:,1] - E_bo[:,0])/2
            NAC = self.potential.calc_NAC(self.nucR)

            d_mapSy = 2 * Vz * self.mapSx
            d_mapSz = -2 * np.sum( NAC * self.nucP / self.mass, axis = 1 ) * self.mapSx

        elif(self.centroid_bool==True):
            R_bar = np.mean(self.nucR, axis = 0)
            Rbar_arr = np.tile(R_bar, (self.nbds, 1))
            E_bo = self.potential.get_bopes(Rbar_arr)
            Vz = (E_bo[:,1] - E_bo[:,0])/2
            NAC = self.potential.calc_NAC(Rbar_arr)

            velo_bar = np.mean(self.nucP/self.mass, axis = 0)
            vbar_arr = np.tile(velo_bar, (self.nbds, 1))

            d_mapSy = 2 * Vz * self.mapSx
            d_mapSz = -2 * np.sum(NAC * vbar_arr, axis = 1) * self.mapSx

        elif(self.bead_bool==True):
            E_bo = self.potential.get_bopes(self.nucR)
            Vz = (E_bo[:,1] - E_bo[:,0])/2
            NAC = self.potential.calc_NAC(self.nucR)

            d_mapSy = np.mean(2 * Vz * self.mapSx) * np.ones(self.nbds)
            d_mapSz = np.mean(-2 * np.sum( NAC * self.nucP / self.mass, axis = 1 ) * self.mapSx) * np.ones(self.nbds)

        return d_mapSy, d_mapSz

   #####################################################################

    def get_PE( self ):
        #Subroutine to calculate potential energy associated with mapping variables and nuclear position

        #Internal ring-polymer modes, 0 if there is only one bead (i.e., LSC-IVR)
        if self.nbds > 1:
            engpe = self.potential.calc_rp_harm_eng( self.nucR, self.beta_p, self.mass )
        else:
            engpe = 0

        #State independent term
        engpe += self.potential.calc_state_indep_eng( self.nucR )

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel(self.nucR)

        if(self.spin_map==False):
            #MMST Term
            engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapR, self.potential.Hel, self.mapR ) )
            engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapP, self.potential.Hel, self.mapP ) )

            if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
                engpe += -0.5 * np.sum( np.einsum( 'inn -> i', self.potential.Hel ) )

        else:
            E_bo = self.potential.get_bopes(self.nucR)
            engpe += np.sum((E_bo[:,1]-E_bo[:,0])/2 * np.sign(self.mapSz) + (E_bo[:,1]+E_bo[:,0])/2)
            
        return engpe

    #####################################################################

    def spin_map_error_check(self):

        if( self.mapSx is not None and self.mapSx.shape != ( self.nbds, ) ):
            print('ERROR: Size of spin mapping variable Sx doesnt match bead number')
            exit()

        if( self.mapSy is not None and self.mapSy.shape != ( self.nbds, ) ):
            print('ERROR: Size of spin mapping variable Sy doesnt match bead number')
            exit()

        if( self.mapSz is not None and self.mapSz.shape != ( self.nbds, ) ):
            print('ERROR: Size of spin mapping variable Sz doesnt match bead number')
            exit()

    #####################################################################

    def init_map_spin(self, init_state = None):

        #Initialize spin mapping variables by uniformly samping them on the spin sphere
        #Sx = sin(theta)cos(phi), Sy = sin(theta)sin(phi), Sz = cos(theta)
        #Sampling by randomly choose cos(theta) in[-1,1], and phi in [0, 2pi]
        #init_state could be 0 or 1, corresponding to cos_theta equal (-1,0) or (0,1), respectively

        print()
        print( '#########################################################' )
        print( 'Initializing SPIN Mapping Variables uniformly on the spin sphere' )
        print( '#########################################################' )
        print()

        if (self.centroid_bool==False and self.bead_bool==False) :
            cos_theta = self.rng.uniform( -1, 1, size = self.nbds )
            if init_state == 0:
                cos_theta = self.rng.uniform( -1, 0, size = self.nbds )
            if init_state == 1:
                cos_theta = self.rng.uniform( 0, 1, size = self.nbds )
            sin_theta = np.sqrt( 1 - cos_theta**2 )
            phi = self.rng.uniform( 0, 2*np.pi, size = self.nbds )
        
            self.mapSx = sin_theta * np.cos(phi)
            self.mapSy = sin_theta * np.sin(phi)
            self.mapSz = cos_theta
        
        else:
            # mapping variables are the same across all beads
            cos_theta = self.rng.uniform( -1, 1 )
            if init_state == 0:
                cos_theta = self.rng.uniform( -1, 0 )
            if init_state == 1:
                cos_theta = self.rng.uniform( 0, 1 )
            sin_theta = np.sqrt( 1 - cos_theta**2 )
            phi = self.rng.uniform( 0, 2*np.pi )

            self.mapSx = sin_theta * np.cos(phi) * np.ones(self.nbds)
            self.mapSy = sin_theta * np.sin(phi) * np.ones(self.nbds)
            self.mapSz = cos_theta * np.ones(self.nbds)

    #####################################################################

    def get_sampling_eng(self):

        #XXX-not applicable
        return None

    #####################################################################

    def print_data( self, current_time):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy associated with mapping variables and nuclear position
        #This also updates the electronic Hamiltonian matrix
        engpe = self.get_PE()

        #Calculate Nuclear Kinetic Energy
        engke = self.potential.calc_nuc_KE( self.nucP, self.mass )

        #Calculate total energy
        etot = engpe + engke

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(5+self.nnuc)
        output[0] = current_time
        output[1] = etot
        output[2] = engke
        output[3] = engpe
        output[4] = np.mean(np.sign(self.mapSz))
        output[5:] = nucR_com
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, current_time ).reshape(1, self.nucR.size+1), fmt_str )
        np.savetxt( self.file_nucP, np.insert( self.nucP.flatten(), 0, current_time ).reshape(1, self.nucP.size+1), fmt_str )

        #columns go as bead_1_state_1 bead_1_state_2 ... bead_1_state_K bead_2_state_1 bead_2_state_2 ...
        np.savetxt( self.file_mapSx, np.insert( self.mapSx.flatten(), 0, current_time ).reshape(1, self.mapSx.size+1), fmt_str )
        np.savetxt( self.file_mapSy, np.insert( self.mapSy.flatten(), 0, current_time ).reshape(1, self.mapSy.size+1), fmt_str )
        np.savetxt( self.file_mapSz, np.insert( self.mapSz.flatten(), 0, current_time ).reshape(1, self.mapSz.size+1), fmt_str )

        self.file_nucR.flush()
        self.file_nucP.flush()
        self.file_mapSx.flush()
        self.file_mapSy.flush()
        self.file_mapSz.flush()
