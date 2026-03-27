#Define parent class for all mapping-rpmd methods

import numpy as np
import scipy.special as scp
from abc import ABC, abstractmethod
import utils
import potential
import integrator
import normal_mode
import sys
import math

####### PARENT CLASS FOR MAPPING-RPMD METHODS######

class map_rpmd(ABC):

    #####################################################################

    @abstractmethod
    def __init__( self, methodname, nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP, langevin=None ):

        #Initialize all variables
        #This is an abstractmethod so all default values are set at sub-class level

        print() #Print blank line for readability

        self.methodname = methodname #string defining the sub-class method name
        self.potype     = potype     #the type of potential

        self.nstates = nstates   #number of electronic states
        self.nnuc    = nnuc      #number of nuclear modes
        self.nbds    = nbds      #number of ring polymer beads
        self.beta    = beta      #inverse temperature
        self.beta_p  = beta/nbds #inverse temperature at higher value due to bead-number
        self.mass    = mass      #mass of nuclear modes - 1d array size of Nnuc b/c mass of all beads for a given nuclei is the same
        self.mapR    = mapR      #position of mapping variables, dimension Nbds x Nstates
        self.mapP    = mapP      #momentum of mapping variables, dimension Nbds x Nstates
        self.nucR    = nucR      #position of nuclear modes, dimension of Nbds x Nnuc
        self.nucP    = nucP      #momentum of nuclear modes, dimension of Nbds x Nnuc
        self.langevin=langevin   #langevin fraction (damping) coefficient, dimension of Nnuc
 
        #Initialize instance of random number generator
        self.rng = np.random.default_rng()

        #Input error check
        if (self.methodname != 'sb-NRPMD'):
            self.init_error_check()

        #Define potential
        self.potential = potential.set_potential( potype, potparams, nstates, nnuc, nbds )

        #Define variables to be used as output files
        self.file_output = None
        self.file_nucR   = None
        self.file_nucP   = None
        self.file_mapR   = None
        self.file_mapP   = None
        self.file_Q      = None
        self.file_phi    = None

    #####################################################################

    def run_dynamics( self, Nsteps=None, Nprint=100, delt=None, intype=None, init_time=0.0, small_dt_ratio=1 ):

        #Top-level routine to run dynamics

        #Number of decimal places when printing current time
        #modf splits the floating number into integer and decimal components
        #converting to string, taking the length, and subtracting 2 (for the 0.) gives us the length of just the decimal component
        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

        #Error checks
        self.dynam_error_check( Nsteps, delt, intype )

        #Initialize the integrator
        self.integ = integrator.integrator( self, delt, intype, small_dt_ratio )

        #Automatically initialize nuclear momentum from MB distribution if none have been specified
        if( self.nucP is None ):
            print('Automatically initializing nuclear momentum to Maxwell-Boltzmann distribution at beta_p = ', self.beta_p*self.nbds ,' / ',self.nbds)
            self.get_nucP_MB()

        print()
        print( '#########################################################' )
        print( 'Running', self.methodname, 'Dynamics for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_nucP   = open( 'nucP.dat', 'w' )
        self.file_mapR   = open( 'mapR.dat','w' )
        self.file_mapP   = open( 'mapP.dat', 'w' )
        self.file_Q      = open( 'Q.dat', 'w')
        self.file_phi    = open( 'phi.dat', 'w')
        self.file_semi   = open( 'mvsq.dat', 'w')

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
        print('Writing data at step ', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for', self.methodname, 'Dynamics calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_nucP.close()
        self.file_mapR.close()
        self.file_mapP.close()
        self.file_Q.close()
        self.file_phi.close()
        self.file_semi.close()

        print()
        print( '#########################################################' )
        print( 'END', self.methodname, 'Dynamics' )
        print( '#########################################################' )
        print()

    #####################################################################

    def run_dynamics_massN( self, Nsteps=None, Nprint=100, delt=None, intype=None, init_time=0.0, small_dt_ratio=1 ):
        
        #Routine to run the dynamics with the partition funciton e^(\beta H) instead of e^(\beta_N H_N)

        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

    def run_MC( self, Nsteps=1000, Nprint=100, disp_nuc=0.1, disp_map=0.1, nmove=1, nm_bool=False, resamp=None, freeze_nuc=False ):

        #Routine to run Monte-Carlo for mapping variables and the nuclei
        #Random displacements done in real-space for mapping variables
        #Option for random displacements using normal modes or real-space for nuclei

        #Nsteps   - number of Monte Carlo steps
        #disp_nuc - max size of random displacement for nuclear positions or zero frequency mode
        #disp_map - max size of random displacement for mapping variables
        #nmove    - number of nuclei that are attempted to be moved at each MC step, all beads for each nuclei are moved
        #resamp   - number of MC steps between resampling the electronic variables from a normal distribution
        #freeze_nuc - freeze nuclear variables for entire mc run

        #Initialize nuclear positions to zero if None specified
        if( self.nucR is None ):
            print('Automatically initializing all nuclear positions to zero')
            self.nucR = np.zeros([self.nbds,self.nnuc])

        #Initilize mapping variables arbitrarily to state zero if None specified
        if( self.mapR is None or self.mapP is None ):
            print('Automatically initializing mapping variables to state zero')
            self.init_map_semiclass_estimator( occstate=0 )

        #Set resamp to Nsteps + 1 so the MC never resamples if None is specified
        if( resamp is None ): resamp = Nsteps + 1
        #Error Checks
        self.MC_error_check( nmove )

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_mapR   = open( 'mapR.dat','w' )
        self.file_mapP   = open( 'mapP.dat', 'w' )

        print()
        print( '#########################################################' )
        print( 'Running', self.methodname, 'MC Routine for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        if( nm_bool ):
            #Calculate frequencies of normal modes of ring-polymer
            #note that these are mass-independent and are thus the same for all nuclei
            nm_freq = normal_mode.calc_normal_mode_freq( self.beta_p, self.nbds )

            #Calculate initial set of normal modes
            nm = np.zeros([self.nbds,self.nnuc])
            for i in range(self.nnuc):
                nm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

        #Initialize original positions
        orig_nucR  = np.copy( self.nucR )
        orig_mapR  = np.copy( self.mapR )
        orig_mapP  = np.copy( self.mapP )
        if( nm_bool ):
            orignm = np.copy( nm )

        #Calculate energy for initial configuration
        engold = self.get_sampling_eng()

        numacc = 0 #Counter to keep track of number of accepted MC moves
        for step in range( Nsteps ):

            #Print data starting with initial step
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at MC step', step, 'for', self.methodname, 'MC routine')
                self.print_MC_data( step, numacc )
                sys.stdout.flush()

            #check if resampling step
            if( np.mod(step + 1, resamp) == 0 ):
                self.mapR = self.rng.normal(scale=1 / np.sqrt(2), size=self.mapR.shape)
                self.mapP = self.rng.normal(scale=1 / np.sqrt(2), size=self.mapP.shape)
            else:
                #### Move Nuclei ###
                if not freeze_nuc:
                    #Randomly pick a starting nuclei
                    strt = self.rng.integers( 0 , self.nnuc )

                    #Loop over nmove nuclei taking into account if strt+nmove > nnuc
                    for i in ( np.arange( strt, strt+nmove ) % self.nnuc ):

                        if( nm_bool ):
                            #Convert position to normal modes
                            nm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

                            #Displace zero-frequency mode
                            nm[0,i] += self.rng.uniform(-1.0,1.0) * disp_nuc

                            #Randomly pull other normal modes from gaussian distribution
                            for k in range(1,self.nbds):
                                sigma   = 1.0 / np.sqrt( self.beta_p * self.mass[i] * nm_freq[k]**2 )
                                nm[k,i] = self.rng.normal( scale = sigma )

                            #Back convert normal modes to real-space positions
                            self.nucR[:,i] = normal_mode.normal_mode_to_real( nm[:,i] )
                        else:
                            self.nucR[:,i] += self.rng.uniform(-1.0, 1.0, self.nbds ) * disp_nuc
                #######

                #Move Mapping Variables
                if self.methodname == 'sb-NRPMD':
                    self.mapR += self.rng.uniform(-1.0, 1.0, self.nstates ) * disp_map
                    self.mapP += self.rng.uniform(-1.0, 1.0, self.nstates ) * disp_map
                else:
                    self.mapR += self.rng.uniform(-1.0, 1.0, (self.nbds, self.nstates) ) * disp_map
                    self.mapP += self.rng.uniform(-1.0, 1.0, (self.nbds, self.nstates) ) * disp_map

            #Calculate energy of trial position
            engnew = self.get_sampling_eng()
            d_eng = engnew - engold

            #Check acceptance condition
            if d_eng < 0:
                #accept new configuration
                numacc += 1
                orig_nucR = np.copy(self.nucR)
                orig_mapR = np.copy(self.mapR)
                orig_mapP = np.copy(self.mapP)
                if( nm_bool ):
                    orig_nm = np.copy(nm)
                engold    = engnew
            else:
                if self.methodname == 'sb-NRPMD': acc_cond = np.exp( -self.beta * ( d_eng ) )
                else: acc_cond = np.exp( -self.beta_p * ( d_eng ) )
                if( self.rng.random() < acc_cond ):
                    #accept new configuration
                    numacc += 1
                    orig_nucR = np.copy(self.nucR)
                    orig_mapR = np.copy(self.mapR)
                    orig_mapP = np.copy(self.mapP)
                    if( nm_bool ):
                        orig_nm = np.copy(nm)
                    engold    = engnew
                else:
                    #reject new configuration and reset positions to original position
                    self.nucR  = np.copy( orig_nucR )
                    self.mapR  = np.copy( orig_mapR )
                    self.mapP  = np.copy( orig_mapP )
                    if( nm_bool ):
                        nm = np.copy( orig_nm )

        #Print data at final step regardless of Nprint
        print('Writing data at MC step', step+1, 'for', self.methodname, 'MC routine')
        self.print_MC_data( step+1, numacc )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_mapR.close()
        self.file_mapP.close()

        print()
        print( '#########################################################' )
        print( 'End',self.methodname, 'MC Routine' )
        print( '#########################################################' )
        print()

    #####################################################################

    def run_PIMD( self, Nsteps=None, resample=None, intype=None, Nprint=100, delt=None, init_time=0.0, small_dt_ratio=1 ):

        #routine to run PIMC only for the nuclei with vv-type integrators
        #resample - the number of steps that equilibrate the temperature (NVT MD)
        #small_dt_ratio - the ratio of electronic time-step with nuclear one. include this only b/c we use the integral package

        #Set resamp to Nsteps + 1 so the MC never resamples if None is specified
        if( resample is None ): resample = Nsteps + 1

        #error check for the intype
        if(intype != 'vv' and intype != 'analyt' and intype != 'cayley'):
            print('ERROR: the intype should be one of these: vv, analyt or cayley. Now it is', intype)
            exit()

        #Initialize the integrator
        self.integ = integrator.integrator( self, delt, intype, small_dt_ratio )

        #converting to string, taking the length, and subtracting 2 (for the 0.) gives us the length of just the decimal component
        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

        #Automatically initialize nuclear momentum from MB distribution if none have been specified
        if( self.nucP is None ):
            print('Automatically initializing nuclear momenta to Maxwell-Boltzmann distribution at beta_p =', self.beta_p*self.nbds ,'/',self.nbds)
            self.get_nucP_MB()

        #Initialize nuclear positions to zero if None specified
        if( self.nucR is None ):
            print('Automatically initializing all nuclear positions to zero')
            self.nucR = np.zeros([self.nbds, self.nnuc])

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_nucP   = open( 'nucP.dat','w' )

        print()
        print( '#########################################################' )
        print('running equilibrium PIMD for', Nsteps, 'at beta_p =', self.beta_p*self.nbds, '/',self.nbds)
        print( '#########################################################' )
        print()

        current_time = init_time
        step = 0

        for step in range( Nsteps ):

            #resample the nucP if we need to
            if( np.mod( step+1, resample ) == 0 ):
                print('resample the bead momentum to run within an NVT ensemble')
                self.get_nucP_MB()

            #Print data starting with initial time
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at step', step, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for PIMD calculation')
                self.print_PIMD_data( current_time )
                sys.stdout.flush()

            #Outer loop to integrate EOM using velocity-verlet like algorithms
            #This includes pengfei's implementation, and the analtyical and cayley modification of it

            #If initial step of dynamics need to initialize derivative of nuclear momentum (aka the force on the nuclei)
            #NOTE: moving forward these calls may need to be generalized to allow for other types of methods
            if( step == 0 ):
                if( intype == 'vv' ):
                    self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )
                else:
                    self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR )

            #Update nuclear momentum by 1/2 a time-step
            self.integ.update_vv_nucP( self )

            #Update nuclear position for full time-step
            if( intype == 'vv' ):
                self.integ.update_vv_nucR( self )
            elif( intype == 'analyt' ):
                self.integ.update_analyt_nucR( self )
            elif( intype == 'cayley' ):
                self.integ.update_cayley_nucR( self )

            #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
            #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
            if( intype == 'vv' ):
                self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )
            else:
                self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR )

            #Update nuclear momentum to full time-step
            self.integ.update_vv_nucP( self )

            #Increase current time
            current_time = init_time + delt * (step+1)

        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for PIMD Dynamics calculation')
        self.print_PIMD_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_nucP.close()

        print()
        print( '#########################################################' )
        print( 'END', self.methodname, 'PIMD' )
        print( '#########################################################' )
        print()
    
    def run_nuc_only_MC( self, Nsteps=1000, Nprint=100, disp=0.1, nmove=1 ):

        #Routine to run Monte-Carlo routine for only the nuclei
        #Random displacements done using normal modes

        #Nsteps - number of Monte Carlo steps
        #disp   - max size of random displacement of zero-frequency mode
        #nmove  - number of nuclei that are attempted to be moved at each MC step, all beads for each nuclei are moved

        #Initialize nuclear positions to zero if None specified
        if( self.nucR is None ):
            print('Automatically initializing all nuclear positions to zero')
            self.nucR = np.zeros([self.nbds,self.nnuc])

        #Error Checks
        self.nuc_only_MC_error_check( nmove )

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )

        print()
        print( '#########################################################' )
        print( 'Running Nuclear only MC Routine for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        #Calculate frequencies of normal modes of ring-polymer
        #note that these are mass-independent and are thus the same for all nuclei
        nm_freq = normal_mode.calc_normal_mode_freq( self.beta_p, self.nbds )

        #Calculate initial set of normal modes
        nm = np.zeros([self.nbds,self.nnuc])
        for i in range(self.nnuc):
            nm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

        #Initialize trial positions and normal modes
        trialR  = np.copy( self.nucR )
        trialnm = np.copy( nm )

        #Calculate energy for initial configuration
        engold = self.potential.calc_tot_PE( trialR, self.beta_p, self.mass )

        numacc = 0 #Counter to keep track of number of accepted MC moves
        for step in range( Nsteps ):

            #Print data starting with initial step
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at MC step', step, 'for nuclear only MC routine')
                self.print_nuconly_data( step, numacc )
                sys.stdout.flush()

            #Randomly pick a starting nuclei
            strt = self.rng.integers( 0 , self.nnuc )

            #Loop over nmove nuclei taking into account if strt+nmove > nnuc
            for i in ( np.arange( strt, strt+nmove ) % self.nnuc ):

                #Convert position to normal modes
                trialnm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

                #Displace zero-frequency mode
                trialnm[0,i] += self.rng.uniform(-1.0,1.0) * disp

                #Randomly pull other normal modes from gaussian distribution
                for k in range(1,self.nbds):
                    sigma   = 1.0 / np.sqrt( self.beta_p * self.mass[i] * nm_freq[k]**2 )
                    trialnm[k,i] = self.rng.normal( scale = sigma )

                #Back convert normal modes to real-space positions
                trialR[:,i] = normal_mode.normal_mode_to_real( trialnm[:,i] )

            #Calculate energy of trial position
            engnew = self.potential.calc_tot_PE( trialR, self.beta_p, self.mass )

            #Check acceptance condition
            if( self.rng.random() < np.exp( -self.beta_p * ( engnew - engold ) ) ):
                #accept new configuration
                numacc += 1
                self.nucR = np.copy(trialR)
                nm        = np.copy(trialnm)
                engold    = engnew
            else:
                #reject new configuration and reset trial positions to old position
                trialR  = np.copy( self.nucR )
                trialnm = np.copy( nm )

        #Print data at final step regardless of Nprint
        print('Writing data at MC step', step+1, 'for nuclear only MC routine')
        self.print_nuconly_data( step+1, numacc )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()

        print()
        print( '#########################################################' )
        print( 'End Nuclear only MC Routine' )
        print( '#########################################################' )
        print()

    #####################################################################

    def get_nucP_MB( self, beta = None ):

        #Obtain nuclear momentum from Maxwell-Boltzmann distribution at beta_p
        #distribution defined as e^(-1/2*x^2/sigma^2) so sigma=sqrt(mass/beta)
        self.nucP = np.zeros([self.nbds,self.nnuc])
        
        if (beta is None):
            for i in range( self.nnuc ):

                mass = self.mass[i]
                sigma = np.sqrt( mass / self.beta_p )

                self.nucP[:,i] = self.rng.normal( 0.0, sigma, self.nbds )
            
        
        else:
            beta_p = beta / self.nbds
            for i in range( self.nnuc ):

                mass = self.mass[i]
                sigma = np.sqrt( mass / beta_p )

                self.nucP[:,i] = self.rng.normal( 0.0, sigma, self.nbds )
                
    #####################################################################

    def calc_nucR_com( self ):

        #Calculate the center of mass of each nuclear ring-polymer

        return( np.sum( self.nucR, axis=0 ) / self.nbds )

    #####################################################################

    def calc_nucP_com( self ):

        #Calculate the momentum of the center of mass of each nuclear ring-polymer

        return( np.sum( self.nucP, axis=0 ) / self.nbds )

    #####################################################################

    def calc_wigner_estimator( self ):

        #Calculate poulation of each electronic state using the Wigner Estimator
        #See Duke and Ananth JPC Lett 2015

        fctr = 2.0**(self.nstates+1) * np.exp( - np.sum( self.mapR**2 + self.mapP**2, 1 ) )
        pop  = np.sum( fctr[:,np.newaxis] * ( self.mapR**2 + self.mapP**2 - 0.5 ), 0 ) / self.nbds

        return pop

    #####################################################################
       
    def calc_semiclass_estimator( self ):

        #Calculate poulation of each electronic state using the semi-classical Estimator
        #See Chowdhury and Huo JCP 2019

        pop = np.sum( 0.5*( self.mapR**2 + self.mapP**2 - 1.0 ), 0 ) / self.nbds

        return pop

    #####################################################################

    def calc_Q_array( self ):
        
        #Calculate the traceless operator Q in each bead and electronic state
        #See Saller, Kelly and Richardson Faraday Discussions 2020
        Q = np.zeros([self.nbds, self.nstates])
        
        for i in range(self.nstates):
            
            Q[:,i] = 0.5 * ( self.nstates * ( self.mapR[:, i]**2 + self.mapP[:, i]**2 ) - np.sum( self.mapR**2 + self.mapP**2, axis = 1 ) )

        return Q #output an array sized [nbds, nstates]
    
    def calc_semi_array( self ):
        
        #Calculate the sum of the squares for each electronic states
        #directly related to the calculation of semi-classical population estimators
        
        return self.mapR**2 + self.mapP**2 #output an array sized [nbds, nstates]

    #####################################################################

    def calc_Q_array_sb( self ):
        
        #Calculate the traceless operator Q in each bead and electronic state if there is only one bead or LSC-IVR
        #See Saller, Kelly and Richardson Faraday Discussions 2020
            
        Q = 0.5 * ( self.nstates * ( self.mapR**2 + self.mapP**2 ) - np.sum( self.mapR**2 + self.mapP**2 ) )

        return Q #output an array sized [nstates]

    #####################################################################

    def calc_phi_fcn( self ):

        #calculate the Gaussian-like prefactor resulting from the Wigner transform on the projected SEO operaters
        #See saller, Kelly and Richardson Faraday Discussion 2020

        phi = 2 ** (self.nstates + 2) * np.exp( np.sum( - self.mapR**2 - self.mapP**2, axis = 1 )) 

        return phi #an array sized [nbds]

    #####################################################################

    def calc_phi_fcn_sb( self ):

        #calculate the Gaussian-like prefactor resulting from the Wigner transform on the projected SEO operaters
        #only for single bead case or LSC-IVR
        #See saller, Kelly and Richardson Faraday Discussion 2020

        phi = 2 ** (self.nstates + 2) * np.exp( np.sum( - self.mapR**2 - self.mapP**2 )) 

        return phi #output a number

    #####################################################################
 
    def init_map_wigner_sampling(self):

        #Initialize mapping variables by pulled from phi distribution
        #phi = 2^(L+2) exp[-\sum_i (x_i^2 + p_i^2)]
        #No state specificity
        #See Geva JCTC 2020

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using Wigner sampling' )
        print( '#########################################################' )
        print()

        self.mapR = self.rng.normal( 0.0, np.sqrt(0.5), [ self.nbds, self.nstates ] )
        self.mapP = self.rng.normal( 0.0, np.sqrt(0.5), [ self.nbds, self.nstates ] )

    #####################################################################

    def init_map_wigner_estimator( self, occstate=None ):

        #Initialize mapping variables to correspond to electronic population
        #on a single state using the wigner estimator
        #See Duke and Ananth JPC Lett 2015

        if( occstate == None ):
            print('ERROR: The occupied electronic state was not specified')
            exit()

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using Wigner Estimator' )
        print( 'System is initialized such that electronic state', occstate, 'is occupied and all others are unoccupied' )
        print( 'IMPORTANT: Indexing for electronic states starts at 0' )
        print( '#########################################################' )
        print()

        #Initialize array of mapping variables
        self.mapR = np.zeros( [ self.nbds, self.nstates ] )
        self.mapP = np.zeros( [ self.nbds, self.nstates ] )

        #Value of radius of circle to sample mapping variables for the unoccupied and occupied states
        r_unocc = np.sqrt(1.0/2.0)
        r_occ   = np.sqrt( 0.5 - scp.lambertw( -1.0 / 2**(1+self.nstates) * np.exp(self.nstates/2), k=-1 ).real )

        for i in range(self.nstates):
            if( i == occstate ):
                r = r_occ
            else:
                r = r_unocc

            angle = 2 * np.pi * self.rng.random(self.nbds)

            self.mapR[:,i] = r * np.cos(angle)
            self.mapP[:,i] = r * np.sin(angle)

    #####################################################################

    def init_map_semiclass_estimator( self, occstate=None ):

        #Initialize mapping variables to correspond to electronic population
        #on a single state using the semi-classical estimator
        #See Chowdhury and Huo JCP 2019

        if( occstate == None ):
            print('ERROR: The occupied electronic state was not specified')
            exit()

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using Semi-Classical Estimator' )
        print( 'System is initialized such that electronic state', occstate, 'is occupied and all others are unoccupied' )
        print( 'IMPORTANT: Indexing for electronic states starts at 0' )
        print( '#########################################################' )
        print()

        #Initialize array of mapping variables
        self.mapR = np.zeros( [ self.nbds, self.nstates ] )
        self.mapP = np.zeros( [ self.nbds, self.nstates ] )

        #Value of radius of circle to sample mapping variables for the unoccupied and occupied states
        r_unocc = 1.0
        r_occ   = np.sqrt(3.0)

        for i in range(self.nstates):
            if( i == occstate ):
                r = r_occ
            else:
                r = r_unocc

            angle = 2 * np.pi * self.rng.random(self.nbds)

            self.mapR[:,i] = r * np.cos(angle)
            self.mapP[:,i] = r * np.sin(angle)

    #####################################################################

    def init_map_restr_semiclass_estimator( self, occstate=None ):

        #Initialize mapping variables to correspond to electronic population
        #on a single state using the restricted semi-classical estimator
        #See Chowdhury and Huo JCP 2019
        #to restrict the off-diagonal coupling 0, theta's are different by pi/2 between the occupied and unoccupied states

        if( occstate == None ):
            print('ERROR: The occupied electronic state was not specified')
            exit()

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using restricted Semi-Classical Estimator' )
        print( 'System is initialized such that electronic state', occstate, 'is occupied and all others are unoccupied' )
        print( 'IMPORTANT: Indexing for electronic states starts at 0' )
        print( '#########################################################' )
        print()

        #Initialize array of mapping variables
        self.mapR = np.zeros( [ self.nbds, self.nstates ] )
        self.mapP = np.zeros( [ self.nbds, self.nstates ] )

        #Value of radius of circle to sample mapping variables for the unoccupied and occupied states
        r_unocc = 1.0
        r_occ   = np.sqrt(3.0)

        angle = 2 * np.pi * self.rng.random(self.nbds)

        for i in range(self.nstates):
            if( i == occstate ):
                r = r_occ            
                self.mapR[:,i] = r * np.cos(angle)
                self.mapP[:,i] = r * np.sin(angle)

            else:
                r = r_unocc
                self.mapR[:,i] = r * np.cos(angle - np.pi/2)
                self.mapP[:,i] = r * np.sin(angle - np.pi/2)
            
    #####################################################################

    def print_PIMD_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy
        engpe = self.potential.calc_tot_PE( self.nucR, self.beta_p, self.mass )

        #Calculate Nuclear Kinetic Energy
        engke = self.potential.calc_nuc_KE( self.nucP, self.mass )

        #Calculate total energy
        etot = engpe + engke

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        ######## PRINT OUT EVERYTHING #######
        output     = np.zeros(4+self.nnuc)
        output[0]  = current_time
        output[1]  = etot
        output[2]  = engke
        output[3]  = engpe
        output[4:] = nucR_com

        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #Print out positions and momenta of nuclei
        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, current_time ).reshape(1, self.nucR.size+1), fmt_str )
        np.savetxt( self.file_nucP, np.insert( self.nucP.flatten(), 0, current_time ).reshape(1, self.nucP.size+1), fmt_str )
        self.file_nucR.flush()
        self.file_nucP.flush()

    #####################################################################

    def print_nuconly_data( self, step, numacc ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy
        engpe = self.potential.calc_tot_PE( self.nucR, self.beta_p, self.mass )

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(3+self.nnuc)
        output[0] = step
        output[1] = engpe
        if( step == 0 ):
            output[2] = 1.0
        else:
            output[2] = numacc/step
        output[3:] = nucR_com

        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #Print out position of nuclei
        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, step ).reshape(1, self.nucR.size+1), fmt_str )
        self.file_nucR.flush()

    #####################################################################

    @abstractmethod
    def get_timederivs( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_timederiv_nucR( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_timederiv_nucP( self, intRP_bool=True ):
        #Top-level subroutine to calculate the time-derivative of the nuclear momenta for each bead
        #Only calculates contribution from internal ring-polyer mode and state-indepndent terms

        #Force associated with state-independent portion of potential
        d_nucP = self.potential.calc_state_indep_force( self.nucR )

        #Force associated with harmonic springs between beads
        if( intRP_bool ):
            d_nucP += self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )

        return d_nucP

    #####################################################################

    @abstractmethod
    def get_timederiv_mapR( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_timederiv_mapP( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_2nd_timederiv_mapR( self, d_mapP ):
        pass

    #####################################################################

    @abstractmethod
    def get_sampling_eng( self ):
        pass
        #Subroutine to calculate energy to use in full MC routine
        #Should not include nuclear kinetic energy
        #If there are pre-exponential terms (like in NRPMD) need to add
        #them into the exponential

    #####################################################################

    @abstractmethod
    def print_data( self, step ):
        pass

    #####################################################################

    def init_error_check( self ):

        #Input Error Checks
        if( self.nucR is not None and self.nucR.shape != (self.nbds, self.nnuc) ):
            print('ERROR: Size of nuclear position array doesnt match bead number or number of nuclei')
            exit()

        if( self.nucP is not None and self.nucP.shape != (self.nbds, self.nnuc) ):
            print('ERROR: Size of nuclear momentum array doesnt match bead number or number of nuclei')
            exit()

        if( self.mapR is not None and self.mapR.shape != (self.nbds, self.nstates) ):
            print('ERROR: Size of mapping position array doesnt match bead number or number of states')
            exit()

        if( self.mapP is not None and self.mapP.shape != (self.nbds, self.nstates) ):
            print('ERROR: Size of mapping momentum array doesnt match bead number or number of states')
            exit()

        if( self.mass.shape[0] != self.nnuc ):
            print('ERROR: Size of nuclear mass array doesnt match number of nuclei')
            exit()

    #####################################################################

    def dynam_error_check( self, Nsteps, delt, intype ):

        #Error Checks when trying to run dynamics calculation

        if( Nsteps == None or delt == None or intype == None ):
            print('ERROR: The number of steps (Nsteps), time-step (delt), or integrator type (intype) was not specified')
            exit()

        if( self.nucR is None  ):
            print( 'ERROR: The position of the nuclei was not defined' )
            exit()

        if( self.nucP is None  ):
            print( 'ERROR: The momentum of the nuclei was not defined' )
            exit()

        if( self.mapR is None ):
            print( 'ERROR: The position of the mapping variables was not defined' )
            exit()

        if( self.mapP is None  ):
            print( 'ERROR: The momentum of the mapping variables was not defined' )
            exit()

        self.init_error_check()

    #####################################################################

    def MC_error_check( self, nmove ):

        #Error Checks when trying to run full MC

        if( nmove > self.nnuc ):
            print('ERROR: Trying to move more nuclei than there are in the system during nuclear only MC')
            exit()

        self.init_error_check()

    #####################################################################

    def nuc_only_MC_error_check( self, nmove ):

        #Error Checks when trying to run nuclear only MC

        if( nmove > self.nnuc ):
            print('ERROR: Trying to move more nuclei than there are in the system during nuclear only MC')
            exit()

        if( self.potential.potname[:12] != "Nuclear Only" ):
            print('ERROR: Did not specify a nuclear only potential during nuclear only MC')
            exit()

        self.init_error_check()

    #########################################Define parent class for all mapping-rpmd methods

import numpy as np
import scipy.special as scp
from abc import ABC, abstractmethod
from scipy.special import jv, struve
import utils
import potential
import integrator
import normal_mode
import sys
import math

####### PARENT CLASS FOR MAPPING-RPMD METHODS######

class map_rpmd(ABC):

    #####################################################################

    @abstractmethod
    def __init__( self, methodname, nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP, langevin=None ):

        #Initialize all variables
        #This is an abstractmethod so all default values are set at sub-class level

        print() #Print blank line for readability

        self.methodname = methodname #string defining the sub-class method name
        self.potype     = potype     #the type of potential

        self.nstates = nstates   #number of electronic states
        self.nnuc    = nnuc      #number of nuclear modes
        self.nbds    = nbds      #number of ring polymer beads
        self.beta    = beta      #inverse temperature
        self.beta_p  = beta/nbds #inverse temperature at higher value due to bead-number
        self.mass    = mass      #mass of nuclear modes - 1d array size of Nnuc b/c mass of all beads for a given nuclei is the same
        self.mapR    = mapR      #position of mapping variables, dimension Nbds x Nstates
        self.mapP    = mapP      #momentum of mapping variables, dimension Nbds x Nstates
        self.nucR    = nucR      #position of nuclear modes, dimension of Nbds x Nnuc
        self.nucP    = nucP      #momentum of nuclear modes, dimension of Nbds x Nnuc
        self.langevin=langevin   #langevin fraction (damping) coefficient, dimension of Nnuc
 
        #Initialize instance of random number generator
        self.rng = np.random.default_rng()

        #Input error check
        if (self.methodname != 'sb-NRPMD'):
            self.init_error_check()

        #Define potential
        self.potential = potential.set_potential( potype, potparams, nstates, nnuc, nbds )

        #Define variables to be used as output files
        self.file_output = None
        self.file_nucR   = None
        self.file_nucP   = None
        self.file_mapR   = None
        self.file_mapP   = None
        self.file_Q      = None
        self.file_phi    = None

    #####################################################################

    def run_dynamics( self, Nsteps=None, Nprint=100, delt=None, intype=None, init_time=0.0, small_dt_ratio=1 ):

        #Top-level routine to run dynamics

        #Number of decimal places when printing current time
        #modf splits the floating number into integer and decimal components
        #converting to string, taking the length, and subtracting 2 (for the 0.) gives us the length of just the decimal component
        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

        #Error checks
        self.dynam_error_check( Nsteps, delt, intype )

        #Initialize the integrator
        self.integ = integrator.integrator( self, delt, intype, small_dt_ratio )

        #Automatically initialize nuclear momentum from MB distribution if none have been specified
        if( self.nucP is None ):
            print('Automatically initializing nuclear momentum to Maxwell-Boltzmann distribution at beta_p = ', self.beta_p*self.nbds ,' / ',self.nbds)
            self.get_nucP_MB()

        print()
        print( '#########################################################' )
        print( 'Running', self.methodname, 'Dynamics for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_nucP   = open( 'nucP.dat', 'w' )
        self.file_mapR   = open( 'mapR.dat','w' )
        self.file_mapP   = open( 'mapP.dat', 'w' )
        self.file_Q      = open( 'Q.dat', 'w')
        self.file_phi    = open( 'phi.dat', 'w')
        self.file_semi   = open( 'mvsq.dat', 'w')

        current_time = init_time
        step = 0

        if self.langevin != None:
            time_pts = np.arange(init_time, (Nsteps+1)*delt, delt )
            self.friction_kernel = np.zeros((Nsteps+1, self.nbds, self.nnuc))
            self.nucP_arr = np.zeros((Nsteps+1, self.nbds, self.nnuc))
            self.nucP_arr[0] = self.nucP
            FrictionKernel_1d = np.zeros((Nsteps+1, self.nbds))

            nm_freq = normal_mode.calc_normal_mode_freq( self.beta_p, self.nbds )
            
            for i in range(self.nbds):
                FrictionKernel_1d[:,1] = -self.langevin*nm_freq[i] + self.langevin*nm_freq[i]**2*time_pts*jv(0, nm_freq[i]*time_pts)*(1-np.pi/2*struve(1, nm_freq[i]*time_pts)) - self.langevin*nm_freq[i]*jv(1,nm_freq[i]*time_pts)*(1-np.pi/2*nm_freq*time_pts*struve(0, nm_freq[i]*time_pts))
            self.friction_kernel = FrictionKernel_1d[:,:,np.newaxis]

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
        print('Writing data at step ', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for', self.methodname, 'Dynamics calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_nucP.close()
        self.file_mapR.close()
        self.file_mapP.close()
        self.file_Q.close()
        self.file_phi.close()
        self.file_semi.close()

        print()
        print( '#########################################################' )
        print( 'END', self.methodname, 'Dynamics' )
        print( '#########################################################' )
        print()

    #####################################################################

    def run_dynamics_massN( self, Nsteps=None, Nprint=100, delt=None, intype=None, init_time=0.0, small_dt_ratio=1 ):
        
        #Routine to run the dynamics with the partition funciton e^(\beta H) instead of e^(\beta_N H_N)

        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

    def run_MC( self, Nsteps=1000, Nprint=100, disp_nuc=0.1, disp_map=0.1, nmove=1, nm_bool=False, resamp=None, freeze_nuc=False ):

        #Routine to run Monte-Carlo for mapping variables and the nuclei
        #Random displacements done in real-space for mapping variables
        #Option for random displacements using normal modes or real-space for nuclei

        #Nsteps   - number of Monte Carlo steps
        #disp_nuc - max size of random displacement for nuclear positions or zero frequency mode
        #disp_map - max size of random displacement for mapping variables
        #nmove    - number of nuclei that are attempted to be moved at each MC step, all beads for each nuclei are moved
        #resamp   - number of MC steps between resampling the electronic variables from a normal distribution
        #freeze_nuc - freeze nuclear variables for entire mc run

        #Initialize nuclear positions to zero if None specified
        if( self.nucR is None ):
            print('Automatically initializing all nuclear positions to zero')
            self.nucR = np.zeros([self.nbds,self.nnuc])

        #Initilize mapping variables arbitrarily to state zero if None specified
        if( self.mapR is None or self.mapP is None ):
            print('Automatically initializing mapping variables to state zero')
            self.init_map_semiclass_estimator( occstate=0 )

        #Set resamp to Nsteps + 1 so the MC never resamples if None is specified
        if( resamp is None ): resamp = Nsteps + 1
        #Error Checks
        self.MC_error_check( nmove )

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_mapR   = open( 'mapR.dat','w' )
        self.file_mapP   = open( 'mapP.dat', 'w' )

        print()
        print( '#########################################################' )
        print( 'Running', self.methodname, 'MC Routine for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        if( nm_bool ):
            #Calculate frequencies of normal modes of ring-polymer
            #note that these are mass-independent and are thus the same for all nuclei
            nm_freq = normal_mode.calc_normal_mode_freq( self.beta_p, self.nbds )

            #Calculate initial set of normal modes
            nm = np.zeros([self.nbds,self.nnuc])
            for i in range(self.nnuc):
                nm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

        #Initialize original positions
        orig_nucR  = np.copy( self.nucR )
        orig_mapR  = np.copy( self.mapR )
        orig_mapP  = np.copy( self.mapP )
        if( nm_bool ):
            orignm = np.copy( nm )

        #Calculate energy for initial configuration
        engold = self.get_sampling_eng()

        numacc = 0 #Counter to keep track of number of accepted MC moves
        for step in range( Nsteps ):

            #Print data starting with initial step
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at MC step', step, 'for', self.methodname, 'MC routine')
                self.print_MC_data( step, numacc )
                sys.stdout.flush()

            #check if resampling step
            if( np.mod(step + 1, resamp) == 0 ):
                self.mapR = self.rng.normal(scale=1 / np.sqrt(2), size=self.mapR.shape)
                self.mapP = self.rng.normal(scale=1 / np.sqrt(2), size=self.mapP.shape)
            else:
                #### Move Nuclei ###
                if not freeze_nuc:
                    #Randomly pick a starting nuclei
                    strt = self.rng.integers( 0 , self.nnuc )

                    #Loop over nmove nuclei taking into account if strt+nmove > nnuc
                    for i in ( np.arange( strt, strt+nmove ) % self.nnuc ):

                        if( nm_bool ):
                            #Convert position to normal modes
                            nm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

                            #Displace zero-frequency mode
                            nm[0,i] += self.rng.uniform(-1.0,1.0) * disp_nuc

                            #Randomly pull other normal modes from gaussian distribution
                            for k in range(1,self.nbds):
                                sigma   = 1.0 / np.sqrt( self.beta_p * self.mass[i] * nm_freq[k]**2 )
                                nm[k,i] = self.rng.normal( scale = sigma )

                            #Back convert normal modes to real-space positions
                            self.nucR[:,i] = normal_mode.normal_mode_to_real( nm[:,i] )
                        else:
                            self.nucR[:,i] += self.rng.uniform(-1.0, 1.0, self.nbds ) * disp_nuc
                #################################

                #Move Mapping Variables
                if self.methodname == 'sb-NRPMD':
                    self.mapR += self.rng.uniform(-1.0, 1.0, self.nstates ) * disp_map
                    self.mapP += self.rng.uniform(-1.0, 1.0, self.nstates ) * disp_map
                else:
                    self.mapR += self.rng.uniform(-1.0, 1.0, (self.nbds, self.nstates) ) * disp_map
                    self.mapP += self.rng.uniform(-1.0, 1.0, (self.nbds, self.nstates) ) * disp_map

            #Calculate energy of trial position
            engnew = self.get_sampling_eng()
            d_eng = engnew - engold

            #Check acceptance condition
            if d_eng < 0:
                #accept new configuration
                numacc += 1
                orig_nucR = np.copy(self.nucR)
                orig_mapR = np.copy(self.mapR)
                orig_mapP = np.copy(self.mapP)
                if( nm_bool ):
                    orig_nm = np.copy(nm)
                engold    = engnew
            else:
                if self.methodname == 'sb-NRPMD': acc_cond = np.exp( -self.beta * ( d_eng ) )
                else: acc_cond = np.exp( -self.beta_p * ( d_eng ) )
                if( self.rng.random() < acc_cond ):
                    #accept new configuration
                    numacc += 1
                    orig_nucR = np.copy(self.nucR)
                    orig_mapR = np.copy(self.mapR)
                    orig_mapP = np.copy(self.mapP)
                    if( nm_bool ):
                        orig_nm = np.copy(nm)
                    engold    = engnew
                else:
                    #reject new configuration and reset positions to original position
                    self.nucR  = np.copy( orig_nucR )
                    self.mapR  = np.copy( orig_mapR )
                    self.mapP  = np.copy( orig_mapP )
                    if( nm_bool ):
                        nm = np.copy( orig_nm )

        #Print data at final step regardless of Nprint
        print('Writing data at MC step', step+1, 'for', self.methodname, 'MC routine')
        self.print_MC_data( step+1, numacc )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_mapR.close()
        self.file_mapP.close()

        print()
        print( '#########################################################' )
        print( 'End',self.methodname, 'MC Routine' )
        print( '#########################################################' )
        print()

    #####################################################################

    def run_PIMD( self, Nsteps=None, resample=None, intype=None, Nprint=100, delt=None, init_time=0.0, small_dt_ratio=1 ):

        #routine to run PIMC only for the nuclei with vv-type integrators
        #resample - the number of steps that equilibrate the temperature (NVT MD)
        #small_dt_ratio - the ratio of electronic time-step with nuclear one. include this only b/c we use the integral package

        #Set resamp to Nsteps + 1 so the MC never resamples if None is specified
        if( resample is None ): resample = Nsteps + 1

        #error check for the intype
        if(intype != 'vv' and intype != 'analyt' and intype != 'cayley'):
            print('ERROR: the intype should be one of these: vv, analyt or cayley. Now it is', intype)
            exit()

        #Initialize the integrator
        self.integ = integrator.integrator( self, delt, intype, small_dt_ratio )

        #converting to string, taking the length, and subtracting 2 (for the 0.) gives us the length of just the decimal component
        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

        #Automatically initialize nuclear momentum from MB distribution if none have been specified
        if( self.nucP is None ):
            print('Automatically initializing nuclear momenta to Maxwell-Boltzmann distribution at beta_p =', self.beta_p*self.nbds ,'/',self.nbds)
            self.get_nucP_MB()

        #Initialize nuclear positions to zero if None specified
        if( self.nucR is None ):
            print('Automatically initializing all nuclear positions to zero')
            self.nucR = np.zeros([self.nbds, self.nnuc])

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_nucP   = open( 'nucP.dat','w' )

        print()
        print( '#########################################################' )
        print('running equilibrium PIMD for', Nsteps, 'at beta_p =', self.beta_p*self.nbds, '/',self.nbds)
        print( '#########################################################' )
        print()

        current_time = init_time
        step = 0

        for step in range( Nsteps ):

            #resample the nucP if we need to
            if( np.mod( step+1, resample ) == 0 ):
                print('resample the bead momentum to run within an NVT ensemble')
                self.get_nucP_MB()

            #Print data starting with initial time
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at step', step, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for PIMD calculation')
                self.print_PIMD_data( current_time )
                sys.stdout.flush()

            #Outer loop to integrate EOM using velocity-verlet like algorithms
            #This includes pengfei's implementation, and the analtyical and cayley modification of it

            #If initial step of dynamics need to initialize derivative of nuclear momentum (aka the force on the nuclei)
            #NOTE: moving forward these calls may need to be generalized to allow for other types of methods
            if( step == 0 ):
                if( intype == 'vv' ):
                    self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )
                else:
                    self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR )

            #Update nuclear momentum by 1/2 a time-step
            self.integ.update_vv_nucP( self )

            #Update nuclear position for full time-step
            if( intype == 'vv' ):
                self.integ.update_vv_nucR( self )
            elif( intype == 'analyt' ):
                self.integ.update_analyt_nucR( self )
            elif( intype == 'cayley' ):
                self.integ.update_cayley_nucR( self )

            #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
            #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
            if( intype == 'vv' ):
                self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR ) + self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )
            else:
                self.integ.d_nucP_for_vv = self.potential.calc_external_force( self.nucR )

            #Update nuclear momentum to full time-step
            self.integ.update_vv_nucP( self )

            #Increase current time
            current_time = init_time + delt * (step+1)

        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for PIMD Dynamics calculation')
        self.print_PIMD_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_nucP.close()

        print()
        print( '#########################################################' )
        print( 'END', self.methodname, 'PIMD' )
        print( '#########################################################' )
        print()
    
    def run_nuc_only_MC( self, Nsteps=1000, Nprint=100, disp=0.1, nmove=1 ):

        #Routine to run Monte-Carlo routine for only the nuclei
        #Random displacements done using normal modes

        #Nsteps - number of Monte Carlo steps
        #disp   - max size of random displacement of zero-frequency mode
        #nmove  - number of nuclei that are attempted to be moved at each MC step, all beads for each nuclei are moved

        #Initialize nuclear positions to zero if None specified
        if( self.nucR is None ):
            print('Automatically initializing all nuclear positions to zero')
            self.nucR = np.zeros([self.nbds,self.nnuc])

        #Error Checks
        self.nuc_only_MC_error_check( nmove )

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )

        print()
        print( '#########################################################' )
        print( 'Running Nuclear only MC Routine for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        #Calculate frequencies of normal modes of ring-polymer
        #note that these are mass-independent and are thus the same for all nuclei
        nm_freq = normal_mode.calc_normal_mode_freq( self.beta_p, self.nbds )

        #Calculate initial set of normal modes
        nm = np.zeros([self.nbds,self.nnuc])
        for i in range(self.nnuc):
            nm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

        #Initialize trial positions and normal modes
        trialR  = np.copy( self.nucR )
        trialnm = np.copy( nm )

        #Calculate energy for initial configuration
        engold = self.potential.calc_tot_PE( trialR, self.beta_p, self.mass )

        numacc = 0 #Counter to keep track of number of accepted MC moves
        for step in range( Nsteps ):

            #Print data starting with initial step
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at MC step', step, 'for nuclear only MC routine')
                self.print_nuconly_data( step, numacc )
                sys.stdout.flush()

            #Randomly pick a starting nuclei
            strt = self.rng.integers( 0 , self.nnuc )

            #Loop over nmove nuclei taking into account if strt+nmove > nnuc
            for i in ( np.arange( strt, strt+nmove ) % self.nnuc ):

                #Convert position to normal modes
                trialnm[:,i] = normal_mode.real_to_normal_mode( self.nucR[:,i] )

                #Displace zero-frequency mode
                trialnm[0,i] += self.rng.uniform(-1.0,1.0) * disp

                #Randomly pull other normal modes from gaussian distribution
                for k in range(1,self.nbds):
                    sigma   = 1.0 / np.sqrt( self.beta_p * self.mass[i] * nm_freq[k]**2 )
                    trialnm[k,i] = self.rng.normal( scale = sigma )

                #Back convert normal modes to real-space positions
                trialR[:,i] = normal_mode.normal_mode_to_real( trialnm[:,i] )

            #Calculate energy of trial position
            engnew = self.potential.calc_tot_PE( trialR, self.beta_p, self.mass )

            #Check acceptance condition
            if( self.rng.random() < np.exp( -self.beta_p * ( engnew - engold ) ) ):
                #accept new configuration
                numacc += 1
                self.nucR = np.copy(trialR)
                nm        = np.copy(trialnm)
                engold    = engnew
            else:
                #reject new configuration and reset trial positions to old position
                trialR  = np.copy( self.nucR )
                trialnm = np.copy( nm )

        #Print data at final step regardless of Nprint
        print('Writing data at MC step', step+1, 'for nuclear only MC routine')
        self.print_nuconly_data( step+1, numacc )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()

        print()
        print( '#########################################################' )
        print( 'End Nuclear only MC Routine' )
        print( '#########################################################' )
        print()

    #####################################################################

    def get_nucP_MB( self, beta = None ):

        #Obtain nuclear momentum from Maxwell-Boltzmann distribution at beta_p
        #distribution defined as e^(-1/2*x^2/sigma^2) so sigma=sqrt(mass/beta)
        self.nucP = np.zeros([self.nbds,self.nnuc])
        
        if (beta is None):
            for i in range( self.nnuc ):

                mass = self.mass[i]
                sigma = np.sqrt( mass / self.beta_p )

                self.nucP[:,i] = self.rng.normal( 0.0, sigma, self.nbds )
            
        
        else:
            beta_p = beta / self.nbds
            for i in range( self.nnuc ):

                mass = self.mass[i]
                sigma = np.sqrt( mass / beta_p )

                self.nucP[:,i] = self.rng.normal( 0.0, sigma, self.nbds )
                
    #####################################################################

    def calc_nucR_com( self ):

        #Calculate the center of mass of each nuclear ring-polymer

        return( np.sum( self.nucR, axis=0 ) / self.nbds )

    #####################################################################

    def calc_nucP_com( self ):

        #Calculate the momentum of the center of mass of each nuclear ring-polymer

        return( np.sum( self.nucP, axis=0 ) / self.nbds )

    #####################################################################

    def calc_wigner_estimator( self ):

        #Calculate poulation of each electronic state using the Wigner Estimator
        #See Duke and Ananth JPC Lett 2015

        fctr = 2.0**(self.nstates+1) * np.exp( - np.sum( self.mapR**2 + self.mapP**2, 1 ) )
        pop  = np.sum( fctr[:,np.newaxis] * ( self.mapR**2 + self.mapP**2 - 0.5 ), 0 ) / self.nbds

        return pop

    #####################################################################
       
    def calc_semiclass_estimator( self ):

        #Calculate poulation of each electronic state using the semi-classical Estimator
        #See Chowdhury and Huo JCP 2019

        pop = np.sum( 0.5*( self.mapR**2 + self.mapP**2 - 1.0 ), 0 ) / self.nbds

        return pop

    #####################################################################

    def calc_Q_array( self ):
        
        #Calculate the traceless operator Q in each bead and electronic state
        #See Saller, Kelly and Richardson Faraday Discussions 2020
        Q = np.zeros([self.nbds, self.nstates])
        
        for i in range(self.nstates):
            
            Q[:,i] = 0.5 * ( self.nstates * ( self.mapR[:, i]**2 + self.mapP[:, i]**2 ) - np.sum( self.mapR**2 + self.mapP**2, axis = 1 ) )

        return Q #output an array sized [nbds, nstates]
    
    def calc_semi_array( self ):
        
        #Calculate the sum of the squares for each electronic states
        #directly related to the calculation of semi-classical population estimators
        
        return self.mapR**2 + self.mapP**2 #output an array sized [nbds, nstates]

    #####################################################################

    def calc_Q_array_sb( self ):
        
        #Calculate the traceless operator Q in each bead and electronic state if there is only one bead or LSC-IVR
        #See Saller, Kelly and Richardson Faraday Discussions 2020
            
        Q = 0.5 * ( self.nstates * ( self.mapR**2 + self.mapP**2 ) - np.sum( self.mapR**2 + self.mapP**2 ) )

        return Q #output an array sized [nstates]

    #####################################################################

    def calc_phi_fcn( self ):

        #calculate the Gaussian-like prefactor resulting from the Wigner transform on the projected SEO operaters
        #See saller, Kelly and Richardson Faraday Discussion 2020

        phi = 2 ** (self.nstates + 2) * np.exp( np.sum( - self.mapR**2 - self.mapP**2, axis = 1 )) 

        return phi #an array sized [nbds]

    #####################################################################

    def calc_phi_fcn_sb( self ):

        #calculate the Gaussian-like prefactor resulting from the Wigner transform on the projected SEO operaters
        #only for single bead case or LSC-IVR
        #See saller, Kelly and Richardson Faraday Discussion 2020

        phi = 2 ** (self.nstates + 2) * np.exp( np.sum( - self.mapR**2 - self.mapP**2 )) 

        return phi #output a number

    #####################################################################
 
    def init_map_wigner_sampling(self):

        #Initialize mapping variables by pulled from phi distribution
        #phi = 2^(L+2) exp[-\sum_i (x_i^2 + p_i^2)]
        #No state specificity
        #See Geva JCTC 2020

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using Wigner sampling' )
        print( '#########################################################' )
        print()

        self.mapR = self.rng.normal( 0.0, np.sqrt(0.5), [ self.nbds, self.nstates ] )
        self.mapP = self.rng.normal( 0.0, np.sqrt(0.5), [ self.nbds, self.nstates ] )

    #####################################################################

    def init_map_wigner_estimator( self, occstate=None ):

        #Initialize mapping variables to correspond to electronic population
        #on a single state using the wigner estimator
        #See Duke and Ananth JPC Lett 2015

        if( occstate == None ):
            print('ERROR: The occupied electronic state was not specified')
            exit()

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using Wigner Estimator' )
        print( 'System is initialized such that electronic state', occstate, 'is occupied and all others are unoccupied' )
        print( 'IMPORTANT: Indexing for electronic states starts at 0' )
        print( '#########################################################' )
        print()

        #Initialize array of mapping variables
        self.mapR = np.zeros( [ self.nbds, self.nstates ] )
        self.mapP = np.zeros( [ self.nbds, self.nstates ] )

        #Value of radius of circle to sample mapping variables for the unoccupied and occupied states
        r_unocc = np.sqrt(1.0/2.0)
        r_occ   = np.sqrt( 0.5 - scp.lambertw( -1.0 / 2**(1+self.nstates) * np.exp(self.nstates/2), k=-1 ).real )

        for i in range(self.nstates):
            if( i == occstate ):
                r = r_occ
            else:
                r = r_unocc

            angle = 2 * np.pi * self.rng.random(self.nbds)

            self.mapR[:,i] = r * np.cos(angle)
            self.mapP[:,i] = r * np.sin(angle)

    #####################################################################

    def init_map_semiclass_estimator( self, occstate=None ):

        #Initialize mapping variables to correspond to electronic population
        #on a single state using the semi-classical estimator
        #See Chowdhury and Huo JCP 2019

        if( occstate == None ):
            print('ERROR: The occupied electronic state was not specified')
            exit()

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using Semi-Classical Estimator' )
        print( 'System is initialized such that electronic state', occstate, 'is occupied and all others are unoccupied' )
        print( 'IMPORTANT: Indexing for electronic states starts at 0' )
        print( '#########################################################' )
        print()

        #Initialize array of mapping variables
        self.mapR = np.zeros( [ self.nbds, self.nstates ] )
        self.mapP = np.zeros( [ self.nbds, self.nstates ] )

        #Value of radius of circle to sample mapping variables for the unoccupied and occupied states
        r_unocc = 1.0
        r_occ   = np.sqrt(3.0)

        for i in range(self.nstates):
            if( i == occstate ):
                r = r_occ
            else:
                r = r_unocc

            angle = 2 * np.pi * self.rng.random(self.nbds)

            self.mapR[:,i] = r * np.cos(angle)
            self.mapP[:,i] = r * np.sin(angle)

    #####################################################################

    def init_map_restr_semiclass_estimator( self, occstate=None ):

        #Initialize mapping variables to correspond to electronic population
        #on a single state using the restricted semi-classical estimator
        #See Chowdhury and Huo JCP 2019
        #to restrict the off-diagonal coupling 0, theta's are different by pi/2 between the occupied and unoccupied states

        if( occstate == None ):
            print('ERROR: The occupied electronic state was not specified')
            exit()

        print()
        print( '#########################################################' )
        print( 'Initializing Mapping Variables using restricted Semi-Classical Estimator' )
        print( 'System is initialized such that electronic state', occstate, 'is occupied and all others are unoccupied' )
        print( 'IMPORTANT: Indexing for electronic states starts at 0' )
        print( '#########################################################' )
        print()

        #Initialize array of mapping variables
        self.mapR = np.zeros( [ self.nbds, self.nstates ] )
        self.mapP = np.zeros( [ self.nbds, self.nstates ] )

        #Value of radius of circle to sample mapping variables for the unoccupied and occupied states
        r_unocc = 1.0
        r_occ   = np.sqrt(3.0)

        angle = 2 * np.pi * self.rng.random(self.nbds)

        for i in range(self.nstates):
            if( i == occstate ):
                r = r_occ            
                self.mapR[:,i] = r * np.cos(angle)
                self.mapP[:,i] = r * np.sin(angle)

            else:
                r = r_unocc
                self.mapR[:,i] = r * np.cos(angle - np.pi/2)
                self.mapP[:,i] = r * np.sin(angle - np.pi/2)
            
    #####################################################################

    def print_PIMD_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy
        engpe = self.potential.calc_tot_PE( self.nucR, self.beta_p, self.mass )

        #Calculate Nuclear Kinetic Energy
        engke = self.potential.calc_nuc_KE( self.nucP, self.mass )

        #Calculate total energy
        etot = engpe + engke

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        ######## PRINT OUT EVERYTHING #######
        output     = np.zeros(4+self.nnuc)
        output[0]  = current_time
        output[1]  = etot
        output[2]  = engke
        output[3]  = engpe
        output[4:] = nucR_com

        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #Print out positions and momenta of nuclei
        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, current_time ).reshape(1, self.nucR.size+1), fmt_str )
        np.savetxt( self.file_nucP, np.insert( self.nucP.flatten(), 0, current_time ).reshape(1, self.nucP.size+1), fmt_str )
        self.file_nucR.flush()
        self.file_nucP.flush()

    #####################################################################

    def print_nuconly_data( self, step, numacc ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ###### CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate potential energy
        engpe = self.potential.calc_tot_PE( self.nucR, self.beta_p, self.mass )

        #Calculate the center of mass of each ring polymer
        nucR_com = self.calc_nucR_com()

        ######## PRINT OUT EVERYTHING #######
        output    = np.zeros(3+self.nnuc)
        output[0] = step
        output[1] = engpe
        if( step == 0 ):
            output[2] = 1.0
        else:
            output[2] = numacc/step
        output[3:] = nucR_com

        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        #Print out position of nuclei
        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        np.savetxt( self.file_nucR, np.insert( self.nucR.flatten(), 0, step ).reshape(1, self.nucR.size+1), fmt_str )
        self.file_nucR.flush()

    #####################################################################

    @abstractmethod
    def get_timederivs( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_timederiv_nucR( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_timederiv_nucP( self, intRP_bool=True ):
        #Top-level subroutine to calculate the time-derivative of the nuclear momenta for each bead
        #Only calculates contribution from internal ring-polyer mode and state-indepndent terms

        #Force associated with state-independent portion of potential
        d_nucP = self.potential.calc_state_indep_force( self.nucR )

        #Force associated with harmonic springs between beads
        if( intRP_bool ):
            d_nucP += self.potential.calc_rp_harm_force( self.nucR, self.beta_p, self.mass )

        return d_nucP

    #####################################################################

    @abstractmethod
    def get_timederiv_mapR( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_timederiv_mapP( self ):
        pass

    #####################################################################

    @abstractmethod
    def get_2nd_timederiv_mapR( self, d_mapP ):
        pass

    #####################################################################

    @abstractmethod
    def get_sampling_eng( self ):
        pass
        #Subroutine to calculate energy to use in full MC routine
        #Should not include nuclear kinetic energy
        #If there are pre-exponential terms (like in NRPMD) need to add
        #them into the exponential

    #####################################################################

    @abstractmethod
    def print_data( self, step ):
        pass

    #####################################################################

    def init_error_check( self ):

        #Input Error Checks
        if( self.nucR is not None and self.nucR.shape != (self.nbds, self.nnuc) ):
            print('ERROR: Size of nuclear position array doesnt match bead number or number of nuclei')
            exit()

        if( self.nucP is not None and self.nucP.shape != (self.nbds, self.nnuc) ):
            print('ERROR: Size of nuclear momentum array doesnt match bead number or number of nuclei')
            exit()

        if( self.mapR is not None and self.mapR.shape != (self.nbds, self.nstates) ):
            print('ERROR: Size of mapping position array doesnt match bead number or number of states')
            exit()

        if( self.mapP is not None and self.mapP.shape != (self.nbds, self.nstates) ):
            print('ERROR: Size of mapping momentum array doesnt match bead number or number of states')
            exit()

        if( self.mass.shape[0] != self.nnuc ):
            print('ERROR: Size of nuclear mass array doesnt match number of nuclei')
            exit()

    #####################################################################

    def dynam_error_check( self, Nsteps, delt, intype ):

        #Error Checks when trying to run dynamics calculation

        if( Nsteps == None or delt == None or intype == None ):
            print('ERROR: The number of steps (Nsteps), time-step (delt), or integrator type (intype) was not specified')
            exit()

        if( self.nucR is None  ):
            print( 'ERROR: The position of the nuclei was not defined' )
            exit()

        if( self.nucP is None  ):
            print( 'ERROR: The momentum of the nuclei was not defined' )
            exit()

        if( self.mapR is None ):
            print( 'ERROR: The position of the mapping variables was not defined' )
            exit()

        if( self.mapP is None  ):
            print( 'ERROR: The momentum of the mapping variables was not defined' )
            exit()

        self.init_error_check()

    #####################################################################

    def MC_error_check( self, nmove ):

        #Error Checks when trying to run full MC

        if( nmove > self.nnuc ):
            print('ERROR: Trying to move more nuclei than there are in the system during nuclear only MC')
            exit()

        self.init_error_check()

    #####################################################################

    def nuc_only_MC_error_check( self, nmove ):

        #Error Checks when trying to run nuclear only MC

        if( nmove > self.nnuc ):
            print('ERROR: Trying to move more nuclei than there are in the system during nuclear only MC')
            exit()

        if( self.potential.potname[:12] != "Nuclear Only" ):
            print('ERROR: Did not specify a nuclear only potential during nuclear only MC')
            exit()

        self.init_error_check()

    #####################################################################
#############################
