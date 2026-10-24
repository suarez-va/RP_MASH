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
                 spinmap_bool=False, centroid_bool=False, bead_bool=False, functional_param=None, langevin=None, langevin_params=None, seed=None, init_memP=None, init_Ffluci=None, jump_index=None, full_jump=True, verbose=True):

        super().__init__( 'RP-MASH', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP, langevin, langevin_params, seed )
        
        self.spin_map = spinmap_bool # Boolean that decides if we use spin mapping variables
        self.centroid_bool = centroid_bool # Boolean that decides if we use the centroid of nuclei to be coupled with electronic states
        self.bead_bool = bead_bool #Boolean that decides if if we use the bead average potential of nuclei that is coupled with electronic states
        self.functional_param = functional_param # The function parameter if the Gaussian function is used for the momentum rescaling
        #Quantum jumps happen at these predetermined step indices: k jumps after k time-steps, at
        #t = init_time + k*delt (1 <= k <= Nsteps). Empty/None (the default) means no jumps.
        self.jump_index = sorted( set( int(k) for k in ( jump_index if jump_index is not None else [] ) ) )
        self.full_jump = full_jump # True -> a jump resamples the whole spin sphere; False -> resample only within the active adiabat
        self.verbose = verbose # False -> no per-print/per-jump/spin-sampling progress lines on stdout (output files unchanged)

        #Quantum-jump weights are not live until init_jump_weights runs at the start of run_dynamics.
        #sample_fac is the sampling factor of the most recent init_map_spin: 2 for a full-sphere
        #sample, 1 for a hemisphere sample or for mapS passed in through the constructor.
        self.W_PP = self.W_CP = self.W_PC = self.W_CC = None
        self.sample_fac = 1.0

        #Time-dependent weights, frozen at the most recent jump t_n: W_*_t hold W^(n-1)(t_n) (already
        #times the sampling factor of jump n), mapS_prev holds the bead-mean S^(n-1)(t_n) right
        #before that jump's resample and mapS_post the bead-mean S^(n)(t_n) right after it. All stay
        #None until the first jump.
        self.W_PP_t = self.W_CP_t = self.W_PC_t = self.W_CC_t = None
        self.mapS_prev = None
        self.mapS_post = None

        # GLE restart state (None/0 => fresh start). Consumed by integrator.__init__ via getattr.
        self.init_memP   = init_memP
        self.init_Ffluci = init_Ffluci

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

        if (len(self.jump_index) > 0):
            if (spinmap_bool != True):
                print('ERROR: jump_index requires the spin mapping variables (spinmap_bool=True)')
                exit()
            if (self.jump_index[0] < 1):
                print('ERROR: jump_index entries must be >= 1 (k jumps at t = init_time + k*delt), got', self.jump_index)
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

        #Quantum jumps fire at the END of the time-steps in jump_steps (step+1 == k, see the loop)
        if( len(self.jump_index) > 0 and self.jump_index[-1] > Nsteps ):
            print('ERROR: jump_index entries must be <= Nsteps =', Nsteps, ', got', self.jump_index)
            exit()
        jump_steps = set( self.jump_index )

        #Quantum-jump log and weights. Written for EVERY spin-mapping run, with or without jumps, so
        #the initial weights W^(0) (including the sampling factor c0) are always part of the output.
        self.file_jump = None
        self.file_weights = None
        if( self.spin_map ):
            self.file_jump = open( 'jump.dat', 'w' )

            #Quantum-jump weights W_AB. One row per SAMPLE event: row 0 is the initial sample at
            #t=init_time, row n is jump n. Row 0 is always written, so a trajectory with zero jumps
            #still yields a one-row file rather than an empty one that np.loadtxt cannot read.
            self.init_jump_weights()
            self.file_weights = open( 'weights.dat', 'w' )
            self.file_weights.write( '#%19s%20s%20s%20s%20s%20s'
                                     % ('t', 'Sz_n', 'W_PP', 'W_CP', 'W_PC', 'W_CC')
                                     + '%20s%20s%20s%20s%20s%20s%20s%20s%20s\n'
                                     % ('Sx_prev', 'Sy_prev', 'Sz_prev', 'W_PP_t', 'W_CP_t', 'W_PC_t', 'W_CC_t',
                                        'Sx_post', 'Sy_post') )
            self.write_jump_weights( init_time, self.Sz0 )

        #For generalized Langevin, save the (time-independent) friction memory kernel memK once.
        #Rows = ring-polymer normal modes, columns = lag index j (memK[:,j] = K_k(j*delt)).
        if( self.langevin == 'generalized' ):
            np.savetxt( 'memK.dat', self.integ.memK, '%20.8e' )

        #Write a human-readable summary of the run parameters to info.txt
        with open( 'info.txt', 'w' ) as f:
            f.write( '#########################################################\n' )
            f.write( 'Running ' + str(self.methodname) + ' Dynamics for ' + str(Nsteps) + ' Steps\n' )
            f.write( '#########################################################\n\n' )

            f.write( '--- System ---\n' )
            f.write( 'Number of beads        : ' + str(self.nbds) + '\n' )
            f.write( 'Number of nuclear DOF  : ' + str(self.nnuc) + '\n' )
            f.write( 'Number of states       : ' + str(self.nstates) + '\n' )
            f.write( 'Nuclear mass(es)       : ' + str(self.mass) + '\n' )
            f.write( 'Temperature beta       : ' + str(self.beta) + '\n' )
            f.write( 'beta_p (beta/nbds)     : ' + str(self.beta_p) + '\n' )
            f.write( 'Random seed            : ' + str(self.seed) + '\n\n' )

            f.write( '--- Potential ---\n' )
            f.write( 'Potential (potype)     : ' + str(self.potype) + '\n' )
            f.write( 'Potential name         : ' + str(self.potential.potname) + '\n' )
            f.write( 'potparams              : ' + str(self.potential.potparams) + '\n\n' )

            f.write( '--- Method ---\n' )
            f.write( 'Spin-mapping           : ' + str(self.spin_map) + '\n' )
            f.write( 'centroid approximation : ' + str(self.centroid_bool) + '\n' )
            f.write( 'bead approximation     : ' + str(self.bead_bool) + '\n' )
            f.write( 'Langevin               : ' + str(self.langevin) + '\n' )
            if( self.langevin is not None ):
                f.write( 'Langevin params        : ' + str(self.langevin_params) + '\n' )
            f.write( 'jump_index             : ' + str(self.jump_index) + '\n' )
            f.write( 'full_jump              : ' + str(self.full_jump) + '\n' )
            f.write( '\n' )

            f.write( '--- Integrator ---\n' )
            f.write( 'Integrator type        : ' + str(intype) + '\n' )
            f.write( 'Time-step (delt)       : ' + str(delt) + '\n' )
            f.write( 'Number of steps        : ' + str(Nsteps) + '\n' )
            f.write( 'Print every (Nprint)   : ' + str(Nprint) + '\n' )
            f.write( 'Initial time           : ' + str(init_time) + '\n' )
            f.write( 'small_dt_ratio         : ' + str(small_dt_ratio) + '\n' )

        current_time = init_time
        step = 0
        for step in range( Nsteps ):

            #Print data starting with initial time
            if( step % Nprint == 0 ):
                if( self.verbose ):
                    print('Writing data at step', step, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for', self.methodname, 'Dynamics calculation')
                self.print_data( current_time )
                sys.stdout.flush()

            #Integrate EOM by one time-step
            self.integ.onestep( self, step )

            #Increase current time
            current_time = init_time + delt * (step+1)

            #Quantum jump. At every predetermined index k in jump_index the mapping variables are
            #resampled on the spin sphere at the END of time-step k (step+1 == k), i.e. at
            #t = init_time + k*delt; nucR and nucP are left exactly as the integrator produced them.
            #The check sits after the current_time update so that the time recorded for the jump is
            #the end-of-step time at which it actually happens.
            if( (step+1) in jump_steps ):

                Sx_before = np.mean( self.mapSx ); Sy_before = np.mean( self.mapSy ); Sz_before = np.mean( self.mapSz )

                #Close out the previous segment's time-dependent weights at t_n, W^(n-1)(t_n). This
                #must precede init_map_spin so that the resample's factor of 2 is folded into them.
                self.update_jump_weights_t( (Sx_before, Sy_before, Sz_before) )

                #full_jump=True resamples over the whole spin sphere. full_jump=False is an
                #approximate decoherence jump: resample only within the adiabat the trajectory is
                #already on, which init_map_spin does by drawing Sz from (-1,0) for init_state=0 and
                #from (0,1) for init_state=1. So mapSz<0 -> state 0 and mapSz>0 -> state 1, and the
                #jump can never move the trajectory to the other surface.
                if( self.full_jump ):
                    self.init_map_spin()
                else:
                    self.init_map_spin( 1 if Sz_before > 0.0 else 0 )

                Sx_after  = np.mean( self.mapSx ); Sy_after  = np.mean( self.mapSy ); Sz_after  = np.mean( self.mapSz )

                #The post-resample spin S^(n)(t_n): with mapS_prev it fixes the time-dependent weights'
                #dot and sgn factors for the whole segment that starts here (see update_jump_weights_t).
                self.mapS_post = ( Sx_after, Sy_after, Sz_after )

                #Advance the four quantum-jump weights across this jump, then log them. Both the
                #pre- and post-resample triples are needed: Sz^(n-1) is the value right before the
                #jump, not one carried over from the previous jump.
                self.update_jump_weights( (Sx_before, Sy_before, Sz_before),
                                          (Sx_after,  Sy_after,  Sz_after) )
                self.write_jump_weights( current_time, Sz_after )

                self.file_jump.write( 'Quantum Jump at t=%.8e:\n' % current_time )
                self.file_jump.write( '%20.8e%20.8e%20.8e\n' % ( Sx_before, Sy_before, Sz_before ) )
                self.file_jump.write( '%20.8e%20.8e%20.8e\n' % ( Sx_after,  Sy_after,  Sz_after  ) )
                self.file_jump.write( '%20.8e%20.8e%20.8e%20.8e\n'
                                      % ( self.W_PP, self.W_CP, self.W_PC, self.W_CC ) )
                self.file_jump.write( '%20.8e%20.8e%20.8e%20.8e\n'
                                      % ( self.W_PP_t, self.W_CP_t, self.W_PC_t, self.W_CC_t ) )
                self.file_jump.write( '\n' )
                self.file_jump.flush()

                if( self.verbose ):
                    print( 'Quantum Jump at step', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'),
                           ': mapS (%.6e, %.6e, %.6e) -> (%.6e, %.6e, %.6e)'
                           % ( Sx_before, Sy_before, Sz_before, Sx_after, Sy_after, Sz_after ) )
                    sys.stdout.flush()

        #Print data at final step regardless of Nprint
        if( self.verbose ):
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
        if( self.file_jump is not None ):
            self.file_jump.close()
        if( self.file_weights is not None ):
            self.file_weights.close()
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
                #d_nucP += -d_Vz * erf(self.mapSz[:,np.newaxis] / self.functional_param) + 4 * np.average(Vz[:,np.newaxis]) * NAC * self.mapSx[:,np.newaxis] * np.exp(-(self.mapSz[:,np.newaxis] / self.functional_param)**2) / (self.functional_param * np.sqrt(np.pi))
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

    def init_jump_weights( self ):

        #Seed the four quantum-jump weights at n=0. Sz^(0) is the initial sampled mapSz, so this
        #must be called once at the start of run_dynamics, after the driver's init_map_spin.
        #  W_PP^(0) = 2|Sz^(0)|,  W_CP^(0) = 2,  W_PC^(0) = 2,  W_CC^(0) = 3
        #all times sample_fac, which is 2 if the initial init_map_spin sampled the full sphere.
        #Sz0 is used only here and for row 0 of weights.dat -- the recursion itself takes both of its
        #operands from the jump event, so nothing is carried between jumps.
        self.Sz0 = np.mean( self.mapSz )
        self.W_PP = self.sample_fac * 2.0 * np.abs( self.Sz0 )
        self.W_CP = self.sample_fac * 2.0
        self.W_PC = self.sample_fac * 2.0
        self.W_CC = self.sample_fac * 3.0

        #Remembered for W^(0)(t) = sample_fac0 (2|Sz^(0)(t)|, 2, 2, 3), closed out at the first jump
        self.sample_fac0 = self.sample_fac

    #####################################################################

    def update_jump_weights( self, S_before, S_after ):

        #Advance the four quantum-jump weights across one jump. BOTH operands come from this jump:
        #S_before is mapS immediately BEFORE the resample (the dynamical value at t_n, written
        #S^(n-1)(t_n)) and S_after is the newly sampled S^(n). Nothing is carried from the previous
        #jump, so there is no stored "last sample".
        #
        #  dot = Sx^(n-1)(t_n) Sx^(n) + Sy^(n-1)(t_n) Sy^(n)
        #  fac = 1 + sgn( Sz^(n-1)(t_n) Sz^(n) )        (2 if the sign is preserved, 0 if flipped)
        #
        #full_jump = False -- the resample stays in the pre-jump hemisphere, so sgn is always +1 and
        #fac is identically 2 here:
        #  W_PP^(n) = |Sz^(n)| fac W_PP^(n-1)      W_PC^(n) = fac W_PP^(n-1)
        #  W_CP^(n) = |Sz^(n)| fac W_CP^(n-1)      W_CC^(n) = fac W_CP^(n-1)
        #
        #full_jump = True -- the resample covers the whole sphere, so the Sx/Sy overlap enters:
        #  W_PP^(n) =       dot W_PC^(n-1) + |Sz^(n)| fac W_PP^(n-1)
        #  W_CP^(n) =       dot W_CC^(n-1) + |Sz^(n)| fac W_CP^(n-1)
        #  W_PC^(n) = (3/2) dot W_PC^(n-1) +          fac W_PP^(n-1)
        #  W_CC^(n) = (3/2) dot W_CC^(n-1) +          fac W_CP^(n-1)
        #
        #The second subscript closes into 2x2 blocks: (PP,PC) read only the old PP/PC and (CP,CC)
        #only the old CP/CC, the first subscript being inert. The False rule is the True one with
        #dot set to 0 -- by definition of the rule, not because the overlap vanishes (Sx and Sy are
        #unconstrained by the hemisphere choice, so the dynamical dot is generally nonzero).
        #Zero is absorbing under full_jump=False (fac=0 kills everything permanently) but NOT under
        #full_jump=True, where fac=0 still leaves the dot terms.
        #Both rules are full_jump_recursion: the False rule is the True one with dot set to 0.
        #The full-sphere factor of 2 is NOT applied here: init_map_spin() already doubled the old
        #weights just before this call, and the recursion is linear in them. Every new value is
        #formed before any assignment: each reads two of the OLD weights.
        self.W_PP, self.W_CP, self.W_PC, self.W_CC = self.full_jump_recursion(
            S_after, S_before, ( self.W_PP, self.W_CP, self.W_PC, self.W_CC ), full=self.full_jump )

    #####################################################################

    @staticmethod
    def full_jump_recursion( S_new, S_old, W_old, absSz=None, full=True ):

        #The full_jump weight recursion shared by the constant and the time-dependent weights:
        #  W_AP =       dot W_AC_old + absSz fac W_AP_old       (A = P or C)
        #  W_AC = (3/2) dot W_AC_old +       fac W_AP_old
        #  dot = Sx_new Sx_old + Sy_new Sy_old,   fac = 1 + sgn( Sz_new Sz_old )
        #S_old/S_new are the spin right before/after ONE jump, so dot and fac belong to that jump.
        #absSz defaults to |Sz_new| (the constant weights). The time-dependent weights pass
        #|Sz^(n)(t)| at a later time t instead -- the ONLY place t enters them. W_old = (W_PP, W_CP,
        #W_PC, W_CC); returns the new tuple in the same order. Works elementwise, so the analysis can
        #pass absSz as an array over time. full=False (full_jump=False) sets dot = 0, which is exactly
        #the hemisphere rule of update_jump_weights, for the constant and the time-dependent weights.
        Sx_n, Sy_n, Sz_n = S_new
        Sx_o, Sy_o, Sz_o = S_old
        W_PP, W_CP, W_PC, W_CC = W_old
        if( absSz is None ):
            absSz = np.abs( Sz_n )

        dot = ( Sx_n * Sx_o + Sy_n * Sy_o ) if full else 0.0
        fac = 1.0 + np.sign( Sz_n * Sz_o )

        return ( dot * W_PC + absSz * fac * W_PP,
                 dot * W_CC + absSz * fac * W_CP,
                 1.5 * dot * W_PC + fac * W_PP,
                 1.5 * dot * W_CC + fac * W_CP )

    #####################################################################

    def update_jump_weights_t( self, S_before ):

        #Close out the time-dependent weights of the segment that jump n ends, i.e. evaluate
        #W^(n-1)(t) at t = t_n. S_before is the bead-mean S^(n-1)(t_n), right before the resample.
        #  n = 1:  W^(0)(t_1) = sample_fac0 (2|Sz^(0)(t_1)|, 2, 2, 3)
        #  n > 1:  W^(n-1)(t_n) = recursion( S^(n-1)(t_(n-1)), S^(n-2)(t_(n-1)), W^(n-2)(t_(n-1)),
        #                                    absSz = |Sz^(n-1)(t_n)| )
        #i.e. dot and fac are those of jump n-1, built from the spins right before (mapS_prev) and
        #right after (mapS_post) that jump, and W^(n-2)(t_(n-1)) = W_*_t from that jump; only the |Sz|
        #factor is taken at t_n. Then S^(n-1)(t_n) becomes the new mapS_prev (mapS_post is set by the
        #caller after the resample). The analysis finishes the chain the same way for W^(n)(t), with
        #|Sz^(n)(t)| from the trajectory.
        #Must be called BEFORE init_map_spin, which multiplies W_*_t by jump n's sampling factor.
        if( self.mapS_prev is None ):
            f0 = self.sample_fac0
            self.W_PP_t = f0 * 2.0 * np.abs( S_before[2] )
            self.W_CP_t = f0 * 2.0
            self.W_PC_t = f0 * 2.0
            self.W_CC_t = f0 * 3.0
        else:
            self.W_PP_t, self.W_CP_t, self.W_PC_t, self.W_CC_t = self.full_jump_recursion(
                self.mapS_post, self.mapS_prev, ( self.W_PP_t, self.W_CP_t, self.W_PC_t, self.W_CC_t ),
                absSz = np.abs( S_before[2] ), full = self.full_jump )

        self.mapS_prev = tuple( S_before )

    #####################################################################

    def write_jump_weights( self, current_time, Sz_n ):

        #One row of weights.dat: the sample time, the sampled Sz, and the four weights that hold
        #from this time until the next jump. The row index is n. The last nine columns are what the
        #analysis needs to build the time-dependent W^(n)(t): S^(n-1)(t_n), W^(n-1)(t_n) (already
        #times jump n's sampling factor), and the sampled Sx^(n), Sy^(n) (Sz^(n) is column 1).
        #Row 0 has no previous segment and writes 0 for the first seven. Both full_jump settings
        #write the same columns (full_jump=False uses them with dot = 0).
        Sx_n, Sy_n = np.mean( self.mapSx ), np.mean( self.mapSy )
        if( self.mapS_prev is None ):
            extra = [ 0.0 ] * 7 + [ Sx_n, Sy_n ]
        else:
            extra = list( self.mapS_prev ) + [ self.W_PP_t, self.W_CP_t, self.W_PC_t, self.W_CC_t,
                                               Sx_n, Sy_n ]
        self.file_weights.write( ( '%20.8e' * 15 + '\n' )
                                 % tuple( [ current_time, Sz_n, self.W_PP, self.W_CP, self.W_PC, self.W_CC ] + extra ) )
        self.file_weights.flush()

    #####################################################################

    def init_map_spin(self, init_state = None):

        #Initialize spin mapping variables by uniformly samping them on the spin sphere
        #Sx = sin(theta)cos(phi), Sy = sin(theta)sin(phi), Sz = cos(theta)
        #Sampling by randomly choose cos(theta) in[-1,1], and phi in [0, 2pi]
        #init_state could be 0 or 1, corresponding to cos_theta equal (-1,0) or (0,1), respectively

        if( getattr( self, 'verbose', True ) ):
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

        #Sampling over the whole sphere (init_state=None) doubles the observables, so it carries a
        #factor of 2 in the weights; a hemisphere sample carries 1. Before run_dynamics the weights
        #are not live yet, so the factor is only recorded and init_jump_weights applies it (it is
        #overwritten, not compounded, by repeated calls). During a jump the weights are live and are
        #scaled here, just before update_jump_weights; so are the time-dependent W_*_t that
        #update_jump_weights_t has just closed out.
        self.sample_fac = 2.0 if init_state is None else 1.0
        if( self.W_PP is not None ):
            self.W_PP *= self.sample_fac
            self.W_CP *= self.sample_fac
            self.W_PC *= self.sample_fac
            self.W_CC *= self.sample_fac
        if( self.W_PP_t is not None ):
            self.W_PP_t *= self.sample_fac
            self.W_CP_t *= self.sample_fac
            self.W_PC_t *= self.sample_fac
            self.W_CC_t *= self.sample_fac

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

        #Each row is written exactly as np.savetxt( f, row.reshape(1,-1), fmt_str ) would write it
        #(fmt_str per column, joined by single spaces), without savetxt's per-call overhead, which
        #dominated the run time once the integrator step was compiled. Only output.dat is flushed
        #every print; the other files are flushed when run_dynamics closes them.
        def write_row( f, row ):
            f.write( ' '.join( [fmt_str] * len(row) ) % tuple(row) + '\n' )

        write_row( self.file_output, output )
        self.file_output.flush()

        #columns go as bead_1_nuclei_1 bead_1_nuclei_2 ... bead_1_nuclei_K bead_2_nuclei_1 bead_2_nuclei_2 ...
        write_row( self.file_nucR, [current_time] + self.nucR.ravel().tolist() )
        write_row( self.file_nucP, [current_time] + self.nucP.ravel().tolist() )

        #columns go as bead_1_state_1 bead_1_state_2 ... bead_1_state_K bead_2_state_1 bead_2_state_2 ...
        write_row( self.file_mapSx, [current_time] + self.mapSx.ravel().tolist() )
        write_row( self.file_mapSy, [current_time] + self.mapSy.ravel().tolist() )
        write_row( self.file_mapSz, [current_time] + self.mapSz.ravel().tolist() )
