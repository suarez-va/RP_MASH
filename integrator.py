import numpy as np
import normal_mode
from scipy.special import jv, struve

class integrator():

    ###############################################################

    def __init__( self, map_rpmd, delt, intype, small_dt_ratio ):

        #map_rpmd is the mapping-rpmd class passed to the routines
        self.delt    = delt #time-step
        self.intype  = intype #integrator type
        self.small_dt_ratio = small_dt_ratio #number of steps taken to update mapping variables for given nuclear update for vv-like integrators

        #Input error check
        self.error_check( map_rpmd )

        print('Using integrator',intype,'with a time-step = ',delt)

        #Want to actually save time-derivative of nuclear momentum from step to step if doing
        #velocity-verlet, analytical, or cayley integration
        if( self.intype == 'vv' or self.intype == 'analyt' or self.intype == 'cayley'):
            self.d_nucP_for_vv = None
            print( 'Number of steps for internal mapping-variable velocity-verlet loop is = ',small_dt_ratio )

        #Initialize normal mode frequencies of ring polymer if doing analytical or cayley integration
        #Note that normal-mode frequencies are mass independent and thus are the same for all nuclei
        #Thus all normal-mode arrays are of dimension nbds
        if( self.intype == 'analyt' or self.intype == 'cayley' ):

            self.nm_freq = normal_mode.calc_normal_mode_freq( map_rpmd.beta_p, map_rpmd.nbds )

            #if using cayley integrator make additional frequency arrays
            if( self.intype == 'cayley' ):
                self.nm_freq_prod = self.delt * self.nm_freq**2
                self.nm_freq_sum  = 1 + 0.25 * self.delt * self.nm_freq_prod
                self.nm_freq_dif  = 1 - 0.25 * self.delt * self.nm_freq_prod
        
        #Want to store the last 4 derivative if using the ABM integrator
        if( self.intype == 'abm' ):
            self.prev_d_nucR = np.zeros( (4,) + map_rpmd.nucR.shape )
            self.prev_d_nucP = np.zeros( (4,) + map_rpmd.nucP.shape )
            self.prev_d_mapR = np.zeros( (4,) + map_rpmd.mapR.shape )
            self.prev_d_mapP = np.zeros( (4,) + map_rpmd.mapP.shape )

        self.rng = np.random.default_rng()

    ###############################################################

    def onestep( self, map_rpmd, step ):

        #Wrapper subroutine to choose appropriate integrator to integrate
        #equations of motion by one time-step

        if( self.intype == 'rk4' ):
            self.rk4( map_rpmd )
        elif( self.intype == 'abm' ):
            self.abm( map_rpmd, step )
        elif( self.intype == 'vv' or self.intype == 'analyt' or self.intype == 'cayley' ):
            if (map_rpmd.langevin == None):
                self.vv_outer( map_rpmd, step )
            else:
                self.vv_outer_GLE( map_rpmd, step)

    ###############################################################

    def rk4( self, map_rpmd ):

        #Integrate EOM using 4th order runge-kutta
        #Note that step isn't used in this routine

        #Copy initial positions and momenta at time t
        init_nucR = np.copy( map_rpmd.nucR )
        init_nucP = np.copy( map_rpmd.nucP )
        init_mapR = np.copy( map_rpmd.mapR )
        init_mapP = np.copy( map_rpmd.mapP )

        #obtain k1 vectors
        k1_nucR, k1_nucP, k1_mapR, k1_mapP = map_rpmd.get_timederivs()

        #obtain k2 vectors
        map_rpmd.nucR = init_nucR + 0.5 * self.delt * k1_nucR
        map_rpmd.nucP = init_nucP + 0.5 * self.delt * k1_nucP
        map_rpmd.mapR = init_mapR + 0.5 * self.delt * k1_mapR
        map_rpmd.mapP = init_mapP + 0.5 * self.delt * k1_mapP

        k2_nucR, k2_nucP, k2_mapR, k2_mapP = map_rpmd.get_timederivs()

        #obtain k3 vectors
        map_rpmd.nucR = init_nucR + 0.5 * self.delt * k2_nucR
        map_rpmd.nucP = init_nucP + 0.5 * self.delt * k2_nucP
        map_rpmd.mapR = init_mapR + 0.5 * self.delt * k2_mapR
        map_rpmd.mapP = init_mapP + 0.5 * self.delt * k2_mapP

        k3_nucR, k3_nucP, k3_mapR, k3_mapP = map_rpmd.get_timederivs()

        #obtain k4 vectors
        map_rpmd.nucR = init_nucR + 1.0 * self.delt * k3_nucR
        map_rpmd.nucP = init_nucP + 1.0 * self.delt * k3_nucP
        map_rpmd.mapR = init_mapR + 1.0 * self.delt * k3_mapR
        map_rpmd.mapP = init_mapP + 1.0 * self.delt * k3_mapP

        k4_nucR, k4_nucP, k4_mapR, k4_mapP = map_rpmd.get_timederivs()

        #update new positions/momenta using RK4
        map_rpmd.nucR = init_nucR + 1.0/6.0 * self.delt * ( k1_nucR + 2.0*k2_nucR + 2.0*k3_nucR + k4_nucR )
        map_rpmd.nucP = init_nucP + 1.0/6.0 * self.delt * ( k1_nucP + 2.0*k2_nucP + 2.0*k3_nucP + k4_nucP )
        map_rpmd.mapR = init_mapR + 1.0/6.0 * self.delt * ( k1_mapR + 2.0*k2_mapR + 2.0*k3_mapR + k4_mapR )
        map_rpmd.mapP = init_mapP + 1.0/6.0 * self.delt * ( k1_mapP + 2.0*k2_mapP + 2.0*k3_mapP + k4_mapP )

    ###############################################################

    def vv_outer( self, map_rpmd, step ):

        #Outer loop to integrate EOM using velocity-verlet like algorithms
        #This includes pengfei's implementation, and the analtyical and cayley modification of it

        #If initial step of dynamics need to initialize electronic hamiltonian
        #and derivative of nuclear momentum (aka the force on the nuclei)
        #NOTE: moving forward these calls may need to be generalized to allow for other types of methods
        if( step == 0 ):
            map_rpmd.potential.calc_Hel( map_rpmd.nucR )
            if( self.intype == 'vv' ):
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
            else:
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum by 1/2 a time-step
        self.update_vv_nucP( map_rpmd )

        #Update mapping variables by 1/2 a time-step
        if(map_rpmd.spin_map==True):
            self.update_vv_mapS( map_rpmd )
        else:
            self.update_vv_mapRP( map_rpmd )

        #Update nuclear position for full time-step
        if( self.intype == 'vv' ):
            self.update_vv_nucR( map_rpmd )
        elif( self.intype == 'analyt' ):
            self.update_analyt_nucR( map_rpmd )
        elif( self.intype == 'cayley' ):
            self.update_cayley_nucR( map_rpmd )

        #Update electronic Hamiltonian given new position
        #NOTE: Moving forward this call may need to be generalized to allow for other types of methods
        map_rpmd.potential.calc_Hel( map_rpmd.nucR )

        #Update mapping variables by 1/2 a time-step
        if(map_rpmd.spin_map==True):
            self.update_vv_mapS( map_rpmd )
        else:
            self.update_vv_mapRP( map_rpmd )

        #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
        #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
        if( self.intype == 'vv' ):
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
        else:
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum to full time-step
        self.update_vv_nucP( map_rpmd )

    def vv_outer_langevin( self, map_rpmd, step ):
        #Outer loop to integrate EOM using velocity-verlet like algorithms with Langevin friction
        if( step == 0 ):
            map_rpmd.potential.calc_Hel( map_rpmd.nucR )
            if( self.intype == 'vv' ):
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
            else:
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Generate two random numbers with zero mean and unit width
        theta = self.rng.normal(0.0, 1.0, size=(map_rpmd.nbds, map_rpmd.nnuc))
        zeta  = self.rng.normal(0.0, 1.0, size=(map_rpmd.nbds, map_rpmd.nnuc))
        
        gamma = map_rpmd.langevin
        sigma = np.sqrt(2*gamma/map_rpmd.beta_p/map_rpmd.mass)

        #Update nuclear momentum by 1/2 a time-step
        #self.update_vv_nucP( map_rpmd )
        map_rpmd.nucP += 0.5 * ( self.delt * self.d_nucP_for_vv
                                - gamma * map_rpmd.nucP * self.delt + map_rpmd.mass * sigma * zeta * np.sqrt(self.delt)
                                - 0.25 * gamma * self.delt**2 * (self.d_nucP_for_vv - gamma * map_rpmd.nucP )
                                - 0.5 * map_rpmd.mass * sigma * self.delt**1.5 * (0.5*zeta + 1/np.sqrt(3)*theta))
        
        #Update mapping variables by 1/2 a time-step
        if(map_rpmd.spin_map==True):
            self.update_vv_mapS( map_rpmd )
        else:
            self.update_vv_mapRP( map_rpmd )

        #Update nuclear position for full time-step
        if( self.intype == 'vv' ):
            self.update_vv_nucR( map_rpmd )
        elif( self.intype == 'analyt' ):
            self.update_analyt_nucR( map_rpmd )
        elif( self.intype == 'cayley' ):
            self.update_cayley_nucR( map_rpmd )
        #Add the extra terms
        map_rpmd.nucR += sigma * theta * self.delt**1.5 / 2 /np.sqrt(3)

        #Update electronic Hamiltonian given new position
        #NOTE: Moving forward this call may need to be generalized to allow for other types of methods
        map_rpmd.potential.calc_Hel( map_rpmd.nucR )

        #Update mapping variables by 1/2 a time-step
        if(map_rpmd.spin_map==True):
            self.update_vv_mapS( map_rpmd )
        else:
            self.update_vv_mapRP( map_rpmd )

        #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
        #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
        if( self.intype == 'vv' ):
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
        else:
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum to full time-step
        map_rpmd.nucP += 0.5 * ( self.delt * self.d_nucP_for_vv
                                - gamma * map_rpmd.nucP * self.delt + map_rpmd.mass * sigma * zeta * np.sqrt(self.delt)
                                - 0.25 * gamma * self.delt**2 * (self.d_nucP_for_vv - gamma * map_rpmd.nucP )
                                - 0.5 * map_rpmd.mass * sigma * self.delt**1.5 * (0.5*zeta + 1/np.sqrt(3)*theta))

    def vv_outer_GLE( self, map_rpmd, step ):
                
        #print('running step', step)
        gamma = map_rpmd.langevin
        #Generate the random fluctuation force using the formula in the interpolation paper
        N_grid = 1000 #The grid number of random force calculation
        delt_omega = np.pi/N_grid/self.delt
        omega_arr = np.arange(0, N_grid+1) * (step+1)*np.pi/N_grid

        #Outer loop to integrate EOM using velocity-verlet like algorithms with generalized Langevin friction
        #Using the normal-mode integrator
        if( step == 0 ):
            map_rpmd.potential.calc_Hel( map_rpmd.nucR )
            if( self.intype == 'vv' ):
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
            else:
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

            #Generate two random numbers with zero mean and unit width
            theta = self.rng.normal(0.0, 1.0, size=(N_grid+1, map_rpmd.nbds, map_rpmd.nnuc))
            zeta  = self.rng.normal(0.0, 1.0, size=(N_grid+1, map_rpmd.nbds, map_rpmd.nnuc))

            a = theta * np.sqrt(2*gamma*delt_omega/np.pi/map_rpmd.beta_p)
            b = zeta * np.sqrt(2*gamma*delt_omega/np.pi/map_rpmd.beta_p)
            a[0] *= 0.5
            a[-1] *= 0.5
            b[0] *= 0.5
            b[-1] *= 0.5

            self.random_force = np.sum( a*np.cos(omega_arr[:, None, None]) + b*np.sin(omega_arr[:,None,None]), axis=0)
        
        #Update the thermostatted nuclear momentum
        map_rpmd.nucP *= 1 - self.delt**2/4/map_rpmd.mass * map_rpmd.friction_kernel[0] - self.delt/2/map_rpmd.mass * gamma
        
        if step != 0:
            step_friction_kernel = np.copy(map_rpmd.friction_kernel[1:step])

            map_rpmd.nucP += -self.delt**2/map_rpmd.mass/2 * np.sum( step_friction_kernel * np.flip(map_rpmd.nucP_arr[0:step-1], axis=0), axis=0 )

        #The random force
        map_rpmd.nucP += self.delt * self.random_force / 2
 
        #Update nuclear momentum by 1/2 a time-step
        #self.update_vv_nucP( map_rpmd )
        map_rpmd.nucP += 0.5 * self.delt * self.d_nucP_for_vv
                                    
        #Update mapping variables by 1/2 a time-step
        if(map_rpmd.spin_map==True):
            self.update_vv_mapS( map_rpmd )
        else:
            self.update_vv_mapRP( map_rpmd )

        #Update nuclear position for full time-step
        if( self.intype == 'vv' ):
            self.update_vv_nucR( map_rpmd )
        elif( self.intype == 'analyt' ):
            self.update_analyt_nucR_GLE( map_rpmd )
        elif( self.intype == 'cayley' ):
            self.update_cayley_nucR_GLE( map_rpmd )

        #Update electronic Hamiltonian given new position
        #NOTE: Moving forward this call may need to be generalized to allow for other types of methods
        map_rpmd.potential.calc_Hel( map_rpmd.nucR )

        #Update mapping variables by 1/2 a time-step
        if(map_rpmd.spin_map==True):
            self.update_vv_mapS( map_rpmd )
        else:
            self.update_vv_mapRP( map_rpmd )

        #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
        #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
        if( self.intype == 'vv' ):
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
        else:
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum to full time-step
        map_rpmd.nucP += 0.5 * self.delt * self.d_nucP_for_vv

        #Generate two random numbers with zero mean and unit width
        theta = self.rng.normal(0.0, 1.0, size=(N_grid+1, map_rpmd.nbds, map_rpmd.nnuc))
        zeta  = self.rng.normal(0.0, 1.0, size=(N_grid+1, map_rpmd.nbds, map_rpmd.nnuc))

        a = gamma * theta * np.sqrt(delt_omega)
        b = gamma * zeta * np.sqrt(delt_omega)
        a[0] *= 0.5
        a[-1] *= 0.5
        b[0] *= 0.5
        b[-1] *= 0.5

        self.random_force = np.sum( a*np.cos(omega_arr[:, None, None]) + b*np.sin(omega_arr[:,None,None]), axis=0)

        #Update the thermostatted nuclear momentum
        
        step_friction_kernel = np.copy(map_rpmd.friction_kernel[1:step+1])

        map_rpmd.nucP += -self.delt**2/map_rpmd.mass/2 * np.sum(step_friction_kernel*np.flip(map_rpmd.nucP_arr[0:step], axis=0), axis=0)
        #The random force
        map_rpmd.nucP += self.delt * self.random_force / 2

        map_rpmd.nucP /= 1 + self.delt**2/4/map_rpmd.mass * map_rpmd.friction_kernel[0] + self.delt/2/map_rpmd.mass * gamma

        #Save the momentum to the time series
        map_rpmd.nucP_arr[step+1] = map_rpmd.nucP

    def vv_outer_nuconly( self, map_rpmd, step):

        #Outer loop to integrate EOM using velocity-verlet like algorithms
        #This includes pengfei's implementation, and the analtyical and cayley modification of it

        #If initial step of dynamics need to initialize electronic hamiltonian
        #and derivative of nuclear momentum (aka the force on the nuclei)
        #NOTE: moving forward these calls may need to be generalized to allow for other types of methods
        if( step == 0 ):
            map_rpmd.potential.calc_Hel( map_rpmd.nucR )
            if( self.intype == 'vv' ):
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
            else:
                self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum by 1/2 a time-step
        self.update_vv_nucP( map_rpmd )

        #Update mapping variables by 1/2 a time-step
        self.update_vv_mapRP( map_rpmd )

        #Update nuclear position for full time-step
        if( self.intype == 'vv' ):
            self.update_vv_nucR( map_rpmd )
        elif( self.intype == 'analyt' ):
            self.update_analyt_nucR( map_rpmd )
        elif( self.intype == 'cayley' ):
            self.update_cayley_nucR( map_rpmd )

        #Calculate derivative of nuclear momentum at new time-step (aka the force on the nuclei)
        #Don't include contribution from internal modes of ring-polymer if doing analyt or cayley
        if( self.intype == 'vv' ):
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=True)
        else:
            self.d_nucP_for_vv = map_rpmd.get_timederiv_nucP(intRP_bool=False)

        #Update nuclear momentum to full time-step
        self.update_vv_nucP( map_rpmd )

    ###############################################################

    def update_vv_nucP( self, map_rpmd ):

        #Update nuclear momentum by 1/2 a time-step
        map_rpmd.nucP += 0.5 * self.delt * self.d_nucP_for_vv

    ###############################################################

    def update_vv_nucR( self, map_rpmd ):

        #Update nuclear position by a full time-step
        map_rpmd.nucR += map_rpmd.nucP * self.delt / map_rpmd.mass

    ###############################################################

    def update_analyt_nucR( self, map_rpmd ):

        #Update nuclear position by a full time-step using analytical
        #result for internal modes of ring-polymer

        #Transform position and velocities to normal-modes using fourier-transform
        #Note this is faster than directly diagonalizing the frequency matrix
        nucR_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        nucV_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        for i in range( map_rpmd.nnuc ):
            nucR_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucR[:,i] )
            nucV_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucP[:,i] / map_rpmd.mass[i] )

        #Evolve position of the zero-freq mode using velocity-verlet
        #this is the force on the centroid and accounts for the external force on the position
        #external force on momentum accounted for in update_nucP call of velocity-verlet algorithm
        nucR_nm[0,:] += nucV_nm[0,:] * self.delt

        #evolve position/velocities of all other modes using analytical result for hamonic oscillators
        #Note that all nuclei have same nm frequency
        c1 = np.copy( nucV_nm[1:,:] / self.nm_freq[1:,None] )
        c2 = np.copy( nucR_nm[1:,:] )
        freq_dt = self.nm_freq[1:] * self.delt

        nucR_nm[1:,:] = c1 * np.sin( freq_dt )[:,None] + c2 * np.cos( freq_dt )[:,None]

        nucV_nm[1:,:] = self.nm_freq[1:,None] * ( c1 * np.cos( freq_dt )[:,None] - c2 * np.sin( freq_dt )[:,None] )

        #Inverse transform back to real space and convert velocity back to momentum
        for i in range( map_rpmd.nnuc ):
            map_rpmd.nucR[:,i] = normal_mode.normal_mode_to_real( nucR_nm[:,i] )
            map_rpmd.nucP[:,i] = map_rpmd.mass[i] * normal_mode.normal_mode_to_real( nucV_nm[:,i] )

    ###############################################################
    def update_analyt_nucR_GLE( self, map_rpmd  ):

        #Update nuclear position by a full time-step using analytical with the generalized Langevin friction
        #result for internal modes of ring-polymer

        gamma = map_rpmd.langevin

        nm_freq = np.sqrt(self.nm_freq**2 + self.nm_freq * gamma)

        #Transform position and velocities to normal-modes using fourier-transform
        #Note this is faster than directly diagonalizing the frequency matrix
        nucR_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        nucV_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        for i in range( map_rpmd.nnuc ):
            nucR_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucR[:,i] )
            nucV_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucP[:,i] / map_rpmd.mass[i] )

        #Evolve position of the zero-freq mode using velocity-verlet
        #this is the force on the centroid and accounts for the external force on the position
        #external force on momentum accounted for in update_nucP call of velocity-verlet algorithm
        nucR_nm[0,:] += nucV_nm[0,:] * self.delt

        #evolve position/velocities of all other modes using analytical result for hamonic oscillators
        #Note that all nuclei have same nm frequency
        c1 = np.copy( nucV_nm[1:,:] / nm_freq[1:,None] )
        c2 = np.copy( nucR_nm[1:,:] )
        freq_dt = nm_freq[1:] * self.delt

        nucR_nm[1:,:] = c1 * np.sin( freq_dt )[:,None] + c2 * np.cos( freq_dt )[:,None]

        nucV_nm[1:,:] = nm_freq[1:,None] * ( c1 * np.cos( freq_dt )[:,None] - c2 * np.sin( freq_dt )[:,None] )

        #Inverse transform back to real space and convert velocity back to momentum
        for i in range( map_rpmd.nnuc ):
            map_rpmd.nucR[:,i] = normal_mode.normal_mode_to_real( nucR_nm[:,i] )
            map_rpmd.nucP[:,i] = map_rpmd.mass[i] * normal_mode.normal_mode_to_real( nucV_nm[:,i] )

    ###############################################################

    def update_cayley_nucR( self, map_rpmd ):

        #Update nuclear position by a full time-step using Cayley integration scheme
        #See Korol, Bou-Rabee and Miller III JCP 2019

        #Transform position and velocities to normal-modes using fourier-transform
        #Note this is faster than directly diagonalizing the frequency matrix
        nucR_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        nucV_nm = np.zeros([map_rpmd.nbds, map_rpmd.nnuc])
        for i in range( map_rpmd.nnuc ):
            nucR_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucR[:,i] )
            nucV_nm[:,i] = normal_mode.real_to_normal_mode( map_rpmd.nucP[:,i] / map_rpmd.mass[i] )

        #evolve position/velocity of all modes using cayley transform
        nucR_nm_copy = np.copy( nucR_nm )
        nucV_nm_copy = np.copy( nucV_nm )

        nucR_nm = ( self.nm_freq_dif[:,None] * nucR_nm_copy + self.delt * nucV_nm_copy ) / self.nm_freq_sum[:,None]

        nucV_nm = ( -self.nm_freq_prod[:,None] * nucR_nm_copy + self.nm_freq_dif[:,None] * nucV_nm_copy ) / self.nm_freq_sum[:,None]

        #Inverse transform back to real space and convert velocity back to momentum
        for i in range( map_rpmd.nnuc ):
            map_rpmd.nucR[:,i] = normal_mode.normal_mode_to_real( nucR_nm[:,i] )
            map_rpmd.nucP[:,i] = map_rpmd.mass[i] * normal_mode.normal_mode_to_real( nucV_nm[:,i] )

    ###############################################################

    def update_vv_mapRP( self, map_rpmd ):

        #Internal velocity-verlet algorithm for mapping variables
        #Run for small_dt_ratio number of steps for each nuclear update

        small_dt = self.delt / self.small_dt_ratio
        for _ in range( self.small_dt_ratio ):

            #Calculate first time derivative of mapping momentum
            d_mapP = map_rpmd.get_timederiv_mapP()
            
            #Update mapping momentum by 1/4 a time-step
            self.update_vv_mapP( map_rpmd, d_mapP, small_dt )

            #Calculate the time derivative of the mapping position accordingly
            d_mapR = map_rpmd.get_timederiv_mapR()

            #Update mapping position by 1/2 a time-step
            self.update_vv_mapR( map_rpmd, d_mapR, small_dt )

            #Update first time derivative of mapping position at new 1/2 time-step
            d_mapP = map_rpmd.get_timederiv_mapP()

            #Update mapping momentum to another 1/4 time-step to reach 1/2 a time-step
            self.update_vv_mapP( map_rpmd, d_mapP, small_dt )

    ###############################################################

    def update_vv_mapP( self, map_rpmd, d_mapP, small_dt ):

        #Update mapping momentum by 1/4 a time-step
        map_rpmd.mapP += 0.25 * small_dt * d_mapP

    ###############################################################

    def update_vv_mapR( self, map_rpmd, d_mapR, small_dt ):

        #Update mapping position by 1/2 a time-step
        map_rpmd.mapR += 0.5 * d_mapR * small_dt

    ###############################################################
    
    def update_vv_mapSyz( self, map_rpmd, d_mapSy, d_mapSz, small_dt ):

        #Update spin-yz by a 1/4 a time-step
        mapSz_0 = np.copy(map_rpmd.mapSz)
        map_rpmd.mapSy += 0.25 * d_mapSy * small_dt
        map_rpmd.mapSz += 0.25 * d_mapSz * small_dt

        if ( np.array_equal( np.sign(mapSz_0), np.sign(map_rpmd.mapSz)) == False ):
            #if there is a different sign of Sz before and after the step
            if ( map_rpmd.centroid_bool==False):
                #if the centroid approximation is not applied, rescale the whole phase space made of ring polymers
                H_bo = map_rpmd.potential.get_bopes(map_rpmd.nucR)
                Vz = (H_bo[:,1]-H_bo[:,0])/2
                NAC = map_rpmd.potential.calc_NAC(map_rpmd.nucR)

            else:
                #if the centroid approximation is applied, all beads are non-adiabatically coupled with the same NAC vectors
                #which are determined by the centroid of ring polymers
                R_bar = np.mean(map_rpmd.nucR, axis = 0)
                Rbar_arr = np.tile(R_bar, (map_rpmd.nbds, 1))
                H_bo = map_rpmd.potential.get_bopes(Rbar_arr)
                Vz = (H_bo[:,1]-H_bo[:,0])/2
                NAC = map_rpmd.potential.calc_NAC(Rbar_arr)

            unit_NAC = NAC / np.sqrt(np.sum( NAC**2 ))

            nucP_NAC = np.sum(map_rpmd.nucP * unit_NAC) # the nuclear momentum norm along the NAC direction
            KE_eff = nucP_NAC**2 * np.sum(unit_NAC**2 / map_rpmd.mass[np.newaxis,:]) / 2 # the effective kinetic energy along the NAC direction

            if ( mapSz_0[0] > 0 or KE_eff > 2 * np.sum(Vz) ):
                # The hopping happens: transferring to a lower state or the kenitic energy is sufficient
                # reach the point of a potential surface hopping. The momentum rescaling is about to be performed
                # KE_eff += 2 * Vz * np.sign( Sz,init )

                nucP2_NAC_new = nucP_NAC**2 + 2*np.sum(Vz*np.sign(mapSz_0)) / np.sum(unit_NAC**2 / map_rpmd.mass[np.newaxis,:] / 2) 
                nucP_NAC_new = np.sign(nucP_NAC) * np.sqrt(nucP2_NAC_new)
                map_rpmd.nucP += (nucP_NAC_new - nucP_NAC) * unit_NAC

            else:
                # The hopping is frustrated, momenta will be bounced back on the direction of NAC
                # Sz will be inverse
                map_rpmd.nucP -= 2*nucP_NAC * unit_NAC
                map_rpmd.mapSz *= -1

                '''for i in range(map_rpmd.nbds):
                    if (np.sign(mapSz_0[i])!=np.sign(map_rpmd.mapSz[i])):

                        unit_NAC = NAC[i] / np.sqrt(np.sum( NAC[i]**2 ))

                        nucP_NAC = np.sum(map_rpmd.nucP[i] * unit_NAC) # the nuclear momentum norm along the NAC direction
                        KE_eff = nucP_NAC**2 * np.sum(unit_NAC**2 / map_rpmd.mass) / 2 # the effective kinetic energy along the NAC direction

                        if ( mapSz_0[i] > 0 or KE_eff > 2 * Vz[i] ):
                            # The hopping happens: transferring to a lower state or the kenitic energy is sufficient
                            # reach the point of a potential surface hopping. The momentum rescaling is about to be performed
                            # KE_eff += 2 * Vz * np.sign( Sz,init )

                            nucP2_NAC_new = nucP_NAC**2 + 2*Vz[i]*np.sign(mapSz_0[i]) / np.sum(unit_NAC**2 / map_rpmd.mass / 2) 
                            nucP_NAC_new = np.sign(nucP_NAC) * np.sqrt(nucP2_NAC_new)
                            map_rpmd.nucP[i] += (nucP_NAC_new - nucP_NAC) * unit_NAC

                        else:
                            # The hopping is frustrated, momenta will be bounced back on the direction of NAC
                            # Sz will be inverse
                            map_rpmd.nucP[i] -= 2*nucP_NAC * unit_NAC
                            map_rpmd.mapSz[i] *= -1
        
            else:
                #if there is a different sign of Sz before and after the step
                R_bar = np.mean(map_rpmd.nucR, axis = 0)
                Rbar_arr = np.tile(R_bar, (map_rpmd.nbds, 1))
                Vz = map_rpmd.potential.get_bopes(Rbar_arr)[0, 1]
                NAC = map_rpmd.potential.calc_NAC(Rbar_arr)[0]

                unit_NAC = NAC / np.sqrt(np.sum( NAC**2 ))

                P_bar = np.mean(map_rpmd.nucP, axis = 0)

                nucP_NAC = np.sum(P_bar * unit_NAC) # the nuclear momentum norm along the NAC direction
                KE_eff = nucP_NAC**2 * np.sum(unit_NAC**2 / map_rpmd.mass) / 2 # the effective kinetic energy along the NAC direction

                if ( mapSz_0[0] > 0 or KE_eff > 2 * Vz ):
                    # The hopping happens: transferring to a lower state or the kenitic energy is sufficient
                    # reach the point of a potential surface hopping. The momentum rescaling is about to be performed
                    # KE_eff += 2 * Vz * np.sign( Sz,init )

                    nucP2_NAC_new = nucP_NAC**2 + 2*Vz*np.sign(mapSz_0[0]) / np.sum(unit_NAC**2 / map_rpmd.mass / 2)
                    nucP_NAC_new = np.sign(nucP_NAC) * np.sqrt(nucP2_NAC_new)
                    map_rpmd.nucP += (nucP_NAC_new - nucP_NAC) * unit_NAC[np.newaxis,:]

                else:
                    # The hopping is frustrated, momenta will be bounced back on the direction of NAC
                    # Sz will be inverse
                    map_rpmd.nucP -= 2*nucP_NAC * unit_NAC[np.newaxis,:]
                    map_rpmd.mapSz *= -1'''

    ###############################################################

    def update_vv_mapSx( self, map_rpmd, d_mapSx, small_dt):

        #Update spin-x by 1/2 a time-step
        map_rpmd.mapSx += 0.5 * d_mapSx * small_dt

    ###############################################################
    def update_vv_mapS(self, map_rpmd):
        #Internal velocity-verlet algorithm for spin mapping variables
        #Run for small_dt_ratio number of steps for each nuclear update

        small_dt = self.delt / self.small_dt_ratio
        for _ in range( self.small_dt_ratio ):

            #Calculate first time derivative of mapping 'momentum' - mapping Sy and Sz
            d_mapSy, d_mapSz = map_rpmd.get_timederiv_mapSyz()

            #Update mapping 'momentum' by 1/4 a time-step
            self.update_vv_mapSyz( map_rpmd, d_mapSy, d_mapSz, small_dt )

            #Calculate the time derivative of mapSx accordingly
            d_mapSx = map_rpmd.get_timederiv_mapSx()

            #Update mapping 'position' by 1/2 a time-step
            self.update_vv_mapSx( map_rpmd, d_mapSx, small_dt )

            #Update first time derivative of mapping 'momentum' at new 1/2 time-step and make it to a full 1/2 time-step
            d_mapSy, d_mapSz = map_rpmd.get_timederiv_mapSyz()
            self.update_vv_mapSyz( map_rpmd, d_mapSy, d_mapSz, small_dt )

    ###############################################################

    def abm( self, map_rpmd, step ):

        #Integrates the equations of motion using a 4th order Adams-Bashforth-Moulton Predictor-Corrector integrator
        #Following equations in https://en.wikipedia.org/wiki/Linear_multistep_method
        #Predictor step is the Adams-Bashforth method for Y_n+4 to get current time derivative
        #Corrector step is the Adams-Moulton method for Y_n+3 with f(t_n+3, Y_n+3) as the result from the Predictor step

        if step < 4:
            #use rk4 integration for the first 4 steps to obtain an initial set of previous derivative values
            #store current state of system
            prev_nucR = np.copy( map_rpmd.nucR )
            prev_nucP = np.copy( map_rpmd.nucP )
            prev_mapR = np.copy( map_rpmd.mapR )
            prev_mapP = np.copy( map_rpmd.mapP )

            #propagate by a single timestep using rk4
            self.rk4( map_rpmd )

            #calculate derivative over last time step and store in previous arrays
            self.prev_d_nucR[step] = ( map_rpmd.nucR - prev_nucR ) / self.delt
            self.prev_d_nucP[step] = ( map_rpmd.nucP - prev_nucP ) / self.delt
            self.prev_d_mapR[step] = ( map_rpmd.mapR - prev_mapR ) / self.delt
            self.prev_d_mapP[step] = ( map_rpmd.mapP - prev_mapP ) / self.delt

        else:
            #Use the previous 4 derivatives to calculate the current derivative
            #store current state of system
            prev_nucR = np.copy( map_rpmd.nucR )
            prev_nucP = np.copy( map_rpmd.nucP )
            prev_mapR = np.copy( map_rpmd.mapR )
            prev_mapP = np.copy( map_rpmd.mapP )

            #Predictor step propagation
            map_rpmd.nucR = prev_nucR + ( self.prev_d_nucR[3]*55 - self.prev_d_nucR[2]*59 + self.prev_d_nucR[1]*37 - self.prev_d_nucR[0]*9 ) * self.delt / 24
            map_rpmd.nucP = prev_nucP + ( self.prev_d_nucP[3]*55 - self.prev_d_nucP[2]*59 + self.prev_d_nucP[1]*37 - self.prev_d_nucP[0]*9 ) * self.delt / 24
            map_rpmd.mapR = prev_mapR + ( self.prev_d_mapR[3]*55 - self.prev_d_mapR[2]*59 + self.prev_d_mapR[1]*37 - self.prev_d_mapR[0]*9 ) * self.delt / 24
            map_rpmd.mapP = prev_mapP + ( self.prev_d_mapP[3]*55 - self.prev_d_mapP[2]*59 + self.prev_d_mapP[1]*37 - self.prev_d_mapP[0]*9 ) * self.delt / 24

            #Shift back previous derivatives
            self.prev_d_nucR[:-1] = self.prev_d_nucR[1:]
            self.prev_d_nucP[:-1] = self.prev_d_nucP[1:]
            self.prev_d_mapR[:-1] = self.prev_d_mapR[1:]
            self.prev_d_mapP[:-1] = self.prev_d_mapP[1:]

            #Calculate and store new current derivatives for corrector step
            self.prev_d_nucR[3], self.prev_d_nucP[3], self.prev_d_mapR[3], self.prev_d_mapP[3] = map_rpmd.get_timederivs()

            #Corrector step propagation
            map_rpmd.nucR = prev_nucR + ( self.prev_d_nucR[3]*9 + self.prev_d_nucR[2]*19 - self.prev_d_nucR[1]*5 + self.prev_d_nucR[0]*1 ) * self.delt / 24
            map_rpmd.nucP = prev_nucP + ( self.prev_d_nucP[3]*9 + self.prev_d_nucP[2]*19 - self.prev_d_nucP[1]*5 + self.prev_d_nucP[0]*1 ) * self.delt / 24
            map_rpmd.mapR = prev_mapR + ( self.prev_d_mapR[3]*9 + self.prev_d_mapR[2]*19 - self.prev_d_mapR[1]*5 + self.prev_d_mapR[0]*1 ) * self.delt / 24
            map_rpmd.mapP = prev_mapP + ( self.prev_d_mapP[3]*9 + self.prev_d_mapP[2]*19 - self.prev_d_mapP[1]*5 + self.prev_d_mapP[0]*1 ) * self.delt / 24

            #Calculate and store new (updated) current derivatives for next timestep
            self.prev_d_nucR[3], self.prev_d_nucP[3], self.prev_d_mapR[3], self.prev_d_mapP[3] = map_rpmd.get_timederivs()

    ###############################################################

    def error_check( self, map_rpmd ):

        if( self.intype not in ['vv', 'analyt', 'cayley', 'rk4', 'abm'] ):
            print("ERROR: intype not one of valid types: 'vv', 'analyt', 'cayley', 'rk4', 'abm'")
            exit()

        if( self.intype in ['vv', 'analyt', 'cayley' ] and map_rpmd.methodname in ['MV-RPMD', 'mod-MV-RPMD'] ):
            print("ERROR: Cannot run", map_rpmd.methodname,"with velocity-verlet based integrator", self.intype)
            exit()

    ###############################################################

