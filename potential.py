#subroutine to calculate potential terms

import numpy as np
import utils
from abc import ABC, abstractmethod
import nuc_only_potential
import scipy.linalg as la
from numba import njit

#############################################

def set_potential( potype, potparams, nstates, nnuc, nbds ):

    #Separate routine which returns the appropriate potential class indicated by potype

    potype_list = ['harm_const_cpl', 'harm_lin_cpl', 'harm_lin_cpl_symmetrized', 'harm_lin_cpl_sym_2', 'tully_1ac', 'tully_2ac', 'tully_EC',
                    'nstate_morse', 'nuc_only_harm', 'nuc_only_Marcus_withbath', 'pengfei_polariton', 'isolated_elec', 'ET_with_bath']

    if( potype not in potype_list ):
        print("ERROR: potype not one of valid types:")
        print( *potype_list, sep="\n" )
        exit()

    if( potype[:8] != "nuc_only" ):
        #Initialize a multi-state potential
        potclass = eval( potype + '( potparams, nstates, nnuc, nbds )' )
    else:
        #Initialize a nuclear only potential
        potclass = eval( 'nuc_only_potential.' + potype + '( potparams, nnuc, nbds)' )

    print( 'The potential has been set to',potclass.potname )

    return potclass

######### PARENT POTENTIAL CLASS ##########

class potential(ABC):

    #####################################################################

    @abstractmethod
    def __init__( self, potname, potparams, nstates, nnuc, nbds ):

        self.potname   = potname #string corresponding to the name of the potential
        self.potparams = potparams #array defining the necessary constants for the potential
        self.nstates   = nstates #number of electronic states
        self.nnuc      = nnuc #number of nuclei
        self.nbds      = nbds #number of beads

        #Initialize set of electronic Hamiltonian matrices and their nuclear derivatives
        self.Hel   = np.zeros( [ nbds, nstates, nstates ] )
        self.d_Hel = np.zeros( [ nbds, nnuc, nstates, nstates ] )

    #####################################################################

    #Compiled (numba) Hamiltonian kernel for the compiled spin-magnus integrator, see jit_hel_kernel.
    #jit_id = 0 means this potential has none, and the integrator keeps the numpy calc_Hel path.
    #A potential with a kernel sets its own jit_id and returns its parameters from jit_params().
    jit_id = 0

    def jit_params( self ):
        return np.zeros( 0 )

    #####################################################################

    def calc_rp_harm_eng( self, nucR, beta_p, mass ):

        #Calculate potential energy associated with harmonic springs between beads

        engpe = 0.0
        for i in range(self.nbds):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                engpe += 0.5 * (1.0/beta_p)**2 * np.sum( mass * ( nucR[i] - nucR[self.nbds-1] )**2 )
            else:
                engpe += 0.5 * (1.0/beta_p)**2 * np.sum( mass * ( nucR[i] - nucR[i-1] )**2 )

        return engpe

    ###############################################################

    def calc_rp_harm_force( self, nucR, beta_p, mass ):

        #Calculate force associated with harmonic springs between beads

        Fharm = np.zeros( [self.nbds, self.nnuc] )

        for i in range(self.nbds):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[self.nbds-1] - nucR[i+1] )
            elif( i == self.nbds-1 ):
                #periodic boundary conditions for the last bead
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[i-1] - nucR[0] )
            else:
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[i-1] - nucR[i+1] )

        return Fharm

    #####################################################################

    def calc_nuc_KE( self, nucP, mass ):

        #Calculate kinetic energy associated with nuclear beads

        engke = 0.5 * np.sum( nucP**2 / mass )

        return engke

    ###############################################################

    def error_wrong_param_numb( self, num ):

        print("ERROR: List potparams does not have enough entries (",num,") for", self.potname,"potential")
        exit()

    ###############################################################

    def calc_NAC(self, nucR):
        self.calc_Hel( nucR )
        self.calc_Hel_deriv( nucR )
        
        Hel = self.Hel
        d_Hel = self.d_Hel
        
        #Function that calculates the non-adiabatic coupling terms from 2-level diabatic surfaces
        NAC = np.zeros([self.nbds, self.nnuc])

        for i in range(self.nbds):
            #symmetrize the Hel
            V0 = (Hel[i,0,0] - Hel[i,1,1]) / 2
            D = Hel[i,0,1]
            d_V0 = (d_Hel[i,:,0,0] - d_Hel[i,:,1,1]) / 2
            d_D = d_Hel[i,:,0,1]

            NAC[i] = ( D * d_V0 - V0 * d_D) / ( D**2 + V0**2 ) / 2

        return NAC

    ###############################################################

    def get_bopes(self, nucR):

        # Calculate the BO PES's by directly diagonalizing the 2-state diabatic Hel
        # NOT need to Make sure that Hel has a symmetric formation, i.e., Hel[0,0] = -Hel[1,1]

        self.calc_Hel( nucR )
        Hel = self.Hel
 
        H_bo = np.zeros((self.nbds, self.nstates))

        if ( Hel.shape != (self.nbds,2,2) ):
            print('ERROR: the diabatic Hel does not have a size of nbds*2*2')
            exit()

        H_bar = (Hel[:,1,1] + Hel[:,0,0])/2
        H_dif = Hel[:,1,1] - Hel[:,0,0]
        Delta = Hel[:,0,1]

        H_bo[:,0] = H_bar - np.sqrt( H_dif**2 / 4 + np.abs(Delta)**2 )
        H_bo[:,1] = H_bar + np.sqrt( H_dif**2 / 4 + np.abs(Delta)**2 )

        return H_bo

    ###############################################################
    
    def get_bopes_derivs(self, nucR):
        self.calc_Hel( nucR )
        self.calc_Hel_deriv( nucR )
        Hel = self.Hel
        d_Hel = self.d_Hel

        d_H_bo = np.zeros((self.nbds, self.nnuc, self.nstates))

        for i in range(self.nbds):
            H_diab = Hel[i]
            dH_diab = d_Hel[i,:]
            if (H_diab.shape != (2,2)):
                print('ERROR: the diabatic Hel does not have a size of 2*2 when BO surfaces are calculated')
                exit()

            #d_H_bo[i,:,0] = -(H_diab[0,0]*dH_diab[:,0,0] + H_diab[0,1]*dH_diab[:,0,1])/np.sqrt(H_diab[0,0]**2 + H_diab[0,1]**2)
            d_H_bo[i,:,1] = (dH_diab[:,0,0]+dH_diab[:,1,1]) / 2 + ( (H_diab[0,0]-H_diab[1,1]) * (dH_diab[:,0,0]-dH_diab[:,1,1]) / 4 + H_diab[0,1] * dH_diab[:,0,1] ) / np.sqrt( (H_diab[0,0]-H_diab[1,1])**2 / 4 + H_diab[0,1]**2 )
            d_H_bo[i,:,0] = (dH_diab[:,0,0]+dH_diab[:,1,1]) / 2 - ( (H_diab[0,0]-H_diab[1,1]) * (dH_diab[:,0,0]-dH_diab[:,1,1]) / 4 + H_diab[0,1] * dH_diab[:,0,1] ) / np.sqrt( (H_diab[0,0]-H_diab[1,1])**2 / 4 + H_diab[0,1]**2 )

        return d_H_bo

    ###############################################################

    @abstractmethod
    def calc_Hel( self ):
        pass

    ###############################################################

    @abstractmethod
    def calc_Hel_deriv( self ):
        pass

    ###############################################################

    @abstractmethod
    def calc_state_indep_eng( self ):
        pass

    ###############################################################

    @abstractmethod
    def calc_state_indep_force( self ):
        pass

    ###############################################################

    @abstractmethod
    def error_check( self ):
        pass

    ###############################################################

####### DEFINED POTENTIALS AS INSTANCES OF PARENT POTENTIAL CLASS #######

class nstate_morse(potential):

    #Class for n-state morse potential with gaussian coupling, see Nandini JPC Lett 2013 and Pengfei JCP 2019

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'n-state morse', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 4 ):
            super().error_wrong_param_numb(4)

        self.Dmat     = potparams[0]
        self.alphamat = potparams[1]
        self.Rmat     = potparams[2]
        self.cvec     = potparams[3]

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):

        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel.fill(0.0)

        for i in range( self.nbds ):
            for j in range( self.nnuc ):

                #position of bead i of nuclei j
                pos     = nucR[i,j]

                #nstate x nstate matrix of position minus R-parameter matrix
                posDif  = ( pos - self.Rmat )

                #calculate vector of diagonal terms of Hel
                diag    = np.diag( self.Dmat ) * ( 1.0 - np.exp( - np.diag(self.alphamat) * np.diag(posDif) ) )**2 + self.cvec

                #calculate vector and then convert to upper-triangle matrix of off-diagonal terms of hel
                iup              = np.triu_indices( self.nstates, 1 )
                offdiag          = self.Dmat[iup] * np.exp( - self.alphamat[iup] * posDif[iup]**2 )
                offdiag_mat      = np.zeros([self.nstates,self.nstates])
                offdiag_mat[iup] = offdiag

                #combine diagonal and off diagonal terms into symmetric Hel
                #adding contributions for each nuclei
                self.Hel[i] += np.diag(diag) + offdiag_mat + offdiag_mat.transpose()

    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.d_Hel.fill(0.0)

        for i in range( self.nbds ):
            for j in range( self.nnuc ):

                #position of bead i of nuclei j
                pos     = nucR[i,j]

                #nstate x nstate matrix of position minus R-parameter matrix
                posDif  = ( pos - self.Rmat )

                #calculate vector of diagonal terms of derivative of Hel
                expvec  = np.exp( - np.diag(self.alphamat) * np.diag(posDif) )
                diag    = 2.0* np.diag( self.Dmat ) * np.diag( self.alphamat ) * ( 1.0 - expvec ) * expvec

                #calculate vector and then convert to upper-triangle matrix of off-diagonal terms of derivative of Hel
                iup              = np.triu_indices( self.nstates, 1 )
                offdiag          = -2.0 * self.Dmat[iup] * self.alphamat[iup] * posDif[iup] * np.exp( - self.alphamat[iup] * posDif[iup]**2 )
                offdiag_mat      = np.zeros([self.nstates,self.nstates])
                offdiag_mat[iup] = offdiag

                #combine diagonal and off diagonal terms into symmetric derivative of Hel
                self.d_Hel[i,j] = np.diag(diag) + offdiag_mat + offdiag_mat.transpose()

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #There is no state independent term for this potential

        return 0.0

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #There is no state independent term for this potential

        return np.zeros_like(nucR)

    ###############################################################

    def error_check( self ):

        if( self.Dmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 1st entry of list potparams should correspond to nstate x nstate D-matrix for n-state morse potential")
            exit()

        if( self.alphamat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nstate x nstate alpha-matrix for n-state morse potential")
            exit()

        if( self.Rmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate x nstate R-matrix for n-state morse potential")
            exit()

        if( self.cvec.shape != (self.nstates,) ):
            print("ERROR: 4th entry of list potparams should correspond to nstate-dimensional c-vector for n-state morse potential")
            exit()

#########################################################################

class ET_with_bath(potential):

    # Class for the Marcus electron-transfer model
    # Bath is defined by the spectral density
    # Bath could be classical (bead of 1) or quantum mechanical (beads of N)

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds):

        super().__init__( 'Marcus ET with bath', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 5 ):
            super().error_wrong_param_numb(5)

        self.kvec = potparams[0] # force constant of the Marcus parabolas, size 1
        self.epsil = potparams[1] # driving force of the reaction, size 1
        self.lbd = potparams[2] # lambda, Marcus reorganization energy, size 1
        self.delta = potparams[3] # constant diabatic coupling, size 1
        self.bathvec = potparams[4] #bath terms including all other vibronic modes, following Ohmic spectral density, has the form of [N_bath, mass, gamma (friction coefficient, gamma=Xi*pi/2, Xi: Kondo parameter), w_b (characteristic freq)]

        #Input error check
        self.error_check()

        #the bath potential follows the Ohmic spectral density
        if self.bathvec[0] != 0:
            self.omega_k = - self.bathvec[3] * np.log( (np.arange(self.bathvec[0])+0.5) / self.bathvec[0] )
            self.c_k     = np.sqrt( 2*self.bathvec[1]*self.bathvec[2]*self.bathvec[3] / np.pi / self.bathvec[0] ) * self.omega_k
            self.k_bath_vec = self.omega_k**2 * self.bathvec[1]

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #Assumes first nuclei corresponds to reaction coordinate
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #Calculate the adiabatic electronic energy for each bead
        #Eg = self.calc_diabatic_energy( nucR[:,0], 0)
        #Ee = self.calc_diabatic_energy( nucR[:,0], 1)

        rxnR = np.copy( nucR[:,0] )

        self.Hel[:,0,0] = 0.5 * self.kvec * ( rxnR + np.sqrt(self.lbd/self.kvec/2) )**2 + self.epsil/2
        self.Hel[:,1,1] = 0.5 * self.kvec * ( rxnR - np.sqrt(self.lbd/self.kvec/2) )**2 - self.epsil/2

        self.Hel[:,0,1] = self.delta
        self.Hel[:,1,0] = self.delta

    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.d_Hel.fill(0.0)
        rxnR = np.copy( nucR[:,0] )
        d_Hel = np.zeros([self.nbds, self.nstates, self.nstates])

        d_Hel[:,0,0] = self.kvec * ( rxnR + np.sqrt(self.lbd/self.kvec/2) ) 
        d_Hel[:,1,1] = self.kvec * ( rxnR - np.sqrt(self.lbd/self.kvec/2) )

        self.d_Hel[:,0,:,:] = d_Hel

    ################################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term
        #H_sb = T_r + \sum_k 1/2 w_k^2 * [r_k + c_k/w_k^2 * R]^2

        eng = 0
        if (self.nnuc == 1):
            return eng #only reaction coordinate. No contribution from bath modes.
        else:
            for ibd in range(self.nbds):
                for inuc in range(self.nnuc-1):
                    eng += 0.5 * self.k_bath_vec[inuc] * ( nucR[ibd, inuc+1] + self.c_k[inuc] / self.k_bath_vec[inuc] * nucR[ibd,0] )**2

        return eng

    ################################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative
        #nucR is the coordinate matrix with a dimension of (nbds, nnuc)
        
        #force from harmonic term with different k for each nuclei
        force = np.zeros([self.nbds, self.nnuc])
        
        if (self.nnuc == 1):
            return force #only reaction coordinate.

        else:
            for ibd in range(self.nbds):
                for inuc in range(self.nnuc-1):

                    force[ibd, 0] -= self.c_k[inuc] * (self.c_k[inuc]/self.k_bath_vec[inuc] * nucR[ibd,0] + nucR[ibd,inuc+1])
                    force[ibd, inuc+1] = - self.k_bath_vec[inuc] * nucR[ibd, inuc+1] - self.c_k[inuc] * nucR[ibd,0]

        return force

    ###############################################################

    def error_check( self ):

        if( self.nstates != 2 ):
            print("ERROR: ET_with_bath only supports two electronic states")

        if( not isinstance(self.kvec, float) ):
            print("ERROR: 1st entry of list potparams should be the force constant of the PESs")
            exit()

        if( not isinstance(self.epsil, float) ):
            print("ERROR: 2nd entry of list potparams should be the ET driving force")
            exit()

        if( self.nnuc != self.bathvec[0] + 1):
            print("ERROR: bath mode number does not equal total nuclear modes -1")
            exit()

#########################################################################


class harm_const_cpl(potential):

    #Class for shifted harmonics with a constant coupling between all states
    #Force constant is same between states, but can differ for different nuclei
    #V = \sum_i 0.5 * k_i * R_i**2 + \sum_i H_el(R_i)
    #with [H_el]_nn = a_in * R_i + c_n / nnuc
    #and [H_el]_nm = delta_nm
    #If 2-states it's the usual spin-boson model with constant coupling

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'shifted harmonics - constant coupling', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 4 ):
            super().error_wrong_param_numb(4)

        self.kvec = potparams[0] #force constants, size nnuc
        self.avec = potparams[1] #linear-coupling to nuclear modes (shift in harmonic potentials), size nnuc x nstates
        self.cvec = potparams[2] #energy shift for different states, size nstates
        self.deltavec = potparams[3] #vector of electronic couplings, vector which should unpack into upper-triangle of electronic hamiltonian

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel.fill(0.0)

        for i in range( self.nbds ):

            #Constant electronic coupling
            iup              =  np.triu_indices( self.nstates, 1 )
            self.Hel[i][iup] =  self.deltavec
            self.Hel[i]      += self.Hel[i].T

            for j in range( self.nnuc ):

                #linear coupling to nuclear modes
                self.Hel[i] += np.diag( self.avec[j,:] * nucR[i,j] )

            #Vertical energy shift for states
            self.Hel[i] += np.diag( self.cvec )

    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.d_Hel.fill(0.0)

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        for j in range( self.nnuc ):
            self.d_Hel[:,j] = np.diag( self.avec[j,:] )

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0.5 * np.sum( self.kvec * nucR**2 )

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term

        #force from harmonic term with different k for each nuclei
        force = -self.kvec * nucR

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for constant coupling harmonic potential")
            exit()

        if( self.avec.shape != (self.nnuc,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc x nstate harmonic-shift matrix for constant coupling harmonic potential")
            exit()

        if( self.cvec.shape != (self.nstates,) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate state energy shift vector for constant coupling harmonic potential")
            exit()

        if( self.deltavec.shape != ( (self.nstates-1)*self.nstates/2,) ):
            print("ERROR: 4th entry of list potparams should correspond to (nstate-1)*nstate/2 coupling vector corresponding to upper triangle of hamiltonian for constant coupling harmonic potential")
            exit()

#########################################################################

class harm_lin_cpl(potential):

    #Class for shifted harmonics where electronic coupling depends linearly on nuclear modes
    #Force constant is same between states, but can differ for different nuclei
    #setting linear couplings to zero reproduces constant coupling potential above
    #See Tamura, Ramon, Bittner, and Burghardt, PRL 2008

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'shifted harmonics - linear coupling', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 3 ):
            super().error_wrong_param_numb(3)

        self.kvec = potparams[0] #force constants, size nnuc, NOTE THAT THIS IS FORCE CONSTANT NOT FREQUENCY!!
        self.amat = potparams[1] #shift in harmonic potential along diagonal, and linear couplings along off-diagonal, size nnuc x nstates x nstates (corresponds to kappa and lambda terms in PRL paper)
        #shift in harmonic potentials, size nnuc x nstates (corresponds to kappa terms in PRL paper)
        self.cmat = potparams[2] #energy shift and constant coupling for different states, size nstates x nstates (corresponds to C-matrix in PRL paper)

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel = np.einsum( 'ijk,ai->ajk', self.amat, nucR )
        self.Hel += self.cmat

    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        self.d_Hel = self.amat[np.newaxis,:,:,:]

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0.5 * np.sum( self.kvec * nucR**2 )

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = -self.kvec * nucR

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for linear coupling harmonic potential")
            exit()

        if( self.amat.shape != (self.nnuc,self.nstates,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc x nstate x nstate harmonic-shift and linear-coupling tensor for linear coupling harmonic potential")
            exit()

        if( self.cmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate x nstate state constant energy/coupling matrix for linear coupling harmonic potential")
            exit()

#########################################################################

class harm_lin_cpl_symmetrized(potential):

    #Class for shifted harmonics where electronic coupling depends linearly on nuclear modes
    #Force constant is same between states, but can differ for different nuclei
    #setting linear couplings to zero reproduces constant coupling potential above
    #See Tamura, Ramon, Bittner, and Burghardt, PRL 2008

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'shifted harmonics - linear coupling - sym', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 3 ):
            super().error_wrong_param_numb(3)

        self.kvec = potparams[0] #force constants, size nnuc, NOTE THAT THIS IS FORCE CONSTANT NOT FREQUENCY!!
        self.amat = potparams[1] #shift in harmonic potential along diagonal, and linear couplings along off-diagonal, size nnuc x nstates x nstates (corresponds to kappa and lambda terms in PRL paper)
        #shift in harmonic potentials, size nnuc x nstates (corresponds to kappa terms in PRL paper)
        self.cmat = potparams[2] #energy shift and constant coupling for different states, size nstates x nstates (corresponds to C-matrix in PRL paper)

        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        self.Hel = np.einsum( 'ijk,ai->ajk', self.amat, nucR )
        self.Hel += self.cmat
        
        #minus the average of diagonal terms
        for i in range(self.nbds):
            self.Hel[i] -= np.mean(np.diag(self.Hel[i])) * np.eye(self.nstates)
        
    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        
        #one-bead d_Hel matrix
        d_Hel = np.copy(self.amat)

        for i in range(self.nnuc):
            d_Hel[i] -= np.mean(np.diag(self.amat[i])) * np.eye(self.nstates)

        #multi-bead d_Hel
        for i in range(self.nbds):
            self.d_Hel[i] = d_Hel

        #self.d_Hel = d_Hel[np.newaxis,:,:,:]

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0.5 * np.sum(self.kvec * nucR ** 2)

        #V/bar (R): the average or diagonal terms
        
        eng += np.mean( np.diag (np.einsum( 'ijk,ai->jk', self.amat, nucR ) + self.cmat*self.nbds) )

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = -self.kvec * nucR

        #minus the average-of-diagonal terms
        avg_amat = np.einsum( 'ijj, ai -> ai', self.amat, np.ones_like(nucR) ) / self.nstates
        force -= avg_amat

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for linear coupling harmonic potential")
            exit()

        if( self.amat.shape != (self.nnuc,self.nstates,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc x nstate x nstate harmonic-shift and linear-coupling tensor for linear coupling harmonic potential")
            exit()

        if( self.cmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate x nstate state constant energy/coupling matrix for linear coupling harmonic potential")
            exit()

#########################################################################

class harm_lin_cpl_sym_2(potential):

    #Class for shifted harmonics where electronic coupling depends linearly on nuclear modes
    #Force constant is same between states, but can differ for different nuclei
    #setting linear couplings to zero reproduces constant coupling potential above
    #See Tamura, Ramon, Bittner, and Burghardt, PRL 2008

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'shifted harmonics - linear coupling - sym', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 3 ):
            super().error_wrong_param_numb(3)

        self.kvec = potparams[0] #force constants, size nnuc, NOTE THAT THIS IS FORCE CONSTANT NOT FREQUENCY!!
        amat = potparams[1] #shift in harmonic potential along diagonal, and linear couplings along off-diagonal, size nnuc x nstates x nstates (corresponds to kappa and lambda terms in PRL paper)
        #shift in harmonic potentials, size nnuc x nstates (corresponds to kappa terms in PRL paper)
        cmat = potparams[2] #energy shift and constant coupling for different states, size nstates x nstates (corresponds to C-matrix in PRL paper)

        self.abar = np.zeros(nnuc)
        self.amat = np.zeros_like(amat)
        for i in range(self.nnuc):
            self.abar[i] = np.mean(np.diag(amat[i]))
            self.amat[i] = amat[i] - self.abar[i] * np.eye(self.nstates)

        self.cbar = np.mean(np.diag(cmat))
        self.cmat = cmat - self.cbar * np.eye(self.nstates)
        #Input error check
        self.error_check()

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc
        self.Hel = np.zeros([self.nbds, self.nstates, self.nstates])
        for bd in range(self.nbds):
            for nuc in range(self.nnuc):
                self.Hel[bd] += self.amat[nuc] * nucR[bd,nuc]
            self.Hel[bd] += self.cmat
        #minus the average of diagonal terms
        
    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc
        for bd in range(self.nbds):
            self.d_Hel[bd] = self.amat

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        eng = 0

        for bd in range(self.nbds):
            for nuc in range(self.nnuc):
                eng += self.kvec[nuc] * nucR[bd,nuc] ** 2 / 2 + self.abar[nuc] * nucR[bd,nuc]
            eng += self.cbar
        
        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = np.zeros([self.nbds, self.nnuc])
        for bd in range(self.nbds):
            for nuc in range(self.nnuc):
                force[bd,nuc] -= self.kvec[nuc] * nucR[bd,nuc] + self.abar[nuc]

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for linear coupling harmonic potential")
            exit()

        if( self.amat.shape != (self.nnuc,self.nstates,self.nstates) ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc x nstate x nstate harmonic-shift and linear-coupling tensor for linear coupling harmonic potential")
            exit()

        if( self.cmat.shape != (self.nstates,self.nstates) ):
            print("ERROR: 3rd entry of list potparams should correspond to nstate x nstate state constant energy/coupling matrix for linear coupling harmonic potential")
            exit()

#########################################################################

class pengfei_polariton(potential):

    #Class for pengfei's polariton model
    #A. Mandal, P. Huo JPC Lett 2019
    #Just work with Hamiltonian restricted to |e,0> and |g,1> subspace

    ###############################################################

    def __init__( self, potparams, nstates, nnuc, nbds, leak = False ):

        super().__init__( 'pengfei_polariton', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 7 and leak == False ):
            super().error_wrong_param_numb(7)
        
        elif( leak == True and len(potparams) != 8):
            super().error_wrong_param_numb(8)

        self.wc = potparams[0] #frequency of photon mode
        self.gc = potparams[1] #coupling to photon mode
        self.Amat = potparams[2] #A-terms in electronic states, first index is the adiabatic state ie A[0,0]=A1, A[0,1]=A2, A[1,0]=A3, A[1,1]=A4 in paper
        self.Bmat = potparams[3] #B-terms as above
        self.Rmat = potparams[4] #R-terms as above
        self.Dvec = potparams[5] #D-terms with D[0]=D1 and D[1]=D2 from paper
        self.bathvec = potparams[6] #bath terms including all other vibronic modes, following Ohmic spectral density, has the form of [N_bath, Xi (Kondo param), w_b (characteristic freq)]
        
        if (leak == True):
            self.W = np.sqrt(potparams[7]/2/np.pi) #the coupling between internal and external modes, defined by \sqrt{\gamma/2pi}, where \gamma is the damping coefficient
        
        #Input error check
        self.error_check()

        #the bath potential follows the Ohmic spectral density
        if self.bathvec[0] != 0:
            self.omega_k = - self.bathvec[2] * np.log( 1 - (np.arange(self.bathvec[0])+1) * ( 1-np.exp(-4) ) / self.bathvec[0] )
            self.c_k     = np.sqrt( self.bathvec[1] * 4 * self.bathvec[2] * (1-np.exp(-4)) / self.bathvec[0] ) * self.omega_k

    ###############################################################

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #Assumes first nuclei corresponds to reaction coordinate
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #Calculate the adiabatic electronic energy for each bead
        #Eg = self.calc_diabatic_energy( nucR[:,0], 0)
        #Ee = self.calc_diabatic_energy( nucR[:,0], 1)

        rxnR = np.copy( nucR[:,0] )

        Vmat = np.zeros([self.nbds,self.nstates,self.nstates])

        Vmat[:,0,0] = self.Amat[0,0] + self.Bmat[0,0] * ( rxnR - self.Rmat[0,0] )**2
        Vmat[:,0,1] = self.Amat[0,1] + self.Bmat[0,1] * ( rxnR - self.Rmat[0,1] )**2
        Vmat[:,1,0] = self.Amat[1,0] + self.Bmat[1,0] * ( rxnR - self.Rmat[1,0] )**2
        Vmat[:,1,1] = self.Amat[1,1] + self.Bmat[1,1] * ( rxnR - self.Rmat[1,1] )**2

        self.Hel = np.zeros([self.nbds,self.nstates,self.nstates])

        self.Hel[:,0,0] = 0.5*( Vmat[:,0,0] + Vmat[:,0,1] ) - np.sqrt( self.Dvec[0]**2 + 0.25* ( Vmat[:,0,0] - Vmat[:,0,1] )**2 )
        self.Hel[:,1,1] = 0.5*( Vmat[:,1,0] + Vmat[:,1,1] ) - np.sqrt( self.Dvec[1]**2 + 0.25* ( Vmat[:,1,0] - Vmat[:,1,1] )**2 )

        #Need to add all the gc wc stuff to the hamiltonian

        self.Hel[:,0,0] += 1.5 * self.wc
        self.Hel[:,1,1] += 0.5 * self.wc

        self.Hel[:,0,1] += self.gc
        self.Hel[:,1,0] += self.gc


    ###############################################################

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        rxnR = np.copy( nucR[:,0] )

        Vmat = np.zeros([self.nbds,self.nstates,self.nstates])
        dVmat = np.zeros([self.nbds,self.nstates,self.nstates])
        d_Hel = np.zeros([self.nbds,self.nstates,self.nstates])

        Vmat[:,0,0] = self.Amat[0,0] + self.Bmat[0,0] * ( rxnR - self.Rmat[0,0] )**2
        Vmat[:,0,1] = self.Amat[0,1] + self.Bmat[0,1] * ( rxnR - self.Rmat[0,1] )**2
        Vmat[:,1,0] = self.Amat[1,0] + self.Bmat[1,0] * ( rxnR - self.Rmat[1,0] )**2
        Vmat[:,1,1] = self.Amat[1,1] + self.Bmat[1,1] * ( rxnR - self.Rmat[1,1] )**2

        dVmat[:,0,0] = 2 * self.Bmat[0,0] * ( rxnR - self.Rmat[0,0] )
        dVmat[:,0,1] = 2 * self.Bmat[0,1] * ( rxnR - self.Rmat[0,1] )
        dVmat[:,1,0] = 2 * self.Bmat[1,0] * ( rxnR - self.Rmat[1,0] )
        dVmat[:,1,1] = 2 * self.Bmat[1,1] * ( rxnR - self.Rmat[1,1] )

        d_Hel[:,0,0] = 0.5 * ( dVmat[:,0,0] + dVmat[:,0,1] ) - ( Vmat[:,0,0] - Vmat[:,0,1] ) * (dVmat[:,0,0] - dVmat[:,0,1]) / ( 4 * np.sqrt( self.Dvec[0]**2 + 0.25* ( Vmat[:,0,0] - Vmat[:,0,1] )**2 ) )
        d_Hel[:,1,1] = 0.5 * ( dVmat[:,1,0] + dVmat[:,1,1] ) - ( Vmat[:,1,0] - Vmat[:,1,1] ) * (dVmat[:,1,0] - dVmat[:,1,1]) / ( 4 * np.sqrt( self.Dvec[1]**2 + 0.25* ( Vmat[:,1,0] - Vmat[:,1,1] )**2 ) )
        self.d_Hel = np.zeros([self.nbds, self.nnuc, self.nstates, self.nstates])
        self.d_Hel[:,0,:,:] = d_Hel

    ################################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term
        #H_sb = T_r + \sum_k 1/2 w_k^2 * [r_k + c_k/w_k^2 * R]^2

        eng = 0
        if (self.nnuc == 1):
            return eng #only reaction coordinate. No contribution from bath modes.
        else:
            for ibd in range(self.nbds):
                for inuc in range(self.nnuc-1):
                    eng += 0.5 * self.omega_k[inuc]**2 * ( nucR[ibd, inuc+1] + self.c_k[inuc] / (self.omega_k[inuc]**2) * nucR[ibd,0] )**2

        return eng

    ################################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative
        #nucR is the coordinate matrix with a dimension of (nbds, nnuc)
        
        #force from harmonic term with different k for each nuclei
        force = np.zeros([self.nbds, self.nnuc])
        
        if (self.nnuc == 1):
            return force #only reaction coordinate.

        else:
            for ibd in range(self.nbds):
                for inuc in range(self.nnuc-1):

                    force[ibd, 0] -= self.c_k[inuc] * (self.c_k[inuc]/self.omega_k[inuc]**2 * nucR[ibd,0] + nucR[ibd,inuc+1])
                    force[ibd, inuc+1] = - self.omega_k[inuc]**2 * nucR[ibd, inuc+1] - self.c_k[inuc] * nucR[ibd,0]

        return force

    ###############################################################

    def error_check( self ):

        if( self.nstates != 2 ):
            print("ERROR: pengfei_polariton only supports two electronic states")

        if( not isinstance(self.wc, float) ):
            print("ERROR: 1st entry of list potparams should be the frequency of photon mode")
            exit()

        if( not isinstance(self.gc, float) ):
            print("ERROR: 2nd entry of list potparams should be the constant coupling to photon mode")
            exit()

        if( self.Amat.shape != (2,2) ):
            print("ERROR: 3rd entry of list potparams should correspond to 2x2 array of A-terms in electronic states")
            exit()

        if( self.Bmat.shape != (2,2) ):
            print("ERROR: 4th entry of list potparams should correspond to 2x2 array of B-terms in electronic states")
            exit()

        if( self.Rmat.shape != (2,2) ):
            print("ERROR: 5th entry of list potparams should correspond to 2x2 array of R-terms in electronic states")
            exit()

        if( self.Dvec.shape != (2,) ):
            print("ERROR: 6th entry of list potparams should correspond to 2-dim vector of D-terms in electronic states")
            exit()

        if( self.bathvec.shape != (3,) ):
            print("ERROR: 7th entry of list potparams should correspond to 3-dim vector of bath term for the state-independent potential")
            exit()

        if( self.nnuc != self.bathvec[0] + 1):
            print("ERROR: bath mode number does not equal total nuclear modes -1")
            exit()

#########################################################################

class isolated_elec(potential):

    #class for isolated constant electronic potential
    #electronic hamiltonian is constant wrt to all variables
    #no state independent potential either

    def __init__(self, potparams, nstates, nnuc, nbds):

        super().__init__( 'isolated electronic', potparams, nstates, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 1 ):
            super().error_wrong_param_numb(1)

        self.Helec = potparams[0] #electronic hamiltonian that is constant

        #Input error check
        self.error_check()

        self.Hel[:] = self.Helec #set all bead hamiltonians to be the same
        self.d_Hel.fill(0.0) #since hamiltonian is constant the derivative is 0

    def calc_Hel(self, *args):
        #hamiltonian is constant so no need to update it
        pass

    def calc_Hel_deriv(self, *args):
        #hamiltonian derivative is constant so no need to update it
        pass

    def calc_state_indep_eng(self, *args):
        #no state independent potential
        return 0.0

    def calc_state_indep_force(self, *args):
        #no state independent potential
        return 0.0

    def error_check(self):
        if (self.Helec.shape != (self.nstates, self.nstates)):
            print("ERROR: 1st entry of list potparams should correspond to full constant electronic hamiltonian")
            exit()

######### COMPILED HAMILTONIAN KERNELS ##########

@njit( cache=True )
def _tully_1ac_hel( nucR, prm, Hel, d_Hel ):

    #Compiled twin of tully_1ac.calc_Hel + calc_Hel_deriv, same expressions in the same order.
    #Fills Hel (n,2,2) and d_Hel (n,1,2,2) for the n rows of nucR (n,1); prm = (a, b, c, d)
    a = prm[0]; b = prm[1]; c = prm[2]; d = prm[3]
    for i in range( nucR.shape[0] ):
        R = nucR[i,0]
        Hel[i,0,0] = a * np.tanh( b * R )
        Hel[i,1,1] = -Hel[i,0,0]
        Hel[i,0,1] = c * np.exp( -d * R**2 )
        Hel[i,1,0] = Hel[i,0,1]
        d_Hel[i,0,0,0] = a * b / np.cosh( b * R ) ** 2
        d_Hel[i,0,1,1] = -d_Hel[i,0,0,0]
        d_Hel[i,0,0,1] = -2 * c * d * R * np.exp( -d * R**2 )
        d_Hel[i,0,1,0] = d_Hel[i,0,0,1]

@njit( cache=True )
def jit_hel_kernel( jit_id, nucR, prm, Hel, d_Hel ):

    #Single entry point for the compiled kernels, chosen by the potential's jit_id. Giving another
    #potential a compiled path means writing its kernel above and adding a branch here.
    if( jit_id == 1 ):
        _tully_1ac_hel( nucR, prm, Hel, d_Hel )

#############################################

class tully_1ac(potential):

    #Class for Tully's single avoided crossing(1ac) model.
    #All Tully models only contain 1 nulcear DOF
    #See Mannouch, Richardson, JCP 2023

    jit_id = 1   #compiled kernel: _tully_1ac_hel

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'single avoided crossing Tully', potparams, nstates, nnuc, nbds )

        self.a = potparams[0]
        self.b = potparams[1]
        self.c = potparams[2]
        self.d = potparams[3]

        #Input error check
        self.error_check()

    def jit_params( self ):
        return np.array( [ np.ravel( self.a )[0], np.ravel( self.b )[0],
                           np.ravel( self.c )[0], np.ravel( self.d )[0] ], dtype=np.float64 )

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc, where nnuc is only 1

        self.Hel[:,0,0] = self.a * np.tanh( self.b * nucR[:,0] )
        self.Hel[:,1,1] = -self.Hel[:,0,0]
        self.Hel[:,0,1] = self.Hel[:,1,0] = self.c * np.exp( -self.d * nucR[:,0]**2 )

    def calc_Hel_deriv( self, nucR ):

        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        self.d_Hel[:,:,0,0] = self.a * self.b / np.cosh( self.b * nucR ) ** 2
        self.d_Hel[:,:,1,1] = - self.d_Hel[:,:,0,0]
        self.d_Hel[:,:,0,1] = self.d_Hel[:,:,1,0] = -2 * self.c * self.d * nucR * np.exp( -self.d * nucR**2 )


    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = 0

        return force

    ###############################################################

    def error_check( self ):

        if( self.nnuc != 1):
            print('ERROR: the number of the nuclear DOF is not 1')
            exit()

        if( self.nstates != 2):
            print("ERROR: the number of electronic states is not 2")
            exit()

        if( self.a.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should have the same dimension with the nnuc")
            exit()

        if( self.b.shape != (self.nnuc,) ):
            print("ERROR: 2nd entry of list potparams should be nnuc-dimensional")
            exit()

        if( self.c.shape != (self.nnuc,) ):
            print("ERROR: 3rd entry of list potparams should be nnuc-dimensional")
            exit()

        if( self.d.shape != (self.nnuc,) ):
            print("ERROR: 4th entry of list potparams should be nnuc-dimensional")
            exit()

    ###############################################################

class tully_2ac(potential):

    #Class for Tully's dual avoided crossing(2ac) model.
    #All Tully models only contain 1 nulcear DOF
    #See Mannouch, Richardson, JCP 2023

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'dual avoided crossing Tully', potparams, nstates, nnuc, nbds )

        self.a = potparams[0]
        self.b = potparams[1]
        self.c = potparams[2]
        self.d = potparams[3]
        self.e = potparams[4]

        #Input error check
        self.error_check()

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc, where nnuc is only 1

        self.Hel[:,0,0] = np.sum(0.5 * ( self.a * np.exp( -self.b * nucR**2 ) - self.e ), axis = 1)
        self.Hel[:,1,1] = -self.Hel[:,0,0]
        self.Hel[:,0,1] = self.Hel[:,1,0] = np.sum( self.c * np.exp( -self.d * nucR**2 ), axis = 1 )

    def calc_Hel_deriv( self, nucR ):
        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        self.d_Hel[:,:,0,0] = -self.a * self.b * np.exp( - self.b * nucR**2 ) * nucR
        self.d_Hel[:,:,1,1] = - self.d_Hel[:,:,0,0]
        self.d_Hel[:,:,0,1] = self.d_Hel[:,:,1,0] = -2 * self.c * self.d * nucR * np.exp( -self.d * nucR**2 )

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = np.sum( -0.5 * ( self.a * np.exp( -self.b * nucR**2 ) - self.e ) )

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = -self.a * self.b * nucR * np.exp( - self.b * nucR**2 )

        return force

    ###############################################################

    def error_check( self ):

        if( self.nnuc != 1):
            print('ERROR: the number of the nuclear DOF is not 1')
            exit()

        if( self.nstates != 2):
            print("ERROR: the number of electronic states is not 2")
            exit()

        if( self.a.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should have the same dimension with the nnuc")
            exit()

        if( self.b.shape != (self.nnuc,) ):
            print("ERROR: 2nd entry of list potparams should be nnuc-dimensional")
            exit()

        if( self.c.shape != (self.nnuc,) ):
            print("ERROR: 3rd entry of list potparams should be nnuc-dimensional")
            exit()

        if( self.d.shape != (self.nnuc,) ):
            print("ERROR: 4th entry of list potparams should be nnuc-dimensional")
            exit()

        if( self.d.shape != (self.nnuc,) ):
            print("ERROR: 5th entry of list potparams should be nnuc-dimensional and it corresponds to the vertical energy shifts for both states")
            exit()
    ###############################################################

class tully_EC(potential):

    #Class for Tully's extended coupling model.
    #All Tully models only contain 1 nulcear DOF
    #See Mannouch, Richardson, JCP 2023

    def __init__( self, potparams, nstates, nnuc, nbds ):

        super().__init__( 'extended coupling Tully', potparams, nstates, nnuc, nbds )

        self.a = potparams[0]
        self.b = potparams[1]
        self.c = potparams[2]

        #Input error check
        self.error_check()

    def calc_Hel( self, nucR ):
        #Subroutine to calculate set of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc, where nnuc is only 1

        self.Hel[:,0,0] = -self.a
        self.Hel[:,1,1] = -self.Hel[:,0,0]

        self.Hel[:,0,1] = self.Hel[:,1,0] = np.sum(self.b * (1 + np.sign(nucR) * ( 1 - np.exp( -self.c * np.abs(nucR) ) )), axis = 1)

    def calc_Hel_deriv( self, nucR ):
        #Subroutine to calculate set of nuclear derivative of electronic Hamiltonian matrices for each bead
        #nucR is the nuclear positions and is of dimension nbds x nnuc

        #linear coupling to nuclear modes leads to same derivative for all beads for each nuclei
        self.d_Hel[:,:,0,0] = 0 * nucR
        self.d_Hel[:,:,1,1] = - self.d_Hel[:,:,0,0]
        
        self.d_Hel[:,:,0,1] = self.d_Hel[:,:,1,0] = self.b*self.c * np.exp( -self.c * np.abs(nucR) )

    ###############################################################

    def calc_state_indep_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the state independent term

        #harmonic term with different k for each nuclei
        eng = 0

        return eng

    ###############################################################

    def calc_state_indep_force( self, nucR ):
        #Subroutine to calculate the force associated with the state independent term
        #Note that this corresponds to the negative derivative

        #force from harmonic term with different k for each nuclei
        force = 0

        return force

    ###############################################################

    def error_check( self ):

        if( self.nnuc != 1):
            print('ERROR: the number of the nuclear DOF is not 1')
            exit()

        if( self.nstates != 2):
            print("ERROR: the number of electronic states is not 2")
            exit()

        if( self.a.shape != (self.nnuc,) ):
            print("ERROR: 1st entry of list potparams should have the same dimension with the nnuc")
            exit()

        if( self.b.shape != (self.nnuc,) ):
            print("ERROR: 2nd entry of list potparams should be nnuc-dimensional")
            exit()

        if( self.c.shape != (self.nnuc,) ):
            print("ERROR: 3rd entry of list potparams should be nnuc-dimensional")
            exit()

    ###############################################################
