#!/usr/bin/python

import numpy as np
import scipy.linalg as la
import scipy.special

#####################################################################

def diagonalize(H,S=None):
    #subroutine to solve the general eigenvalue problem HC=SCE
    #returns the matrix of eigenvectors C, and a 1-d array of eigenvalues
    #NOTE that H must be Hermitian

    if S is None:
        S = np.identity(len(H))

    E,C = la.eigh(H,S)

    return E,C

#####################################################################

def rot1el( h_orig, rotmat ):
    #subroutine to rotate one electron integrals

    tmp = np.dot( h_orig, rotmat )
    if( np.iscomplexobj(rotmat) ):
        h_rot = np.dot( rotmat.conjugate().transpose(), tmp )
    else:
        h_rot = np.dot( rotmat.transpose(), tmp )

    return h_rot

#####################################################################

def rot2el_chem( V_orig, rotmat ):
    #subroutine to rotate two electron integrals, V_orig must be in chemist notation

    #V_orig starts as Nb x Nb x Nb x Nb and rotmat is Nb x Ns

    if( np.iscomplexobj(rotmat) ):
        rotmat_conj = rotmat.conjugate().transpose()
    else:
        rotmat_conj = rotmat.transpose()

    V_new = np.einsum( 'trus,sy -> truy', V_orig, rotmat )
    #V_new now Nb x Nb x Nb x Ns

    V_new = np.einsum( 'vu,truy -> trvy', rotmat_conj, V_new )
    #V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum( 'trvy,rx -> txvy', V_new, rotmat )
    #V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum( 'wt,txvy -> wxvy', rotmat_conj, V_new )
    #V_new now Ns x Ns x Ns x Ns

    return V_new

#####################################################################

def rot2el_phys( V_orig, rotmat ):
    #subroutine to rotate two electron integrals, V_orig must be in physics notation
    #Returns V_new in physics notation

    #V_orig starts as Nb x Nb x Nb x Nb and rotmat is Nb x Ns

    if( np.iscomplexobj(rotmat) ):
        rotmat_conj = rotmat.conjugate().transpose()
    else:
        rotmat_conj = rotmat.transpose()

    V_new = np.einsum( 'turs,sy -> tury', V_orig, rotmat )
    #V_new now Nb x Nb x Nb x Ns

    V_new = np.einsum( 'tury,rx -> tuxy', V_new, rotmat )
    #V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum( 'vu,tuxy -> tvxy', rotmat_conj, V_new )
    #V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum( 'wt,tvxy -> wvxy', rotmat_conj, V_new )
    #V_new now Ns x Ns x Ns x Ns

    return V_new

#####################################################################

def commutator( Mat1, Mat2 ):
    #subroutine to calculate the commutator of two matrices

    return np.dot(Mat1,Mat2) - np.dot(Mat2,Mat1)

#####################################################################

def matprod( Mat1, *args ):
    #subroutine to calculate matrix product of arbitrary number of matrices

    Result = Mat1
    for Mat in args:
        Result = np.dot( Result, Mat )

    return Result

#####################################################################

def adjoint( Mat ):
    #subroutine to calculate the conjugate transpose (ie adjoint) of a matrix

    return np.conjugate( np.transpose( Mat ) )

#####################################################################

def chemps2_to_pyscf_CIcoeffs( CIcoeffs_chemps2, Norbs, Nalpha, Nbeta ):
    #subroutine to unpack the 1d vector of CI coefficients obtained from a FCI calculation using CheMPS2
    #to the correctly formatted 2d-array of CI coefficients for use with pyscf

    Nalpha_string = scipy.special.binom( Norbs, Nalpha )
    Nbeta_string = scipy.special.binom( Norbs, Nbeta )

    CIcoeffs_pyscf = np.reshape( CIcoeffs_chemps2, (Nalpha_string, Nbeta_string), order='F' )

    return CIcoeffs_pyscf

#####################################################################

def matrix2array( mat, diag=False ):

    #Subroutine to flatten a symmetric matrix into a 1d array
    #Returns a 1d array corresponding to the upper triangle of the symmetric matrix
    #if diag=True, all diagonal elements of the matrix should be the same
    #and first index of 1d array will be the diagonal term, and the rest the upper triagonal of the matrix

    if( diag ):
        array = mat[ np.triu_indices( len(mat),1 ) ] 
        array = np.insert(array, 0, mat[0,0])
    else:
        array = mat[ np.triu_indices( len(mat) ) ] 

    return array

#####################################################################

def array2matrix( array, diag=False ):

    #Subroutine to unpack a 1d array into a symmetric matrix
    #Returns a symmetric matrix
    #if diag=True, all diagonal elements of the returned matrix will be the same corresponding to the first element of the 1d array

    if( diag ):
        dim = (1.0+np.sqrt(1-8*( 1-len(array) )))/2.0
    else:
        dim = (-1.0+np.sqrt(1+8*len(array)))/2.0

    mat = np.zeros( [dim,dim] )

    if( diag ):
        mat[ np.triu_indices(dim,1) ] = array[1:]
        np.fill_diagonal( mat, array[0] )
    else:
        mat[ np.triu_indices(dim) ] = array

    mat = mat + mat.transpose() - np.diag(np.diag(mat))

    return mat 

#####################################################################

def matrix2array_nosym( mat, diag=False ):

    #Subroutine to flatten a general matrix into a 1d array
    #Returns a 1d array corresponding to the upper triangle of the symmetric matrix
    #if diag=True, all diagonal elements of the matrix should be the same
    #and first index of 1d array will be the diagonal term, and the rest the upper triagonal of the matrix

    if( diag ):
        array = mat[ np.triu_indices( len(mat),1 ) ] 
        array = np.insert(array, 0, mat[0,0])
    else:
        array = mat[ np.triu_indices( len(mat) ) ] 

    return array

#####################################################################

def make_histo( data, Nbins=100, minval=None, maxval=None ):

    histo = np.zeros( [Nbins,2] )

    #Set minimum and maximum of histogram
    if( minval is None ):
        minval = data.min()

    if( maxval is None ):
        maxval = data.max()

    #calculate normalized histogram
    histo[:,1], bin_edges   = np.histogram( data, bins=Nbins, range=( minval, maxval ), density=True )

    #calculate midpoint of each bin
    for j in range( Nbins ):
        histo[j,0] = ( bin_edges[j] + bin_edges[j+1] ) / 2.0

    return histo

#####################################################################

def printarray( array, filename='array.dat', long_fmt=False ):
    #subroutine to print out an ndarry of 2,3 or 4 dimensions to be read by humans

    dim = len(array.shape)

    filehandle = open(filename,'w')

    comp_log = np.iscomplexobj( array )

    if( comp_log ):

        if( long_fmt ):
            #fmt_str = '%20.8e%+.8ej'
            fmt_str = '%25.14e%+.14ej'
        else:
            fmt_str = '%10.4f%+.4fj'

    else:

        if( long_fmt ):
            fmt_str = '%20.8e'
        else:
            fmt_str = '%8.4f'


    if ( dim == 1 ):

        Ncol = 1
        np.savetxt(filehandle, array, fmt_str*Ncol )

    elif ( dim == 2 ):

        Ncol = array.shape[1]
        np.savetxt(filehandle, array, fmt_str*Ncol )

    elif ( dim == 3 ):

        for dataslice in array:
            Ncol = dataslice.shape[1]
            np.savetxt(filehandle, dataslice, fmt_str*Ncol )
            filehandle.write('\n')

    elif ( dim == 4 ):

        for i in range( array.shape[0] ):
            for dataslice in array[i,:,:,:]:
                Ncol = dataslice.shape[1]
                np.savetxt(filehandle, dataslice, fmt_str*Ncol )
                filehandle.write('\n')
            filehandle.write('\n')

    else:
        print('ERROR: Input array for printing is not of dimension 2, 3, or 4')
        exit()

    filehandle.close()

#####################################################################

def readarray( filename='array.dat' ):
    #subroutine to read in arrays generated by the printarray subroutine defined above
    #currently only works with 1d or 2d arrays

    array = np.loadtxt( filename, dtype = np.complex128 )

    chk_cmplx = np.any( np.iscomplex( array ) )

    if( not chk_cmplx ):
        array = np.copy( np.real( array ) )

    return array

#####################################################################

def friction_kernel( t, gamma, omega ):

    #Generalized-Langevin friction memory kernel K_k(t) for the ring-polymer normal modes
    #
    #  K_k(t) = -gamma*omega_k
    #           + gamma*omega_k^2 * t * J0(omega_k t) * ( 1 - (pi/2) H1(omega_k t) )
    #           - gamma*omega_k   * J1(omega_k t) * ( 1 - (pi/2) omega_k t H0(omega_k t) )
    #
    #where J0,J1 are Bessel functions of the first kind and H0,H1 are Struve functions.
    #This returns only the CONTINUOUS part of the kernel; the singular 2*gamma*delta(t)
    #contribution is handled by the caller (which discretizes the t=0 entry). At t=0 the
    #continuous part reduces to -gamma*omega_k.
    #
    #t     - 1d array of time points, shape (nt,)
    #gamma - friction coefficient (scalar)
    #omega - (normal-mode) frequencies omega_k: 1d array of shape (nk,), or a single scalar
    #
    #Returns K of shape (nk, nt) with K[k,j] = K_{omega_k}(t_j), dtype float64.
    #If omega is passed as a scalar, the leading axis is dropped and a 1d array of shape (nt,)
    #is returned instead.

    omega_scalar = np.ndim( omega ) == 0                            # remember if omega came in scalar
    t     = np.atleast_1d( np.asarray( t,     dtype=np.float64 ) )   # (nt,)
    omega = np.atleast_1d( np.asarray( omega, dtype=np.float64 ) )   # (nk,)

    wt = np.outer( omega, t )   # (nk, nt) = omega_k * t_j

    J0 = scipy.special.jv( 0, wt )
    J1 = scipy.special.jv( 1, wt )
    H0 = scipy.special.struve( 0, wt )
    H1 = scipy.special.struve( 1, wt )

    w = omega[:, np.newaxis]    # (nk, 1)

    term1 = -gamma * w                                                        # broadcasts to (nk, nt)
    term2 =  gamma * w**2 * t[np.newaxis,:] * J0 * ( 1 - (np.pi/2) * H1 )
    term3 = -gamma * w * J1 * ( 1 - (np.pi/2) * wt * H0 )

    K = term1 + term2 + term3      # (nk, nt)
    return K[0] if omega_scalar else K   # (nt,) for scalar omega, else (nk, nt)

#####################################################################

def fluctuating_coeffs( delt, beta_p, gamma, omega_k, N, rng=None ):

    #Draw the fluctuating-force noise coefficients a_k^(j), b_k^(j) (Eq. A58 of Lawrence et al.,
    #JCP 151, 114119 (2019)) ONCE for a single stochastic realization. F_k^(i) at every time index
    #then reuses these SAME coefficients (only the i-dependent phase changes); see fluctuating_force.
    #
    #Built from the bath spectral density J_k(omega) (Eq. A39, Ohmic) and the noise weight
    #G_k(omega) (Eq. A52), sampled on the frequency grid omega_j = j*dw, j = 0..N, with
    #dw = pi/(N*delt) (omega_N = pi/delt is the Nyquist frequency).
    #
    #delt    - time step dt
    #beta_p  - bead inverse temperature (beta_n)
    #gamma   - friction coefficient
    #omega_k - 1d array of ring-polymer normal-mode frequencies, shape (nbds,)
    #N       - number of frequency-grid intervals (large integer)
    #
    #Returns (akj, bkj), each a 2d array of shape (nbds, N+1)

    omega_k = np.atleast_1d( np.asarray( omega_k, dtype=np.float64 ) )   # (nbds,)
    nbds    = omega_k.size

    #Frequency grid omega_j = j*dw, j = 0..N
    dw      = np.pi / ( N * delt )
    omega_j = np.arange( N + 1 ) * dw                                    # (N+1,)

    #Eq. (A39) Ohmic spectral density: J_k(omega) = theta(omega-omega_k) * gamma * sqrt(omega^2 - omega_k^2)
    arg = omega_j[np.newaxis,:]**2 - omega_k[:,np.newaxis]**2            # (nbds, N+1)
    Jk  = gamma * np.sqrt( np.clip( arg, 0.0, None ) )                   # (nbds, N+1); 0 for omega < omega_k

    #Eq. (A52) noise weight: G_k(omega) = sqrt( 2/(pi beta_p) * J_k(omega)/omega )  (set to 0 at omega = 0)
    ratio = np.divide( Jk, omega_j[np.newaxis,:], out=np.zeros_like(Jk),
                       where=( omega_j[np.newaxis,:] > 0.0 ) )           # (nbds, N+1)
    Gk    = np.sqrt( 2.0 / ( np.pi * beta_p ) * ratio )                  # (nbds, N+1)

    #Frequency-integral trapezoid weight sqrt(dw), halved at the two endpoints j = 0 and j = N (Eq. A58)
    weight    = np.full( N + 1, np.sqrt(dw) )
    weight[0] = np.sqrt(dw) / 2
    weight[N] = np.sqrt(dw) / 2

    #Independent unit-Gaussian noise, one draw per (mode, frequency) so the fluctuating forces of
    #different normal modes are uncorrelated (each mode has its own bath / spectral density).
    gen = rng if rng is not None else np.random   # seeded Generator (reproducible) or the global RNG
    eps_aj = gen.standard_normal( (nbds, N + 1) )                        # (nbds, N+1)
    eps_bj = gen.standard_normal( (nbds, N + 1) )                        # (nbds, N+1)

    #Eq. (A58a,b) coefficients a_k^(j), b_k^(j) as 2d arrays (nbds, N+1)
    akj = Gk * eps_aj * weight[np.newaxis,:]
    bkj = Gk * eps_bj * weight[np.newaxis,:]

    return akj, bkj

#####################################################################

def fluctuating_force( i, akj, bkj ):

    #Colored fluctuating (random) force F_k^(i) at time-step index i (Eq. A57 of Lawrence et al.,
    #JCP 151, 114119 (2019)), evaluated from PRE-DRAWN noise coefficients a_k^(j), b_k^(j).
    #The coefficients are fixed for a single stochastic realization (see fluctuating_coeffs); only
    #the i-dependent phase changes with the time index, so calling this at successive i traces out
    #one continuous colored-noise trajectory rather than independent draws.
    #
    #i        - integer time-step index
    #akj, bkj - noise coefficients of shape (nbds, N+1) from fluctuating_coeffs
    #
    #Returns F_k^(i) as a 1d array of shape (nbds,)

    N = akj.shape[1] - 1                                                 #frequency-grid size from coeff shape

    #Eq. (A57): F_k^(i) = sum_{j=0}^{N} a_k^(j) cos(i j pi/N) + b_k^(j) sin(i j pi/N)
    #contract the frequency index j of the 2d coefficients against the trig factors -> (nbds,)
    phase = i * np.arange( N + 1 ) * np.pi / N                           # (N+1,)
    F_k   = akj @ np.cos( phase ) + bkj @ np.sin( phase )                # (nbds,)

    return F_k

#####################################################################

def fluctuating_force_series( akj, bkj ):

    #Precompute the ENTIRE periodic fluctuating-force trajectory F_k^(n), n = 0..2N-1, in ONE FFT,
    #replacing the O(N) per-step sinusoid sum of fluctuating_force (Eq. A57). Since the phase in Eq.
    #(A57) is n j pi/N, F_k^(n) is periodic in n with period 2N (N = akj.shape[1]-1) and equals
    #    F_k^(n) = Re{ sum_{j=0}^{N} ( a_k^(j) - i b_k^(j) ) exp( 2 pi i n j / (2N) ) },
    #i.e. the real part of the inverse DFT of the (N+1) coefficients (a_k - i b_k) zero-padded to
    #length 2N. Evaluating at successive n therefore just indexes this array (mod 2N), turning the
    #per-step cost from O(N) into O(nbds). Verified bit-equivalent to fluctuating_force to ~1e-13.
    #
    #akj, bkj - noise coefficients of shape (nbds, N+1) from fluctuating_coeffs
    #
    #Returns F_k^(n) for n = 0..2N-1 as a 2d array of shape (nbds, 2N)

    nbds = akj.shape[0]
    N    = akj.shape[1] - 1
    M    = 2 * N                                                         #period of the noise (steps)

    C = np.zeros( (nbds, M), dtype=complex )
    C[:, :N+1] = akj - 1j * bkj                                          #zero-padded spectrum

    return np.real( np.fft.ifft( C, axis=1 ) * M )                       #(nbds, 2N)

#####################################################################


