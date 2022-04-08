import qutip as qp
import numpy as np
import scipy

class GFE_circuit():
    #----------------------------------------
    '''
    Set parameters of Gaussian filter eigensolver circuit.
    
    Attributes
    ----------
    H (Qobj): Hamiltonian to be solved.
    N (int): number of qubits.
    ini_state (Qobj): initial state.
    dy (number): slice size of Fourier approximation.
    My (int): cutoff number of terms of unitary operators.
    sig_squ (number): variance of Gaussian filter.
    mu (number): expectation value (minus shift-energy) of
    Gaussian filter.
    h_list (list of Qobj): local operator of the Hamiltonian,
    which is used for Trotter composition.
    px, pz (number): probablity of bit-flip and phase-flip noise.
    TN (int): Trotter number.
    ----------
    '''
    def set_Hamiltonian(self, H):
        self.H = H
        self.N = len(H.dims[0])
    def set_initial_state(self, ini_state):
        self.ini_state = ini_state
    def set_discrete_parameters(self, dy, My):
        self.dy = dy
        self.My = My
    def set_Gaussian_parameters(self, sig_squ, mu):
        self.sig_squ = sig_squ
        self.mu = mu
    def set_local_operator_list(self, h_list):
        # local operator list of the Hamiltonian to be solved
        self.h_list = h_list
    def set_noise_parameters(self, px = 0, pz = 0):
        self.px = px # bit flip probablity
        self.pz = pz # phase flip probablity
    def set_Trottrt_number(self, TN):
        self.TN = TN # trotter number
        
    def gen_rand_initial_state(self, ope_list, no_of_layers):
        '''
        generate a random initial state by rotating through the
        given axis by random angles.
        
        Parameters
        ----------
        ope_list (list of Qobj): the axis of rotations.
        no_of_layers (int): number of layers.
        ----------
        ''' 
        ini_state = qp.tensor([qp.basis(2)]*self.N)
        for i in range(no_of_layers):
            para = np.random.rand(len(ope_list))*2*np.pi
            for j in range(len(ope_list)):
                ini_state = (-1j*para[j]*ope_list[j]).expm()*ini_state
        self.ini_state = ini_state.unit()
    #----------------------------------------
    
    
    def demo_approximate_GF(self, lambda_list):
        '''
        Compute the approximate Gaussian function of the
        given eigenvalue list. It shows the effect of
        the Gaussian filter eigensolver for a range of 
        eigenvalues.
        
        Parameters
        ----------
        lamda_list (list): the eigenvalue list.
        ----------
        '''
        def approximate_GF(dy, My, sig_squ, mu, lamb):
            # Compute the value of Gaussian function for lambda
            n_fact = np.sqrt(sig_squ/np.pi)/2 # normalization factor
            val = 0
            for y in range(-My, My+1):
                val += np.exp(-sig_squ*(y*dy)**2/4 + 1j*mu*y*dy - 1j*lamb*y*dy)*dy
            val = n_fact*val
            return val
        
        approximate_GF_list = []
        for lamb in lambda_list:
            approximate_GF_value = approximate_GF(self.dy, self.My, self.sig_squ, self.mu, lamb)
            approximate_GF_list.append(approximate_GF_value)
        return approximate_GF_list
    
    
    def generate_coefficient(self):
        '''
        Compute the coefficient list {by} in Eq. 4.
        '''
        My = self.My
        dy = self.dy
        sig_squ = self.sig_squ
        mu = self.mu
        by_list = []
        n_fact = np.sqrt(sig_squ/np.pi)/2
        for y in range(-My, My+1):
            by = np.exp(-sig_squ*(y*dy)**2/4 + 1j*mu*y*dy)
            by_list.append(by*n_fact)
        return by_list
    
    
    def directly_compute_overlaps(self):
        '''
        Directly solve each overlap term of Eq. 4, whose weighted
        summation is estimated ground state energy.
        There are 4*My*My terms in Eq.4. But only 4*My terms are unique.
        The evolution time is in a range of [-2*My*dy, 2*My*dy].
        '''
        My = self.My
        dy = self.dy
        ty_list = np.arange(-2*My*dy, (2*My+1)*dy, dy)
        num_list = [] # store the numertor term
        den_list = [] # store the denominator term
        ini_state = self.ini_state
        H = self.H
        for ty in ty_list:
            evo_state = (-1j*ty*H).expm()*ini_state
            num = (ini_state.dag()*H*evo_state).tr()
            den = (ini_state.dag()*evo_state).tr()
            num_list.append(num)
            den_list.append(den)
        return num_list, den_list
    
    def compute_E(self, by_list, num_list, den_list):
        '''
        Compute the estimated energy of Eq. 4 by weighted summation
        of overlaps.
        
        Parameters
        ----------
        by_list (list): weights of overlaps.
        num_list (list): list of numerator terms.
        den_list (list): list of denominator terms.
        ----------
        '''
        My = self.My
        by_list = np.array(by_list)
        num_mat = 1j*np.zeros([2*My+1, 2*My+1])
        den_mat = 1j*np.zeros([2*My+1, 2*My+1])
        # transfer the list to matrix
        for i in range(2*My+1):
            for j in range(2*My+1):
                num_mat[i][j] = num_list[2*My+j-i]
                den_mat[i][j] = den_list[2*My+j-i]
        E_num = np.dot(np.matrix.getH(by_list), np.dot(num_mat, by_list))
        E_den = np.dot(np.matrix.getH(by_list), np.dot(den_mat, by_list))
        E = E_num/E_den
        return E
    
    
    def directly_compute_additional_overlaps(self, pre_overlaps, new_My, new_dy):
        '''
        Compute the overlaps after changing the discrete parameters.
        It only solve the overlap whose corresponding ty is not in previous
        ty list; otherwise, it reads from the previous overlap list.
        
        Parameters
        ----------
        pre_overlaps (list): previous overlap that contains numerator list and denominator list.
        new_My (int): new My.
        new_dy (number): new dy.
        ----------
        '''
        
        My = self.My # previous My
        dy = self.dy # previous dy
        pre_ty_list = np.arange(-2*My*dy, (2*My+1)*dy, dy) # previous ty list
        pre_num_list, pre_den_list = pre_overlaps # previous overlaps
        # create a hash map to connect the previous ty and corresponding overlap
        ty_num_map = {}
        ty_den_map = {}
        for i in range(len(pre_ty_list)):
            ty_num_map[pre_ty_list[i]] = pre_num_list[i]
            ty_den_map[pre_ty_list[i]] = pre_den_list[i]
        
        # new ty list and update the attributes
        ty_list = np.arange(-2*new_My*new_dy, (2*new_My+1)*new_dy, new_dy)
        self.My = new_My
        self.dy = new_dy
        
        num_list = [] # store the total numertor term
        den_list = [] # store the total denominator term
        ini_state = self.ini_state
        H = self.H
        
        for ty in ty_list:
            if ty in pre_ty_list:
                # If ty is in previous ty list, it means we have computed
                # it before. We directly read it from the hash map.
                num_list.append(ty_num_map[ty])
                den_list.append(ty_den_map[ty])
            else:
                # Otherwise, we compute it.
                evo_state = (-1j*ty*H).expm()*ini_state
                num = (ini_state.dag()*H*evo_state).tr()
                den = (ini_state.dag()*evo_state).tr()
                num_list.append(num)
                den_list.append(den)
        return num_list, den_list
    
    
    #----------------------------------------
    # Consider Hadmard test and noise        
    def compute_overlap_by_Hadmard_test(self):
        def bit_flip_channel(state, p):
            N = len(state.dims[0]) # number of qubit
            X = qp.tensor([qp.sigmax()]*N) # X-gate for all the qubit
            state = (1 - p)*state + p*X*state*X
            return state
        def phase_flip_channel(state, p):
            N = len(state.dims[0]) # number of qubit
            Z = qp.tensor([qp.sigmaz()]*N) # Z-gate for all the qubit
            state = (1 - p)*state + p*Z*state*Z
            return state
        
        # read hamiltonian information
        h_list = self.h_list
        H = self.H
        ini_state = qp.ket2dm(self.ini_state)
        
        # read noise parameters
        px = self.px
        pz = self.pz
        
        num_list = [] # store the numertor term
        den_list = [] # store the denominator term
        
        dt = self.dy/self.TN # rotation angle for each rotation
        ide = qp.tensor([qp.qeye(2)]*self.N) # identity matrix
        zero = qp.ket2dm(qp.basis(2, 0)) # zero density matrix
        one = qp.ket2dm(qp.basis(2, 1)) # one density matrix
        H_ancilla = qp.tensor(qp.snot(), ide) # H-gate on ancilalry qubit
        C_H = qp.tensor(zero, ide) + qp.tensor(one, H) # controlled H, 0 for identity and 1 for H
        ope_pt_list = [] # controlled operator list of positive evolution time
        ope_nt_list = [] # controlled operator list of negative evolution time
        for i in range(len(h_list)-1):
            ope_pt = (-1j*dt*h_list[i]).expm()
            ope_nt = (1j*dt*h_list[i]).expm()
            ope_pt_list.append(qp.tensor(zero, ide) + qp.tensor(one, ope_pt))
            ope_nt_list.append(qp.tensor(zero, ide) + qp.tensor(one, ope_nt))
        # shift_energy unitary operator
        ope_pt_shift = qp.tensor(zero, ide) + qp.tensor(one, (-1j*self.dy*h_list[-1]).expm())
        ope_nt_shift = qp.tensor(zero, ide) + qp.tensor(one, (1j*self.dy*h_list[-1]).expm())
        
        # Negative time
        ancilla_real = qp.ket2dm(qp.snot()*qp.basis(2)) # ancillary qubit for real part
        ancilla_imag = qp.ket2dm(qp.sigmaz()*qp.s_gate()*qp.snot()*qp.basis(2)) # ancillary qubit for imaginary part
        state_real = qp.tensor(ancilla_real, ini_state) # initial state for real part
        state_imag = qp.tensor(ancilla_imag, ini_state) # initial state for imaginary part
        for i in range(2*self.My):
            print(str(round((i+1)/4/self.My*100, 2)) + '% completed.')
            # shift energy rotation
            state_real = ope_nt_shift*state_real*ope_nt_shift.dag()
            state_imag = ope_nt_shift*state_imag*ope_nt_shift.dag()
            # Trotter evolution
            for j in range(self.TN):
                for k in range(len(ope_nt_list)):
                    state_real = ope_nt_list[k]*state_real*ope_nt_list[k].dag()
                    state_imag = ope_nt_list[k]*state_imag*ope_nt_list[k].dag()
                    if px != 0:
                        state_real = bit_flip_channel(state_real, px)
                        state_imag = bit_flip_channel(state_imag, px)
                    if pz != 0:
                        state_real = phase_flip_channel(state_real, pz)
                        state_imag = phase_flip_channel(state_imag, pz)
            state_measure_real = H_ancilla*state_real*H_ancilla
            state_measure_real = state_measure_real.ptrace([0])
            state_measure_real = qp.sigmaz()*state_measure_real
            state_measure_imag = H_ancilla*state_imag*H_ancilla
            state_measure_imag = state_measure_imag.ptrace([0])
            state_measure_imag = qp.sigmaz()*state_measure_imag
            den_list.append(state_measure_real.tr() + 1j*state_measure_imag.tr())
            state_H_measure_real = H_ancilla*C_H*state_real*C_H.dag()*H_ancilla
            state_H_measure_real = state_H_measure_real.ptrace([0])
            state_H_measure_real = qp.sigmaz()*state_H_measure_real
            state_H_measure_imag = H_ancilla*C_H*state_imag*C_H.dag()*H_ancilla
            state_H_measure_imag = state_H_measure_imag.ptrace([0])
            state_H_measure_imag = qp.sigmaz()*state_H_measure_imag
            num_list.append(state_H_measure_real.tr() + 1j*state_H_measure_imag.tr())
            
        # reverse the list; then it start from -2My to -1
        num_list.reverse()
        den_list.reverse()
        
        # at t = 0
        num_list.append(qp.expect(H, ini_state))
        den_list.append(1)
        
        # Positive time
        ancilla_real = qp.ket2dm(qp.snot()*qp.basis(2)) # ancillary qubit for real part
        ancilla_imag = qp.ket2dm(qp.sigmaz()*qp.s_gate()*qp.snot()*qp.basis(2)) # ancillary qubit for imaginary part
        state_real = qp.tensor(ancilla_real, ini_state) # initial state for real part
        state_imag = qp.tensor(ancilla_imag, ini_state) # initial state for imaginary part
        for i in range(2*self.My):
            print(str(round(50 + (i+1)/4/self.My*100, 2)) + '% completed.')
            # shift energy rotation
            state_real = ope_pt_shift*state_real*ope_pt_shift.dag()
            state_imag = ope_pt_shift*state_imag*ope_pt_shift.dag()
            # Trotter evolution
            for j in range(self.TN):
                for k in range(len(ope_pt_list)):
                    state_real = ope_pt_list[k]*state_real*ope_pt_list[k].dag()
                    state_imag = ope_pt_list[k]*state_imag*ope_pt_list[k].dag()
                    if px != 0:
                        state_real = bit_flip_channel(state_real, px)
                        state_imag = bit_flip_channel(state_imag, px)
                    if pz != 0:
                        state_real = phase_flip_channel(state_real, pz)
                        state_imag = phase_flip_channel(state_imag, pz)
            state_measure_real = H_ancilla*state_real*H_ancilla
            state_measure_real = state_measure_real.ptrace([0])
            state_measure_real = qp.sigmaz()*state_measure_real
            state_measure_imag = H_ancilla*state_imag*H_ancilla
            state_measure_imag = state_measure_imag.ptrace([0])
            state_measure_imag = qp.sigmaz()*state_measure_imag
            den_list.append(state_measure_real.tr() + 1j*state_measure_imag.tr())
            state_H_measure_real = H_ancilla*C_H*state_real*C_H.dag()*H_ancilla
            state_H_measure_real = state_H_measure_real.ptrace([0])
            state_H_measure_real = qp.sigmaz()*state_H_measure_real
            state_H_measure_imag = H_ancilla*C_H*state_imag*C_H.dag()*H_ancilla
            state_H_measure_imag = state_H_measure_imag.ptrace([0])
            state_H_measure_imag = qp.sigmaz()*state_H_measure_imag
            num_list.append(state_H_measure_real.tr() + 1j*state_H_measure_imag.tr())
            
        return num_list, den_list
    
    
    def solve_ground_state_with_qumode(self, s, cut, max_cut):
        '''
        Solve the approximate ground state by Gaussian filter assisted with qumode.
        It returns a not normalizable approximate ground state since this approach
        contains a post-selection process.
        
        Parameters
        ----------
        s (number): squeezing factor of the ancillary qumode.
        cut (int): truncation of the Fock state while preparing the resource state.
        max_cut (int): truncation of the Fock space.
        ----------
        '''
        def generate_squeezed_state(s, cut):
            # Generate a finite squeezed state
            def myHermite(n, p):
                factor = np.sqrt(2**n*np.math.factorial(n)*np.sqrt(np.pi))
                return (1j)**n*scipy.special.hermite(n)(p)*np.exp(-p**2/2)/factor

            def rstate(s, p):
                return np.sqrt(1/(np.sqrt(np.pi)*s))*np.exp(-p**2/(2*s**2))

            def coeff_rstate(s, n):
                def integrand_real(p, s, n):
                    return np.real(myHermite(n, p))*rstate(s, p)
                def integrand_imag(p, s, n):
                    return np.imag(myHermite(n, p))*rstate(s, p)
                cn_real = scipy.integrate.quad(integrand_real, -np.infty, np.infty, args = (s, n))
                cn_imag = scipy.integrate.quad(integrand_imag, -np.infty, np.infty, args = (s, n))
                cn = cn_real[0] + 1j*cn_imag[0]
                return cn

            def r_state_fock(s, cut):
                return [coeff_rstate(s, n) for n in range(cut)] 

            resource_state = np.array(r_state_fock(s, cut+1))
            return resource_state
        
        # resource state
        r_array = generate_squeezed_state(s, cut)
        r_state = np.r_[r_array, [0]*(max_cut - cut - 1)]
        r_state = qp.Qobj(r_state)
        
        eiHP = (-1j*qp.tensor(self.H, qp.operators.momentum(max_cut))).expm() # e^{-iHP}
        ide = qp.tensor([qp.qeye(2)]*self.N) # identity matrix
        proj = qp.tensor(ide, qp.basis(max_cut)*r_state.dag()) # projection state
        
        state = eiHP*qp.tensor(self.ini_state, r_state) # entangling the qubit-qumode state
        state = proj*state # projecting the qumode state
        state = qp.ptrace(state, range(self.N)) # partial trace the ancillary qubits
        return state