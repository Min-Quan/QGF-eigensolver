import qutip as qp

def build_Ising_model(N, J, g, shift_E):
    '''
    This function construct the transverse field Ising model,
    which is presented as Eq. 6 in the article. It returns a Qobj.
    
    Parameters
    ----------
    N (int): number of qubits.
    J (number): interation strength of nearby sites.
    g (number): strength of the external transverse field.
    shift_E (number): shift-energy applied.
    ----------
    '''
    sx_list = [] # i-th element is sigma x performing on i-th qubit
    sz_list = [] # i-th element is sigma z performing on i-th qubit
    
    for i in range(N):
        ide = [qp.qeye(2)]*N # a list of identity operator
        ide[i] = qp.sigmax() # replace the i-th element by sigma x
        sx_list.append(qp.tensor(ide))
        
        ide = [qp.qeye(2)]*N
        ide[i] = qp.sigmaz()
        sz_list.append(qp.tensor(ide))
        
    Hzz = 0
    for i in range(N - 1):
        Hzz = Hzz + sz_list[i]*sz_list[i + 1]
    # Bound condition $\sigma^{z}_{N} = \sigma^{z}_{0}$
    Hzz = Hzz + sz_list[N - 1]*sz_list[0]
    
    Hx = 0
    for i in range(N):
        Hx = Hx + sx_list[i]
        
    H = -J*Hzz + g*Hx + shift_E
    return H


def Ising_model_decomposition(N, J, g, shift_E):
    '''
    This function gives the local operator of transverse
    field Ising model, which is presented as Eq. 6 in 
    the article. It returns a list of Qobj.
    
    Parameters
    ----------
    N (int): number of qubits.
    J (number): interation strength of nearby sites.
    g (number): strength of the external transverse field.
    shift_E (number): shift-energy applied.
    ----------
    '''
    sx_list = [] # i-th element is sigma x performing on i-th qubit
    sz_list = [] # i-th element is sigma z performing on i-th qubit
    
    for i in range(N):
        ide = [qp.qeye(2)]*N # a list of identity operator
        ide[i] = qp.sigmax() # replace the i-th element by sigma x
        sx_list.append(qp.tensor(ide))
        
        ide = [qp.qeye(2)]*N
        ide[i] = qp.sigmaz()
        sz_list.append(qp.tensor(ide))
        
    h_list = []
    for i in range(N - 1):
        h_list.append(-J*sz_list[i]*sz_list[i + 1])
    # Bound condition $\sigma^{z}_{N} = \sigma^{z}_{0}$
    h_list.append(-J*sz_list[N - 1]*sz_list[0])
    
    for i in range(N):
        h_list.append(g*sx_list[i])
        
    h_list.append(shift_E*qp.tensor([qp.qeye(2)]*N))
    return h_list