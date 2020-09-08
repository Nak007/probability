'''
Available methods are the followings:
[1] markov_chains
[2] mean_first_passage_time
[3] mean_recurrence_time
[4] absorbing_chain
[5] is_reversible

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 08-09-2020
'''
from warnings import warn
import numbers, numpy as np

def markov_chains(p, u=None, n_steps=float('Inf'), decimal=2) :

    '''
    Let (P) be the transition matrix, and let (u) be the 
    probability vector which represents the starting distribution. 
    Then the probability that the chain is in state s(i) after (n) 
    steps is the ith entry in the vector
    
                        u(n) = u(0)P^n
    
    Moreover, as "n" approaches infinity, P approaches a limiting 
    matrix "Π" with all rows the same vector "π" e.g. w = 
    [π1, π2, ... πk] where k is number of transitions. The vector 
    w is a strictly positive probability vector. For P to possess 
    this property, it must be irreducible (recurrent) and regular,
    which is called ergodic.
    
    .. versionadded:: 08-09-2020
    
    Parameters
    ---------- 
    p : array of float, of shape (n_transitons, n_transitons)
    \t Transition probabilities matrix from (i) to (j), 
    \t where (i) and (j) are row and column, respectively

    u : array of float, of shape (n_transitons,)
    \t Probabtility distribution of initial state or Initial 
    \t probability vector. If None, equal probability is 
    \t assigned to all vector.

    n_steps : float, optional, (default:Inf)
    \t Number of steps. If Inf is assigned, the steady state 
    \t is determined, meaning the probability of all the 
    \t states will approach their limiting value (no change) as 
    \t time goes to infinity.
    
    decimal : int, optional, (default:2)
    \t Decimal places

    Returns
    -------
    dictionary of
    - Transition probabilities matrix (p)
    - Fixed probability vector (π)
    
    References
    ----------
    Introduction to Probability by Joseph K. Blitzstein, 
    Jessica Hwang, Chapter 11

    Example
    -------
    >>> p = np.array([[.9,.1],[.7,.3]])
    >>> u = np.array([.2,.8]).reshape(1,-1)
    >>> markov_chains(p, u, n_state=3)
    Output:
    {'p': array([[0.876, 0.124], [0.868, 0.132]]),  
     'u': array([[0.8696, 0.1304]])}
    '''
    n_steps = max(1.0, n_steps)
    if u is None: u0 = np.full((1,len(p)),1/len(p))
    else: u0 = np.array(u).reshape(1,-1)
    v, u = np.linalg.eig(p)
    if n_steps==float('Inf'): 
        v[np.round(v,0)!=1] = 0
        v = np.identity(len(v))*v
    else: v = (np.identity(len(v))*v)**(n_steps)
    p = np.round(u.dot(v).dot(np.linalg.inv(u)).real,decimal)
    return dict(p=p,u=np.dot(u0,p))

def mean_first_passage_time(p, n_state=None, decimal=2):
    
    '''
    ** Mean First Passage Time **
    
    If an ergodic Markov chain is started in state s(i), the 
    expected number of steps to reach state s(j) for the
    first time is called the mean first passage time from 
    s(i) to s(j). It is defined as
    
            m(i,j) = 1 + ∑( p(i,k) * m(k,j) | k ≠ j )
    
    Alternatively, above equation can be expressed as:
            
            E[min{n>=0 such that s(n)=s(j)} | s(0)=s(i)]
    
    .. versionadded:: 08-09-2020
    
    Parameters
    ----------
    p : array of float, of shape (n_transitons, n_transitons)
    \t Transition probabilities matrix from (i) to (j), 
    \t where (i) and (j) are row and column, respectively
    
    n_state : float, optional, (default:1)
    \t It is a state expected to reach for the first time. 
    \t This ergodic chain shall be made into an absorbing 
    \t chain by making "n_state" an absorbing state. If None,
    \t the mean first passage matrix is calculated, where all 
    \t the diagonal entries are 0. 
    
    decimal : int, optional, (default:2)
    \t Decimal places
    
    Returns
    -------
    If n_state is not None, then function reutrns dictionary 
    of expected number of steps, denoted by u(i,j) e.g. u(1,3) 
    meaning expected time from state 1 to state 3, otherwise 
    returns mean first passage matrix. Also, an array of all
    state s(i).
    
    References
    ----------
    Introduction to Probability by Joseph K. Blitzstein, 
    Jessica Hwang, Chapter 11
    
    Example
    -------                       
    >>> p = np.array([[0.  , 1.  , 0.  , 0.  , 0.  ],
                      [0.25, 0.  , 0.75, 0.  , 0.  ],
                      [0.  , 0.5 , 0.  , 0.5 , 0.  ],
                      [0.  , 0.  , 0.75, 0.  , 0.25],
                      [0.  , 0.  , 0.  , 1.  , 0.  ]])
    >>> mean_first_passage_time(p,5)
    Output:
    ({'u(1,5)': 21.33, 'u(2,5)': 20.33, 
      'u(3,5)': 18.67, 'u(4,5)': 15.0},
     array([1, 2, 3, 4]))
    >>> mean_first_passage_time(p)
    Output:
    array([[ 0.  ,  1.  ,  2.67,  6.33, 21.33],
           [15.  ,  0.  ,  1.67,  5.33, 20.33],
           [18.67,  3.67,  0.  ,  3.67, 18.67],
           [20.33,  5.33,  1.67,  0.  , 15.  ],
           [21.33,  6.33,  2.67,  1.  ,  0.  ]])
    '''
    # Nc = ((I - Q)^-1)c
    if n_state != None:
        p = p-np.identity(len(p))
        index = np.arange(len(p))
        index = index[index!=max(n_state-1,0)]
        p = p[index,:][:,index]

        # Eliminate rows with all zeros (absorbing)  
        # This creates singular matric (non-inversible)
        r = ((p==0).sum(axis=1)==len(p))
        z = np.arange(len(p))[~(r)]
        p = p[z,:][:,z]; index = index[z]

        if len(p)>0:
            x = np.full((len(p),1),-1)
            m = np.round(np.dot(np.linalg.inv(p),x),decimal)
            return dict(('u({},{})'.format(u,n_state),float(m[n]))
                        for n,u in enumerate(index+1)), index+1
        else: return None
    else:
        m = np.full(p.shape,0.0)
        for j in np.arange(1,len(p)+1):
            u, index = mean_first_passage_time(p,j)
            for (i,v) in zip(index,u.values()):
                m[i-1,j-1] = v
        return m
 
def mean_recurrence_time(p, decimal=2):
    
    '''
    ** Mean Recurrence Time **
    
    Assume that we start in state s(i); consider the length of time
    before we return to s(i) for the first time. It is clear that 
    we must return, since we either stay at s(i) the first step or 
    go to some other state s(j), and from any other state s(j), we 
    will eventually reach s(i) because the chain is ergodic. It is
    definded as:

                   m(i) = 1 + ∑( p(i,k) * m(k,i) | k ∈ S)
          
    Let us now define two matrices M and D. The ijth entry m(i,j) of 
    M is the mean first passage time to go from s(i) to s(j) if i ̸= j; 
    the diagonal entries are 0. The matrix M is called the mean first 
    passage matrix. The matrix D is the matrix with all entries 0 
    except the diagonal entries d(i,i) = ri. The matrix D is called 
    the mean recurrence matrix. Let C be an r × r matrix with all 
    entries 1. The matrix equation is:
    
                            M = PM - C - D
                            
    Thus, we obtain  
    
                           D = C - (I - P)M
    
    Alternatively, r(i) can be computed from 1/w(i), where w(i) is
    a fixed probability of i
    
    .. versionadded:: 08-09-2020
    
    Parameters
    ----------
    p : array of float, of shape (n_transitons, n_transitons)
    \t Transition probabilities matrix from (i) to (j), 
    \t where (i) and (j) are row and column, respectively
    
    decimal : int, optional, (default:2)
    \t Decimal places 
    
    Returns
    -------
    dictionary of 
    - Fixed probability vector w(i)
    - Mean Recurrence Time r(i)
    
    References
    ----------
    Introduction to Probability by Joseph K. Blitzstein, 
    Jessica Hwang, Chapter 11
    
    Example
    -------
    periodic matrix
    >>> p = np.array([[  0,1/2,  0,  0,  0,1/2,  0,  0,  0],
                      [1/3,  0,1/3,  0,1/3,  0,  0,  0,  0],
                      [  0,1/2,  0,1/2,  0,  0,  0,  0,  0],
                      [  0,  0,1/3,  0,1/3,  0,  0,  0,1/3],
                      [  0,1/4,  0,1/4,  0,1/4,  0,1/4,  0],
                      [1/3,  0,  0,  0,1/3,  0,1/3,  0,  0],
                      [  0,  0,  0,  0,  0,1/2,  0,1/2,  0],
                      [  0,  0,  0,  0,1/3,  0,1/3,  0,1/3],
                      [  0,  0,  0,1/2,  0,  0,  0,1/2,  0]])
    >>> mean_recurrence_time(p)
    Output:
    ({'w(1)': 0.08, 'w(2)': 0.12, 'w(3)': 0.08, 'w(4)': 0.12,
      'w(5)': 0.17, 'w(6)': 0.12, 'w(7)': 0.08, 'w(8)': 0.12,
      'w(9)': 0.08},
     {'r(1)': 12.0, 'r(2)': 8.0, 'r(3)': 12.0, 'r(4)': 8.0,
      'r(5)': 6.0, 'r(6)': 8.0, 'r(7)': 12.0, 'r(8)': 8.0,
      'r(9)': 12.0})
    '''
    m = mean_first_passage_time(p)
    c = np.full(m.shape,1)
    i = np.identity(len(m))
    d = (c-(i-p).dot(m))[i.astype(bool)]
    r = dict(('r({})'.format(n),np.round(r,2)) for n,r in enumerate(d,1))
    w = dict(('w({})'.format(n),np.round(w,2)) for n,w in enumerate(1/d,1))
    return w,r

def absorbing_chain(p, decimal=2):
    
    '''
    ** Cononical Matrix **
    
    Renumber the states so that the transient states come 
    first. If there are r absorbing states and t transient 
    states, the transition matrix will have the following 
    canonical form
    
                            ---------
                            | Q | R |
                        P = ---------
                            | 0 | I |
                            ---------
    
    where Here I is an r-by-r indentity matrix, 0 is an 
    r-by-t zero matrix, R is a nonzero t-by-r matrix, and 
    Q is an t-by-t matrix. The first t states are transient 
    and the last r states are absorbing
    
    Parameters
    ----------
    p : array of float, of shape (n_trans, n_trans)
    \t Transition probabilities matrix from (i) to (j), 
    \t where (i) and (j) are row and column, respectively
    
    decimal : int, optional, (default:2)
    \t Decimal places
    
    .. versionadded:: 08-09-2020
    
    Returns
    -------
    dictionary of
    - Transient Matrix (r-by-r)
    - Absorbing Matrix (r-by-t)
    - Cononical Matrix (n-by-n)
    - Fundamental Matrix (r-by-r) - note (a)
    - Time to Absorption (r) - note (b)
    - Absorption Probabilities Matrix (r-by-r) - note (c)
    - index
    
    Notes
    -----
    (a) The matrix N = (I − Q)−1 is called the fundamental 
        matrix for P. The entry n(i)(j) of N gives the 
        expected number of times that the process is in the 
        transient state sj if it is started in the transient 
        state si before being absorbed.
    (b) Given that the chain starts in state si, what is the 
        expected number of steps before the chain is absorbed?
        t = Nc, where c is a column vector, whose entries 
        are 1
    (c) The probability that an absorbing chain will be 
        absorbed in the absorbing state sj if it starts in the 
        transient state si
        
    Example
    -------
    >>> p = np.array([[1. , 0. , 0. , 0. , 0. ],
                      [0. , 0.6, 0.4, 0. , 0. ],
                      [0.8, 0. , 0. , 0.2, 0. ],
                      [0.5, 0.3, 0. , 0. , 0.2],
                      [0. , 0. , 0. , 0. , 1. ]])
    >>> absorbing_chain(p)
    Output:
    {'transient': array([[0.6, 0.4, 0. ],
                         [0. , 0. , 0.2],
                         [0.3, 0. , 0. ]]]), 
     'absorbing': array([[0. , 0. ],
                         [0.8, 0. ],
                         [0.5, 0.2]]), 
     'cononical': array([[0.6, 0.4, 0. , 0. , 0. ],
                         [0. , 0. , 0.2, 0.8, 0. ],
                         [0.3, 0. , 0. , 0.5, 0.2],
                         [0. , 0. , 0. , 1. , 0. ],
                         [0. , 0. , 0. , 0. , 1. ]]), 
     'fundamental': array([[2.66, 1.06, 0.21],
                           [0.16, 1.06, 0.21],
                           [0.8 , 0.32, 1.06]]), 
     'time': array([[3.94],
                    [1.44],
                    [2.18]]), 
     'absorption_p': aarray([[0.96, 0.04],
                             [0.96, 0.04],
                             [0.79, 0.21]]), 
     'index': array([1, 2, 3, 0, 4])}
    '''
    diag = p[np.identity(len(p)).astype(bool)]
    # t = trransient matrix, i = identitry matrix
    t = np.arange(len(p))[diag!=1].ravel()
    i = np.arange(len(p))[diag==1].ravel()
    if len(i)>0:
        index = np.hstack((t,i))
        p = p[index,:][:,index]
        q = p[:len(t),:len(t)] # transient matrix
        r = p[:len(t),-len(i):] # absorbing matrix
        n = np.linalg.inv(np.identity(len(q))-q)
        t = np.dot(n,np.full((len(n),1),1))
        a = np.dot(n,r)
        return dict(transient=np.round(q,decimal), 
                    absorbing=np.round(r,decimal),
                    cononical=np.round(p,decimal), 
                    fundamental=np.round(n,decimal), 
                    time=np.round(t,decimal),
                    absorption_p=np.round(a,decimal), 
                    index=index)
    return 'no absorbing state'

def is_reversible(p, decimal=2):
    
    '''
    ** Reversiblility or Detailed Balance **
    
    A class of Markov processes is said to be reversible if, 
    on every time interval, the distribution of the process 
    is the same when it is run backward as when it is run 
    forward. Reversibility is defined as:
    
                    π(i) * p(i,j) = π(j) * p(j,i) 
    
    where all i,j ∈ S
    
    .. versionadded:: 08-09-2020
    
    Parameters
    ----------
    p : array of float, of shape (n_transitons, n_transitons)
    \t Transition probabilities matrix from (i) to (j), 
    \t where (i) and (j) are row and column, respectively
    
    decimal : int, optional, (default:2)
    \t Decimal places
    
    Returns
    -------
    dictionary of
    - Matrix of π(i) * p(i,j) of all i,j ∈ S
    - Reversibility (bool)
    
    Example
    -------
    Reversible
    >>> p = np.array([[0.5  , 0.25 , 0.25 ],
                      [0.3  , 0.4  , 0.3  ],
                      [0.125, 0.125, 0.75 ]])
    >>> is_reversible(p, decimal=3)
    Output:
    {'p': array([[0.13 , 0.065, 0.065],
                 [0.065, 0.087, 0.065],
                 [0.065, 0.065, 0.392]]), 
     'reversible': True}
     
    Irreversible
    >>> p = np.array([[0. , 0.8, 0.2],
                      [0.2, 0. , 0.8],
                      [0.8, 0.2, 0. ]])
    >>> is_reversible(p)
    Output:
    {'p': array([[0.  , 0.26, 0.07],
                 [0.07, 0.  , 0.26],
                 [0.26, 0.07, 0.  ]]), 
     'reversible': False}
    '''
    u = markov_chains(p, decimal=decimal)['u']
    u = np.identity(len(p)) * u
    k = np.round(p.T.dot(u).T, decimal)
    return dict(p=k,reversible=(k==k.T).sum()==p.size)
