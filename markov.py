'''
Methods
-------
(1) markov_chains
(2) mean_first_passage_time
(3) mean_recurrence_time
(4) absorbing_chain
'''
import numpy as np

def markov_chains(p, u=None, n_steps=float('Inf'), decimal=2) :

    '''
    Let (P) be the transition matrix, and let (u) be the 
    probability vector which represents the starting distribution. 
    Then the probability that the chain is in state s(i) after (n) 
    steps is the ith entry in the vector
    
                        u(n) = u(0)P^n
    
    Moreover, as "n" approaches infinity, P approaches a limiting 
    matrix "W" with all rows the same vector "w" e.g. w = 
    [w1, w2, ... wk] where k is number of transitions. The vector 
    w is a strictly positive probability vector. For P to possess 
    this property, it must be irreducible (recurrent) and regular,
    which is called ergodic.
    
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
    - Fixed probability vector (u)
    
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
