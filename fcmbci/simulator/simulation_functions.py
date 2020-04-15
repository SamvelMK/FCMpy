import numpy as np

def sig(x, l):
    """ Sigmoidal transfer function.
        
        Parameters
        ----------
        x : float,
        l : A parameter that determines the steepness of the sigmoid and hyperbolic tangent function at values around 0. 
        
        Return
        ----------
        y : float,
            domain R,
            range [0,1].
        """
    e = np.exp(1)
    res = 1/(1+(e**(-l*x)))
    return res

def bi(x):
    """ Bivalent transfer function.
        
        Parameters
        ----------
        x : float,
        
        Return
        ----------
        y : float,
            domain R,
            range [0;1].
        """

    if x > 0:
        res = 1
    else:
        res = 0

    return res

def tri(x):
    """ Trivalent transfer function.
        
        Parameters
        ----------
        x : float,
        
        Return
        ----------
        y : float,
            domain R,
            range [-1,0,1].
        """

    if x > 0:
        res = 1
    elif x < 0:
        res = -1
    else:
        res = 0

    return res