import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
from abc import ABC, abstractmethod


class Transfer(ABC):
    
    """
    Class of FCM transfer methods.
    """
    
    @abstractmethod
    def transfer():
         raise NotImplementedError('Transfer method is not defined!')

class Sigmoid(Transfer):
    """
    Sigmoid transfer method
    """

    @staticmethod
    def transfer(**kwargs):

        """ 
        Sigmoidal transfer function.
            
        Parameters
        ----------
        x : numpy.array,
                the results of the FCM update function.
        l : int/float
                A parameter that determines the steepness of the sigmoid function at values around 0. 
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [0,1].
        """
        
        x = kwargs['x']
        l = kwargs['params']['l']
        e = np.exp(1)
        res = 1/(1+(e**(-l*x)))
        return res

class Bivalent(Transfer):

    """
    Bivalent transfer method
    """

    @staticmethod
    def transfer(**kwargs):

        """ 
        Bivalent transfer function.
            
        Parameters
        ----------
        x : numpy.array,
                the results of the FCM update function.
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [0;1].
        """

        x = kwargs['x']

        res = np.array([1 if i > 0 else 0 for i in n])

        return res

class Trivalent(Transfer):

    """
    Trivalent transfer function.
    """

    @staticmethod
    def transfer(**kwargs):

        """ 
        Trivalent transfer function.
            
        Parameters
        ----------
        x : numpy.array,
                the results of the FCM update function.
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [-1,0,1].
        """

        x = kwargs['x']

        res = np.array([1 if i > 0 else -1 if i < 0 else 0 for i in x])

        return res

class HyperbolicTangent(Transfer):

    """
    Hyperbolic tangent transfer function.
    """

    @staticmethod
    def transfer(**kwargs):

        """ 
        Hyperbolic tangent transfer function.

        Parameters
        ----------
        x : numpy.array
                the results of the FCM update function.
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [-1,1].
        """

        x = kwargs['x']

        return np.tanh(x)    