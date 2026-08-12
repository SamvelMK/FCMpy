import numpy as np
from abc import ABC, abstractmethod


class Transfer(ABC):    
    """
        Class of FCM transfer methods.
    """ 
    @abstractmethod
    def transfer() -> np.array:
         raise NotImplementedError('Transfer method is not defined!')


class Sigmoid(Transfer):
    """
        Sigmoid transfer method
    """
    @staticmethod
    def transfer(**kwargs) -> np.array:
        """ 
            Sigmoid transfer function.
                
            Parameters
            ----------
            x : numpy.array,
                    the results of the FCM update function.
            l : int/float
                    A parameter that determines the steepness of the sigmoid function at values around 0. 
            
            Return
            -------
            y : numpy.array,
                    domain R,
                    range [0,1].
        """
        x = kwargs['x']
        l = kwargs['params']['l']

        return 1 / (1 + np.exp(-l * x))


class Bivalent(Transfer):
    """
        Bivalent transfer method
    """
    @staticmethod
    def transfer(**kwargs) -> np.array:
        """ 
            Bivalent transfer function.
                
            Parameters
            ----------
            x : numpy.array,
                    the results of the FCM update function.
            
            Return
            -------
            y : numpy.array,
                    domain R,
                    range [0;1].
        """
        x = kwargs['x']

        return np.where(x > 0, 1, 0)


class Trivalent(Transfer):
    """
        Trivalent transfer function.
    """
    @staticmethod
    def transfer(**kwargs) -> np.array:
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

        return np.sign(x)


class HyperbolicTangent(Transfer):
    """
        Hyperbolic tangent transfer function.
    """
    @staticmethod
    def transfer(**kwargs) -> np.array:
        """ 
            Hyperbolic tangent transfer function.

            Parameters
            ----------
            x : numpy.array
                    the results of the FCM update function.
            
            Return
            -------
            y : numpy.array,
                    domain R,
                    range [-1,1].
        """
        x = kwargs['x']

        return np.tanh(x)
