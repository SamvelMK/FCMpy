###########################################################################
##            Classes for normalization of matrix error scores           ##
###########################################################################
from abc import ABC
from abc import abstractmethod
from fcmpy.expert_fcm.input_validator import type_check


class Normalization(ABC):
    """
        Interface for normalization classes for RCGA.
    """
    @abstractmethod
    def normalize(**kwargs):
        raise NotImplementedError('normalize method is not defined.')


class NT(Normalization):
    """
        Normalization class for Taxicab and Euclidean normalizations.
    """
    @staticmethod
    @type_check
    def normalize(**kwargs):
        """
            Normalization method for Taxicab and Euclidean normalizations (p = 1, and p e {1,2}).

            Parameters
            ----------
            n: int
                number of Concepts
            
            x: float
                matrix error
            
            t: int
                length of the data (i.e., time points of the longitudinal data)
            
            Return
            ------
            y: float
                normalized matrix error
        """
        x = kwargs['x']
        n = kwargs['n']
        t = kwargs['t']

        return x/(n*(t-1))


class T(Normalization):
    """
        Normalization class for maximum normalization.
    """
    def normalize(**kwargs):
        """
            Normalization method for maximum normalization (p=inf).

            Parameters
            ----------
            x: float
                matrix error
                
            t: int
                length of the data (i.e., time points of the longitudinal data)
            
            Return
            ------
            y: float
                normalized matrix error
        """
        x = kwargs['x']
        t = kwargs['t']
        
        return x/(t-1)