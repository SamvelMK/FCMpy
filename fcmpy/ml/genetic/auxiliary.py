###########################################################################
##       Classes for auxiliary methods for the fitness scores            ##
###########################################################################
from abc import ABC
from abc import abstractmethod


class Auxiliary(ABC):
    """
        Interface for the auxiliary classes for the fitness score
    """
    @abstractmethod
    def f(**kwargs):
        raise NotImplementedError('f method is not defined.')


class H(Auxiliary):
    """
        Auxiliary classes for the fitness score
    """
    def f(**kwargs):
        """
            Auxiliary function for the fitness score

            Parameters
            ----------
            a: int
                a coefficient established experimentally
            
            x: float
                normalized matrix error
        
            Return
            ------
            y: float
                fitness score
        """
        a=kwargs['a']
        x=kwargs['x']
        return 1/(a*x+1)