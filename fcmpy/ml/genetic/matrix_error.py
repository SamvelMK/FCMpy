###########################################################################
##                   Classes for matrix error methods                    ##
###########################################################################
from abc import ABC
from abc import abstractmethod
import numpy as np


class MatrixError(ABC):
    """
        Interface for matrix error classes for RCGA.
    """
    @abstractmethod
    def calculate(**kwargs):
        raise NotImplementedError('calculate method is not defined.')


class StachError(MatrixError):
    """
        Calculate Matrix Error for RCGA based on Stach et al. proposal
    """
    @staticmethod
    def calculate(**kwargs):
        """
            Function for calculating matrix error for RCGA based on Stach et al. proposal

            Parameters
            ----------
            data_simulated: numpy.ndarray
                                data generated based on the candidate solution
            
            data: numpy.ndarray
                    training data
            
            p: int for p={1,2} or np.inf for infinity norm.
                normalization type
            
            Return
            ------
            y: float, int
                Total matrix error
        """
        data_simulated = kwargs['data_simulated']
        data = kwargs['data']
        p = kwargs['p']
        
        return np.linalg.norm(data - data_simulated, p)