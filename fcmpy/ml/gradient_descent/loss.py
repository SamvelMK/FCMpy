from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    """
        Interface for computing the loss.
    """
    @abstractmethod
    def compute(**kwargs):
        raise NotImplementedError('compute method is not defined.')


class MSE(Loss):
    """
        Class for computing the MSE
    """
    @staticmethod
    def compute(**kwargs):
        """
            Parameters
            ----------
            observed: np.array
                        the observed data
            
            predicted: np.array
                        the predicted data
        """        
        obs = kwargs['observed']
        pred = kwargs['predicted']
        n = kwargs['n']

        assert len(obs) == len(pred)
        assert type(n)==int
        
        return np.mean(np.mean((obs-pred)**2, axis=1))