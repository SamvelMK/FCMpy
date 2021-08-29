import numpy as np
from abc import ABC
from abc import abstractmethod

class WeightUpdate(ABC):
    """
        Interface for updating the FCM weight matrix
    """
    @abstractmethod
    def update() -> np.ndarray:
        """
            Update the FCM weight matrix
        """
        raise NotImplementedError('Update method is not defined!')


class NhlWeightUpdate(WeightUpdate):
    """
        Synchronous (i.e., NHL) Update the weight matrix according to the adjusted Hebbian rule.
    """
    @staticmethod
    def update(initial_state: np.array, weight_matrix: np.ndarray, gamma=0.98, eta=0.1):
        """
            Update the weight matrix according to the adjusted Hebbian rule.

            Parameters
            ----------
            initial_state: np.array
                            initial state vector of the concepts

            weight_matrix: np.ndarray
                            N*N weight matrix of the FCM.
            
            gamma: float, int
                    decay coefficient
            
            eta: float,
                    learning rate
        
            Return
            ----------
            y: np.ndarray
                updated weight matrix.
        """
        a = gamma*weight_matrix
        b = eta*initial_state
        c = abs(weight_matrix)*initial_state

        res = a + abs(np.sign(weight_matrix))*b*(initial_state-c.T).T
        
        return res


class AhlWeightUpdate(WeightUpdate):
    """
        Asynchronous (i.e., AHL) Update the weight matrix according to the adjusted Hebbian rule.
    """
    @staticmethod
    def update(initial_state: np.array, weight_matrix: np.ndarray, gamma, eta):
        pass