from typing import Union
import numpy as np
import pandas as pd
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
        Synchronously (i.e., NHL) update the weight matrix according 
        to the adjusted Hebbian rule.
    """
    @staticmethod
    def update(initial_state: dict, weight_matrix: pd.DataFrame, 
                gamma:Union[float, int]=0.98, eta:float=0.1):
        """
            Update the weight matrix according to the adjusted Hebbian rule.

            Parameters
            ----------
            initial_state: dict
                            initial state vector of the concepts

            weight_matrix: pd.DataFrame
                            N*N weight matrix of the FCM.
            
            gamma: float, int
                    decay coefficient
            
            eta: float,
                    learning rate
        
            Return
            ----------
            y: pd.DataFrame
                updated weight matrix.
        """
        # Convert pd -> ndarray
        # convert dict val -> array
        w = weight_matrix.to_numpy()
        s = np.array(list(initial_state.values()))

        # update the W matrix
        a = gamma*w
        b = eta*s
        c = abs(w)*s
        res = a + abs(np.sign(w))*b*(s-c.T).T

        # convert it back to df
        res = pd.DataFrame(res, columns = weight_matrix.columns, index = weight_matrix.index)
        return res


class AhlWeightUpdate(WeightUpdate):
    """
        Asynchronous (i.e., AHL) Update the weight matrix according to the adjusted Hebbian rule.
    """
    @staticmethod
    def update(initial_state: np.array, weight_matrix: np.ndarray, gamma, eta):
        pass