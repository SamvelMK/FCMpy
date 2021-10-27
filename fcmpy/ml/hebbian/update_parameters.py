from fcmpy.expert_fcm.input_validator import type_check
import numpy as np
from abc import ABC
from abc import abstractmethod

class UpdateParameters(ABC):
    @abstractmethod
    def update(**kwargs) -> float:
        raise NotImplementedError('update method is not defined.')


class Eta(UpdateParameters):
    """
        Update the learning parameter Eta.
    """
    @staticmethod
    @type_check
    def update(**kwargs) -> float:
        """
            Update learning parameter eta.
            
            Parameters
            ----------
            cycle: int,
                    learning cycle
            b1: float
                    default -> 0.02
            l1: float
                    default -> 0.1
        """
        b1 = kwargs['b1']
        l1 = kwargs['l1']
        cycle = kwargs['cycle']
        
        return b1 * np.exp(-l1*cycle)


class Gamma(UpdateParameters):
    """
        Update the decay parameter gamma.
    """
    @staticmethod
    @type_check
    def update(**kwargs):
        """
            Update learning parameter eta.
            
            Parameters
            ----------
            cycle: int,
                    learning cycle
            b2: float
                    default -> 0.04
            bl: float
                    default -> 1
        """
        b2 = kwargs['b2']
        l2 = kwargs['l2']
        cycle = kwargs['cycle']

        return b2*np.exp(-l2*cycle)