import numpy as np
from abc import ABC, abstractmethod  
from fcmpy.expert_fcm.input_validator import type_check

class Aggregate(ABC):
    """
        Fuzzy aggregation rule.
    """
    @abstractmethod
    def aggregate() -> np.ndarray:
        raise NotImplementedError('aggregate method is not defined.')


class Fmax(Aggregate):
    """
        Family max aggregation rule.
    """
    @staticmethod
    @type_check
    def aggregate(**kwargs) -> np.ndarray:
        """
            Family max aggregation rule.

            Other Parameters
            ----------
            **x, **y: numpy.ndarray,
                        "activated" membership functions of the linguistic terms
                            that need to be aggregated
            
            Return
            -------
            y: np.ndarray
                an aggregated membership function
        """
        x = kwargs['x']
        y = kwargs['y']

        return np.fmax(x, y)


class AlgSum(Aggregate):
    """
        Family Algebraic Sum.
    """
    @staticmethod
    @type_check
    def aggregate(**kwargs) -> np.ndarray:
        """
            Family Algebraic sum aggregation rule.

            Other Parameters
            ----------
            **x, **y: numpy.ndarray,
                        "activated" membership functions of the linguistic terms that need to be aggregated
            
            Return
            -------
            y: np.ndarray
                an aggregated membership function
        """
        x = kwargs['x']
        y = kwargs['y']

        return x+y - (x*y)


class EinsteinSum(Aggregate):
    """
        Family Einstein Sum.
    """
    @staticmethod
    @type_check
    def aggregate(**kwargs) -> np.ndarray:
        """
            Family Einstein sum aggregation rule.

            Other Parameters
            ----------
            **x, **y: numpy.ndarray,
                        "activated" membership functions of the linguistic terms that need to be aggregated
            
            Return
            -------
            y: float
                an aggregated membership function
        """
        x = kwargs['x']
        y = kwargs['y']

        return (x+y) / (1.0 + x*y)


class HamacherSum(Aggregate):
    """
        Family Hamacher Sum.
    """
    @staticmethod
    @type_check
    def aggregate(**kwargs) -> np.ndarray:
        """
            Family Hamacher sum aggregation rule.

            Other Parameters
            ----------------
            **x, **y: numpy.ndarray,
                        "activated" membership functions of the linguistic terms that need
                        to be aggregated
            
            Return
            -------
            y: np.ndarray
                an aggregated membership function
        """
        x = kwargs['x']
        y = kwargs['y']

        if x * y != 1.0:
            return (x+y - 2.0*x*y) / (1.0 - x*y)
        else:
            return 1.0