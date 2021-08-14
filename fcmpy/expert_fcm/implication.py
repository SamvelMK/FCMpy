import numpy as np
from abc import ABC, abstractclassmethod  
from fcmpy.expert_fcm.input_validator import type_check


class FuzzyImplication:
    """
        Fuzzy implication rules.
    """
    @abstractclassmethod
    def implication() -> np.ndarray:
        raise NotImplementedError('implication method is not defined.')


class Mamdani(FuzzyImplication):
    """
        Mamdani minimum implication rule.
    """
    @staticmethod
    @type_check
    def implication(**kwargs) -> np.ndarray: 
        """
            Mamdani min fuzzy implication rule.

            Other Parameters
            ----------------
            **mf_x: numpy.ndarray,
                    membership function of a linguistic term (x)

            **weight: float,
                        the weight at which the membership function x
                        should be activated (i.e., the cut point)
            
            Return
            -------
            y: numpy.ndarray
                the 'activated' membership function
        """
        mf_x = kwargs['mf_x']
        weight = kwargs['weight']

        return np.fmin(weight, mf_x)


class Larsen(FuzzyImplication):
    """
        Larsen's product implication rule.
    """
    @staticmethod
    @type_check
    def implication(**kwargs) -> np.ndarray:
        """
            Larsen's product fuzzy implication rule.

            Other Parameters
            ----------------
            **mf_x: numpy.ndarray
                    membership function of a linguistic term (x)

            **weight: float,
                        the weight at which the membership function x should be activated (i.e., rescaled)
            
            Return
            -------
            y: numpy.ndarray
                the activated (rescaled) membership function
        """
        mf_x = kwargs['mf_x']
        weight = kwargs['weight']

        return  np.dot(mf_x, weight)