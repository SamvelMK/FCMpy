import numpy as np

class FuzzyInference:

    """
    The class includes fuzzy inference methods for deriving FCM edge weights.

    Methods:
            __init__(self)
            __mamdaniMin(mf_x, weight, **params)
            __mamdaniProduct(mf_x, weight, **params)
            add_fuzzy_inference_func(self, func)
            remove_fuzzy_inference_func(self, func_name)
    """

    def __init__(self):
        self.fuzzy_inference_funcs = {"mamdaniMin" : self.__mamdaniMin, "mamdaniProduct" : self.__mamdaniProduct}

    @staticmethod
    def __mamdaniMin(mf_x, weight):
        """
        Mamdani min inference method.

        Parameters
        ----------
        mf_x: numpy.ndarray,
                membership function of a linguistic term (x)

        weight: float,
                    the weight at which the membership function x should be activated (i.e., the cut point)
        
        Return
        ---------
        y: numpy.ndarray
            the activated membership function
        """

        return np.fmin(weight, mf_x)

    @staticmethod
    def __mamdaniProduct(mf_x, weight):
        """
        mf_x: numpy.ndarray.
                membership function of a linguistic term (x)

        weight: float,
                    the weight at which the membership function x should be activated (i.e., rescaled)
        
        Return
        ---------
        y: numpy.ndarray
            the activated (rescaled) membership function
        """

        return  np.dot(mf_x, weight)
    
    def add_fuzzy_inference_func(self, func):
    
        """
        Add a fuzzy inference function.

        Parameters
        ----------
        func: dict,
                key is the name of the function, value is the associated function.
        """

        self.fuzzy_inference_funcs.update(func)
    
    def remove_fuzzy_inference_func(self, func_name):
        """
        Remove a fuzzy inference function.

        Parameters
        ----------
        func_name: str
                    name of the function to be removed.
        """
        if 'FuzzyInference.__' not in str(self.fuzzy_inference_funcs[func_name]):
            del self.fuzzy_inference_funcs[func_name]
        else:
            raise ValueError('Cannot remove a base function!')
