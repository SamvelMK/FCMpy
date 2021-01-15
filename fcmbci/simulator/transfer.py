import numpy as np

class Transfer:
        """
        The class includes transfer methods for FCM update function.

        Methods:
                __init__(self)
                __sig(x, **params)
                __tri(x, **params)
                __tanh(x, **params)
                add_transfer_func(self, func)
                remove_transfer_func(self, func_name)
        """
    def __init__(self):
        self.transfer_funcs = {"sigmoid" : self.__sig, "bivalent" : self.__bi, "trivalent" : self.__tri, "tanh" : self.__tanh}

    @staticmethod
    def __sig(x, **params):
        
        """ 
        Sigmoidal transfer function.
            
        Parameters
        ----------
        x : numpy.array,
                the results of the FCM update function.
        l : int/float
                A parameter that determines the steepness of the sigmoid function at values around 0. 
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [0,1].
        """
        
        l = params['l']
        e = np.exp(1)
        res = 1/(1+(e**(-l*x)))
        return res

    @staticmethod
    def __bi(x, **params):
        """ 
        Bivalent transfer function.
            
        Parameters
        ----------
        x : numpy.array,
                the results of the FCM update function.
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [0;1].
        """
        res = np.array([1 if i > 0 else 0 for i in n])

        return res

    @staticmethod
    def __tri(x, **params):
        
        """ 
        Trivalent transfer function.
            
        Parameters
        ----------
        x : numpy.array,
                the results of the FCM update function.
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [-1,0,1].
        """

        res = np.array([1 if i > 0 else -1 if i < 0 else 0 for i in x])

        return res
    
    @staticmethod
    def __tanh(x, **params):
        """ 
        Hyperbolic tangent transfer function.

        Parameters
        ----------
        x : numpy.array
                the results of the FCM update function.
        
        Return
        ----------
        y : numpy.array,
                domain R,
                range [-1,1].
        """

        return np.tanh(x)
    
    def add_transfer_func(self, func):

        """
        Add a transfer function.

        Parameters
        ----------
        func: dict,
                key is the name of the function, value is the associated function.
        """

        self.transfer_funcs.update(func)
    
    def remove_transfer_func(self, func_name):
        """
        Remove a transfer function.

        Parameters
        ----------
        func_name: str
                    name of the function to be removed.
        """
        if 'Transfer.__' not in str(self.transfer_funcs[func_name]):
            del self.transfer_funcs[func_name]
        else:
            raise ValueError('Cannot remove a base function!')