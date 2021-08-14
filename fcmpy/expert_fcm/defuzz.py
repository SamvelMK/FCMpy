from abc import ABC, abstractclassmethod 
import skfuzzy as fuzz
from fcmpy.expert_fcm.input_validator import type_check


class Defuzzification(ABC):
    """
        Defuzzification methods.
    """
    @abstractclassmethod
    def defuzz() -> float:
        raise NotImplementedError('defuzzification method is not defined!')


class Centroid(Defuzzification):
    """
        Centroid difuzzification method (i.e., center of gravity).
    """
    @staticmethod
    @type_check
    def defuzz(**kwargs) -> float:
        """
            Centroid difuzzification method (i.e., center of gravity).
            
            Other Parameters
            ----------
            **x: numpy.ndarray
                universe of discourse 
            
            **mfx: numpy.ndarray,
                        "aggregated" membership functions
            
            Return
            -------
            y: float
                defuzzified value
        """
        method = kwargs['method']
        x = kwargs['x']
        mfx = kwargs['mfx']

        return fuzz.defuzz(x, mfx, method)


class Bisector(Defuzzification):
    """
        Bisector difuzzification method.
    """
    @staticmethod
    @type_check
    def defuzz(**kwargs) -> float:
        """
            Bisector difuzzification method.
            
            Other Parameters
            ----------
            **x: numpy.ndarray
                universe of discourse 
            
            **mfx: numpy.ndarray,
                        "aggregated" membership functions
            
            Return
            -------
            y: float
                defuzzified value
        """
        method = kwargs['method']
        x = kwargs['x']
        mfx = kwargs['mfx']

        return fuzz.defuzz(x, mfx, method)


class MeanOfMax(Defuzzification):
    """
        MeanOfMax difuzzification method.
    """
    @staticmethod
    @type_check
    def defuzz(**kwargs) -> float:
        """
            MeanOfMax difuzzification method.
            
            Other Parameters
            ----------
            **x: numpy.ndarray
                universe of discourse 
            
            **mfx: numpy.ndarray,
                        "aggregated" membership functions
            
            Return
            -------
            y: float
                defuzzified value
        """
        method = kwargs['method']
        x = kwargs['x']
        mfx = kwargs['mfx']

        return fuzz.defuzz(x, mfx, method)


class MinOfMax(Defuzzification):
    """
        MinOfMax difuzzification method.
    """
    @staticmethod
    @type_check
    def defuzz(**kwargs) -> float:
        """
            MinOfMax difuzzification method.
            
            Other Parameters
            ----------
            **x: numpy.ndarray
                universe of discourse 
            
            **mfx: numpy.ndarray,
                        "aggregated" membership functions
            
            Return
            -------
            y: float
                defuzzified value
        """
        method = kwargs['method']
        x = kwargs['x']
        mfx = kwargs['mfx']

        return fuzz.defuzz(x, mfx, method)


class MaxOfMax(Defuzzification):
    """
        MaxOfMax difuzzification method.
    """
    @staticmethod
    @type_check
    def defuzz(**kwargs) -> float:
        """
            MaxOfMax difuzzification method.
            
            Other Parameters
            ----------
            **x: numpy.ndarray
                universe of discourse 
            
            **mfx: numpy.ndarray,
                        "aggregated" membership functions
            
            Return
            -------
            y: float
                defuzzified value
        """
        method = kwargs['method']
        x = kwargs['x']
        mfx = kwargs['mfx']

        return fuzz.defuzz(x, mfx, method)