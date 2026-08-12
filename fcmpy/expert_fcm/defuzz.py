from abc import ABC
import skfuzzy as fuzz
from fcmpy.expert_fcm.input_validator import type_check


class Defuzzification(ABC):
    """
        Defuzzification methods.

        skfuzzy.defuzz() is itself dispatched on its 'method' argument, so every
        rule below (centroid, bisector, mom, som, lom) shares this one
        implementation. The named subclasses exist so DefuzzStore can look them
        up by name and so each documents what its method means; none of them
        need to override defuzz().
    """
    @staticmethod
    @type_check
    def defuzz(**kwargs) -> float:
        """
            Defuzzify an aggregated membership function.

            Other Parameters
            ----------------
            **x: numpy.ndarray
                universe of discourse

            **mfx: numpy.ndarray,
                        "aggregated" membership functions

            **method: str,
                        the skfuzzy defuzzification method name
                        ('centroid', 'bisector', 'mom', 'som', 'lom')

            Return
            -------
            y: float
                defuzzified value
        """
        method = kwargs['method']
        x = kwargs['x']
        mfx = kwargs['mfx']

        return fuzz.defuzz(x, mfx, method)


class Centroid(Defuzzification):
    """
        Centroid difuzzification method (i.e., center of gravity).
    """


class Bisector(Defuzzification):
    """
        Bisector difuzzification method.
    """


class MeanOfMax(Defuzzification):
    """
        MeanOfMax difuzzification method.
    """


class MinOfMax(Defuzzification):
    """
        MinOfMax difuzzification method.
    """


class MaxOfMax(Defuzzification):
    """
        MaxOfMax difuzzification method.
    """
