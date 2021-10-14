# Store all the methods here with their respective get methods.

from fcmpy.expert_fcm.reader import XLSX, CSV, JSON
from fcmpy.expert_fcm.entropy import InformationEntropy
from fcmpy.expert_fcm.membership import TriangularMembership, GaussianMembership, TrapezoidalMembership
from fcmpy.expert_fcm.implication import Mamdani, Larsen
from fcmpy.expert_fcm.aggregation import Fmax, AlgSum, EinsteinSum, HamacherSum
from fcmpy.expert_fcm.defuzz import Centroid, Bisector, MeanOfMax, MinOfMax, MaxOfMax
from fcmpy.expert_fcm.input_validator import type_check

from abc import ABC, abstractmethod

class GetMethod(ABC):

    """
    Get methods from a store.
    """
    
    @abstractmethod
    def get():
        raise NotImplementedError('Get method is not defined!')

class ReaderStore(GetMethod):

    """
    Methods of reading data files.
    """

    __methods = {'csv' : CSV, 'xlsx' : XLSX, 'json' : JSON}

    @staticmethod
    @type_check
    def get(method:str):
        if method in ReaderStore.__methods.keys():
            return ReaderStore.__methods[method]
        else:
            raise ValueError('The reader method is not defined.')

class EntropyStore(GetMethod):

    """
    Methods of calculating entropy.
    """

    __methods = {'entropy' : InformationEntropy}

    @staticmethod
    @type_check
    def get(method:str):
        if method in EntropyStore.__methods.keys():
            return EntropyStore.__methods[method]
        else:
            raise ValueError('The enrtopy method is not defined.')

class MembershipStore(GetMethod):

    """
    Methods of generating membership functions.
    """

    __methods = {'trimf' : TriangularMembership, 'gaussmf': GaussianMembership, 'trapmf' : TrapezoidalMembership}

    @staticmethod
    @type_check
    def get(method:str):
        if method in MembershipStore.__methods.keys():
            return MembershipStore.__methods[method]
        else:
            raise ValueError('The membership method is not defined.')

class ImplicationStore(GetMethod):

    """
    Fuzzy implication rules.
    """

    __methods = {'Mamdani' : Mamdani, 'Larsen' : Larsen}

    @staticmethod
    @type_check
    def get(method:str):
        if method in ImplicationStore.__methods.keys():
            return ImplicationStore.__methods[method]
        else:
            raise ValueError('The implication method is not defined.')

class AggregationStore(GetMethod):

    """
    Fuzzy aggregation rules.
    """

    __methods = {'fMax' : Fmax, 'algSum' : AlgSum, 'eSum' : EinsteinSum, 'hSum' : HamacherSum}

    @staticmethod
    @type_check
    def get(method:str):
        if method in AggregationStore.__methods.keys():
            return AggregationStore.__methods[method]
        else:
            raise ValueError('The aggregation method is not defined.')

class DefuzzStore(GetMethod):

    """
    Defuzzification methods.
    """

    __methods = {'centroid' : Centroid, 'bisector' : Bisector, 'mom' : MeanOfMax, 'som' : MinOfMax, 'lom' : MaxOfMax}

    @staticmethod
    @type_check
    def get(method:str):
        if method in DefuzzStore.__methods.keys():
            return DefuzzStore.__methods[method]
        else:
            raise ValueError('The defuzzification method is not defined.')
