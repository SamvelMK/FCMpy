# Store all the methods here with their respective get methods.

# Imports for the expert based FCMs
from fcmpy.expert_fcm.reader import XLSX, CSV, JSON
from fcmpy.expert_fcm.entropy import InformationEntropy
from fcmpy.expert_fcm.membership import TriangularMembership, GaussianMembership, TrapezoidalMembership
from fcmpy.expert_fcm.implication import Mamdani, Larsen
from fcmpy.expert_fcm.aggregation import Fmax, AlgSum, EinsteinSum, HamacherSum
from fcmpy.expert_fcm.defuzz import Centroid, Bisector, MeanOfMax, MinOfMax, MaxOfMax

# Imports for the Simulator
from fcmpy.simulator.convergence import AbsDifference
from fcmpy.simulator.inference import Kosko, ModifiedKosko, Rescaled
from fcmpy.simulator.transfer import Sigmoid, Bivalent, Trivalent, HyperbolicTangent
from fcmpy.simulator.convergence import AbsDifference

# Imports for the Interventions
from fcmpy.intervention.interventionConstructor import SingleShot, Continuous

# other imports
from fcmpy.expert_fcm.input_validator import type_check
from abc import ABC, abstractmethod

class GetMethod(ABC):

    """
    Get methods from a store.
    """
    
    @abstractmethod
    def get():
        raise NotImplementedError('Get method is not defined!')

# Expert-based FCMs
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

# Simulator
class InferenceStore(GetMethod):
    
    """
    Methods of FCM inference.
    """

    __methods = {'kosko' : Kosko, 'mKosko' : ModifiedKosko, 'rescaled' : Rescaled}

    
    @staticmethod
    @type_check
    def get(method:str):
        if method in InferenceStore.__methods.keys():
            return InferenceStore.__methods[method]
        else:
            raise ValueError('The ifnerence method is not defined.')

class TransferStore(GetMethod):

    """
    Methods of FCM Transfer.
    """

    __methods = {'sigmoid' : Sigmoid, 'bi' : Bivalent, 'tri' : Trivalent, 'tanh' : HyperbolicTangent}

    
    @staticmethod
    @type_check
    def get(method:str):
        if method in TransferStore.__methods.keys():
            return TransferStore.__methods[method]
        else:
            raise ValueError('The transfer method is not defined.')

class ConvergenceStore(GetMethod):

    """
    Methods for checking the convergence.
    """

    __methods = {'absDiff' : AbsDifference}

    @staticmethod
    @type_check
    def get(method:str):
        if method in ConvergenceStore.__methods.keys():
            return ConvergenceStore.__methods[method]
        else:
            raise ValueError('The convergence method is not defined.')


# Intervention
class InterventionStore(GetMethod):
    
    """
    Methods of FCM Interventions.
    """

    __methods = {'single_shot' : SingleShot, 'continuous' : Continuous}

    @staticmethod
    @type_check
    def get(method:str):
        if method in InterventionStore.__methods.keys():
            return InterventionStore.__methods[method]
        else:
            raise ValueError('The intervention type is not defined.')