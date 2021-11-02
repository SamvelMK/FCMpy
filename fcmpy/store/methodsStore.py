from abc import ABC, abstractmethod
from fcmpy.expert_fcm.input_validator import type_check


class GetMethod(ABC):
    """
        Get methods from a store.
    """ 
    @abstractmethod
    def get(**kwargs):
        raise NotImplementedError('Get method is not defined!')


# Expert-based FCMs
class ReaderStore(GetMethod):
    """
        Methods of reading data files.
    """
    from fcmpy.expert_fcm.reader import XLSX
    from fcmpy.expert_fcm.reader import CSV
    from fcmpy.expert_fcm.reader import  JSON

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
    from fcmpy.expert_fcm.entropy import InformationEntropy

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
    from fcmpy.expert_fcm.membership import TriangularMembership
    from fcmpy.expert_fcm.membership import GaussianMembership
    from fcmpy.expert_fcm.membership import TrapezoidalMembership

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
    from fcmpy.expert_fcm.implication import Mamdani
    from fcmpy.expert_fcm.implication import Larsen

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
    from fcmpy.expert_fcm.aggregation import Fmax
    from fcmpy.expert_fcm.aggregation import AlgSum
    from fcmpy.expert_fcm.aggregation import EinsteinSum
    from fcmpy.expert_fcm.aggregation import HamacherSum

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
    from fcmpy.expert_fcm.defuzz import Centroid
    from fcmpy.expert_fcm.defuzz import Bisector
    from fcmpy.expert_fcm.defuzz import MeanOfMax
    from fcmpy.expert_fcm.defuzz import MinOfMax
    from fcmpy.expert_fcm.defuzz import MaxOfMax

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
    from fcmpy.simulator.inference import Kosko
    from fcmpy.simulator.inference import ModifiedKosko
    from fcmpy.simulator.inference import Rescaled

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
    from fcmpy.simulator.transfer import Sigmoid
    from fcmpy.simulator.transfer import Bivalent
    from fcmpy.simulator.transfer import Trivalent
    from fcmpy.simulator.transfer import HyperbolicTangent
    __methods = {'sigmoid' : Sigmoid, 'bivalent' : Bivalent, 'trivalent' : Trivalent, 'tanh' : HyperbolicTangent}
    
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
    from fcmpy.simulator.convergence import AbsDifference

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
    from fcmpy.intervention.interventionConstructor import SingleShot
    from fcmpy.intervention.interventionConstructor import Continuous
    __methods = {'single_shot' : SingleShot, 'continuous' : Continuous}

    @staticmethod
    @type_check
    def get(method:str):
        if method in InterventionStore.__methods.keys():
            return InterventionStore.__methods[method]
        else:
            raise ValueError('The intervention type is not defined.')


# RCGA
class InitializationStore(GetMethod):
    """
        Initialization methods for the RCGA.
    """
    from fcmpy.ml.genetic.initialization import UniformInitialize

    __methods = {'uniform' : UniformInitialize}

    @staticmethod
    @type_check
    def get(method:str):
        if method in InitializationStore.__methods.keys():
            return InitializationStore.__methods[method]
        else:
            raise ValueError('The initialization method is not defined.')


# Normalization for RCGA
class NormalizationStore(GetMethod):  
    """
        Methods of normalizing the RCGA fitness function.
    """
    from fcmpy.ml.genetic.normalization import NT
    from fcmpy.ml.genetic.normalization import T
    __methods = {'L1' : NT, 'L2' : NT, 'LInf' : T}

    @staticmethod
    @type_check
    def get(method:str):
        if method in NormalizationStore.__methods.keys():
            return NormalizationStore.__methods[method]
        else:
            raise ValueError('The normalization method is not defined.')


# Auxilary methods for RCGA
class AuxiliaryStore(GetMethod):
    """
        Auxiliary functions for the RCGA fitness function.
    """
    from fcmpy.ml.genetic.auxiliary import H

    __methods = {'h' : H}

    @staticmethod
    @type_check
    def get(method:str):
        if method in AuxiliaryStore.__methods.keys():
            return AuxiliaryStore.__methods[method]
        else:
            raise ValueError('The auxiliary function is not defined.')


# Error functions for RCGA
class MatrixErrorStore(GetMethod):
    """
        Methods for calculating matrix error for the RCGA fitness function.
    """
    from fcmpy.ml.genetic.matrix_error import StachError

    __methods = {'stach_error' : StachError}

    @staticmethod
    @type_check
    def get(method:str):
        if method in MatrixErrorStore.__methods.keys():
            return MatrixErrorStore.__methods[method]
        else:
            raise ValueError('The error function is not defined.')


# Fitness functions for RCGA
class FitnessStore(GetMethod):
    """
        Methods for calculating fitness.
    """
    from fcmpy.ml.genetic.fitness import StachFitness

    __methods = {'stach_fitness' : StachFitness}

    @staticmethod
    @type_check
    def get(method:str):
        if method in FitnessStore.__methods.keys():
            return FitnessStore.__methods[method]
        else:
            raise ValueError('The fitness function is not defined.')


# Selection functions for RCGA
class SelectionStore(GetMethod):
    """
        Methods of selection for RCGA.
    """
    from fcmpy.ml.genetic.selection import Tournament
    from fcmpy.ml.genetic.selection import RouletteWheel


    __methods = {'tournament' : Tournament, 'roulette':RouletteWheel}

    @staticmethod
    @type_check
    def get(method:str):
        if method in SelectionStore.__methods.keys():
            return SelectionStore.__methods[method]
        else:
            raise ValueError('The select function is not defined.')


# Recombination functions for RCGA
class RecombinationStore(GetMethod):
    """
        Methods of selection for RCGA.
    """
    from fcmpy.ml.genetic.recombination import OnePointCrossover
    from fcmpy.ml.genetic.recombination import TwoPointCrossover


    __methods = {'one_point_crossover' : OnePointCrossover, 'two_point_crossover':TwoPointCrossover}

    @staticmethod
    @type_check
    def get(method:str):
        if method in RecombinationStore.__methods.keys():
            return RecombinationStore.__methods[method]
        else:
            raise ValueError('The recombine function is not defined.')


# Mutation functions for RCGA
class MutationStore(GetMethod):
    """
        Methods of mutation operations for RCGA.
    """
    from fcmpy.ml.genetic.mutation import RandomMutation
    from fcmpy.ml.genetic.mutation import NonUniformMutation


    __methods = {'random' : RandomMutation, 'non_uniform':NonUniformMutation}

    @staticmethod
    @type_check
    def get(method:str):
        if method in MutationStore.__methods.keys():
            return MutationStore.__methods[method]
        else:
            raise ValueError('The mutate function is not defined.')


# Replacement functions for RCGA
class ReplacementStore(GetMethod):
    """
        Methods of replacing candidate solutions for steady state RCGA.
    """
    from fcmpy.ml.genetic.replacement import CdrwReplacement

    __methods = {'CRDW': CdrwReplacement}

    @staticmethod
    @type_check
    def get(method:str):
        if method in ReplacementStore.__methods.keys():
            return ReplacementStore.__methods[method]
        else:
            raise ValueError('The replace function is not defined.')
