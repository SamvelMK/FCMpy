from abc import ABC
from fcmpy.expert_fcm.input_validator import type_check


class GetMethod(ABC):
    """
        Get methods from a store.

        Every *Store subclass just declares its own _methods dict (name -> class);
        get() is identical across all of them and lives here once. Note: the dict
        is named with a single leading underscore (not name-mangled) so that this
        shared implementation -- defined in GetMethod, not in the subclass -- can
        resolve cls._methods correctly regardless of which subclass it's called on.
    """
    _methods = {}

    @classmethod
    @type_check
    def get(cls, method: str):
        if method in cls._methods:
            return cls._methods[method]
        else:
            raise ValueError(f"'{method}' is not a valid {cls.__name__} method. "
                                f"Valid options are: {sorted(cls._methods)}.")


# Expert-based FCMs
class ReaderStore(GetMethod):
    """
        Methods of reading data files.
    """
    from fcmpy.expert_fcm.reader import XLSX
    from fcmpy.expert_fcm.reader import CSV
    from fcmpy.expert_fcm.reader import  JSON

    _methods = {'csv' : CSV, 'xlsx' : XLSX, 'json' : JSON}


class EntropyStore(GetMethod):
    """
        Methods of calculating entropy.
    """
    from fcmpy.expert_fcm.entropy import InformationEntropy

    _methods = {'entropy' : InformationEntropy}


class MembershipStore(GetMethod):
    """
        Methods of generating membership functions.
    """
    from fcmpy.expert_fcm.membership import TriangularMembership
    from fcmpy.expert_fcm.membership import GaussianMembership
    from fcmpy.expert_fcm.membership import TrapezoidalMembership

    _methods = {'trimf' : TriangularMembership, 'gaussmf': GaussianMembership, 'trapmf' : TrapezoidalMembership}


class ImplicationStore(GetMethod):
    """
        Fuzzy implication rules.
    """
    from fcmpy.expert_fcm.implication import Mamdani
    from fcmpy.expert_fcm.implication import Larsen

    _methods = {'Mamdani' : Mamdani, 'Larsen' : Larsen}


class AggregationStore(GetMethod):
    """
        Fuzzy aggregation rules.
    """
    from fcmpy.expert_fcm.aggregation import Fmax
    from fcmpy.expert_fcm.aggregation import AlgSum
    from fcmpy.expert_fcm.aggregation import EinsteinSum
    from fcmpy.expert_fcm.aggregation import HamacherSum

    _methods = {'fMax' : Fmax, 'algSum' : AlgSum, 'eSum' : EinsteinSum, 'hSum' : HamacherSum}


class DefuzzStore(GetMethod):
    """
        Defuzzification methods.
    """
    from fcmpy.expert_fcm.defuzz import Centroid
    from fcmpy.expert_fcm.defuzz import Bisector
    from fcmpy.expert_fcm.defuzz import MeanOfMax
    from fcmpy.expert_fcm.defuzz import MinOfMax
    from fcmpy.expert_fcm.defuzz import MaxOfMax

    _methods = {'centroid' : Centroid, 'bisector' : Bisector, 'mom' : MeanOfMax, 'som' : MinOfMax, 'lom' : MaxOfMax}


# Simulator
class InferenceStore(GetMethod):
    """
        Methods of FCM inference.
    """
    from fcmpy.simulator.inference import Kosko
    from fcmpy.simulator.inference import ModifiedKosko
    from fcmpy.simulator.inference import Rescaled

    _methods = {'kosko' : Kosko, 'mKosko' : ModifiedKosko, 'rescaled' : Rescaled}


class TransferStore(GetMethod):
    """
        Methods of FCM Transfer.
    """
    from fcmpy.simulator.transfer import Sigmoid
    from fcmpy.simulator.transfer import Bivalent
    from fcmpy.simulator.transfer import Trivalent
    from fcmpy.simulator.transfer import HyperbolicTangent

    _methods = {'sigmoid' : Sigmoid, 'bivalent' : Bivalent, 'trivalent' : Trivalent, 'tanh' : HyperbolicTangent}


class ConvergenceStore(GetMethod):
    """
        Methods for checking the convergence.
    """
    from fcmpy.simulator.convergence import AbsDifference

    _methods = {'absDiff' : AbsDifference}


# Intervention
class InterventionStore(GetMethod):
    """
        Methods of FCM Interventions.
    """
    from fcmpy.intervention.interventionConstructor import SingleShot
    from fcmpy.intervention.interventionConstructor import Continuous

    _methods = {'single_shot' : SingleShot, 'continuous' : Continuous}


# RCGA
class InitializationStore(GetMethod):
    """
        Initialization methods for the RCGA.
    """
    from fcmpy.ml.genetic.initialization import UniformInitialize

    _methods = {'uniform' : UniformInitialize}


# Normalization for RCGA
class NormalizationStore(GetMethod):
    """
        Methods of normalizing the RCGA fitness function.
    """
    from fcmpy.ml.genetic.normalization import NT
    from fcmpy.ml.genetic.normalization import T

    _methods = {'L1' : NT, 'L2' : NT, 'LInf' : T}


# Auxilary methods for RCGA
class AuxiliaryStore(GetMethod):
    """
        Auxiliary functions for the RCGA fitness function.
    """
    from fcmpy.ml.genetic.auxiliary import H

    _methods = {'h' : H}


# Error functions for RCGA
class MatrixErrorStore(GetMethod):
    """
        Methods for calculating matrix error for the RCGA fitness function.
    """
    from fcmpy.ml.genetic.matrix_error import StachError

    _methods = {'stach_error' : StachError}


# Fitness functions for RCGA
class FitnessStore(GetMethod):
    """
        Methods for calculating fitness.
    """
    from fcmpy.ml.genetic.fitness import StachFitness

    _methods = {'stach_fitness' : StachFitness}


# Selection functions for RCGA
class SelectionStore(GetMethod):
    """
        Methods of selection for RCGA.
    """
    from fcmpy.ml.genetic.selection import Tournament
    from fcmpy.ml.genetic.selection import RouletteWheel

    _methods = {'tournament' : Tournament, 'roulette':RouletteWheel}


# Recombination functions for RCGA
class RecombinationStore(GetMethod):
    """
        Methods of recombination for RCGA.
    """
    from fcmpy.ml.genetic.recombination import OnePointCrossover
    from fcmpy.ml.genetic.recombination import TwoPointCrossover

    _methods = {'one_point_crossover' : OnePointCrossover, 'two_point_crossover':TwoPointCrossover}


# Mutation functions for RCGA
class MutationStore(GetMethod):
    """
        Methods of mutation operations for RCGA.
    """
    from fcmpy.ml.genetic.mutation import RandomMutation
    from fcmpy.ml.genetic.mutation import NonUniformMutation

    _methods = {'random' : RandomMutation, 'non_uniform':NonUniformMutation}


# Replacement functions for RCGA
class ReplacementStore(GetMethod):
    """
        Methods of replacing candidate solutions for steady state RCGA.
    """
    from fcmpy.ml.genetic.replacement import CdrwReplacement

    _methods = {'CRDW': CdrwReplacement}
