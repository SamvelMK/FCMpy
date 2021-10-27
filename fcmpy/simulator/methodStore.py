from fcmpy.simulator.inference import Kosko, ModifiedKosko, Rescaled
from fcmpy.simulator.transfer import Sigmoid, Bivalent, Trivalent, HyperbolicTangent
from fcmpy.expert_fcm.methodsStore import GetMethod
from fcmpy.expert_fcm.input_validator import type_check
from abc import ABC, abstractmethod

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

    __methods = {'sigmoid' : Sigmoid, 'bivalent' : Bivalent, 'trivalent' : Trivalent, 'tanh' : HyperbolicTangent}

    
    @staticmethod
    @type_check
    def get(method:str):
        if method in TransferStore.__methods.keys():
            return TransferStore.__methods[method]
        else:
            raise ValueError('The transfer method is not defined.')
