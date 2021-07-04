from fcmpy.expert_fcm.methodsStore import GetMethod
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.intervention.interventionConstructor import SingleShot, Continuous

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