###########################################################################
##                       Classes of fitness methods                      ##
###########################################################################
from abc import ABC
from abc import abstractmethod
from typing import Union
from fcmpy.store.methodsStore import AuxiliaryStore
from fcmpy.store.methodsStore import NormalizationStore
from fcmpy.store.methodsStore import MatrixErrorStore


class GaFit(ABC):
    """
        Interface for fitness classes for RCGA.
    """
    @abstractmethod
    def calculate_fitness(**kwargs):
        raise NotImplementedError('calculate_fitness method is not defined.')


class StachFitness(GaFit):
    """
        Fitness function based on Stach et al. 2005
    """
    @staticmethod
    def calculate_fitness(**kwargs) -> Union[float, int]:
        """
            Calculate the fitness of a candidate FCM (chromosome).

            Parameters
            ----------
            normalization_type: str
                                type of normalization function to be used

            data_simulated: pd.DataFrame
                                data generated based on the candidate FCM (chromosome)
            
            data: pd.DataFrame
                    real data
            
            p: int
                parameter for the matrix error

            a: int
                parameter for the auxiliary function

            Return
            ------
            y: float
                fitness score of a candidate solution
        """
        normalization_type = kwargs['normalization_type']
        data_simulated = kwargs['data_simulated']
        data = kwargs['data']
        a = kwargs['a']
        p= kwargs['p']
        
        error_type = 'stach_error'
        auxiliary_function_type = 'h'
        t = len(data)
        nConcepts = len(data.keys())

        # Get the methods for calculating the fitness function
        auxiliary = AuxiliaryStore.get(method=auxiliary_function_type)
        normalization = NormalizationStore.get(method=normalization_type)
        error = MatrixErrorStore.get(method=error_type)
        
        # Calculate the matrix error
        matError= error.calculate(data_simulated=data_simulated, data=data, p=p)
        # Normalize the calculated error
        normalized = normalization.normalize(x=matError, n=nConcepts, t=t)
        # Apply the auxiliary function on the normalized matrix error
        aux = auxiliary.f(x=normalized, a=a)

        return aux