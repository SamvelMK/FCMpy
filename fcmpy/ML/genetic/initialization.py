###########################################################################
##       Classes for initializing a population of candidate solutions    ##
###########################################################################
import numpy as np
from abc import ABC
from abc import abstractmethod


class Initialization(ABC):
    """
        Interface for initialization classes for RCGA.
    """
    @abstractmethod
    def initialize(**kwargs):
        raise NotImplementedError('initialize method is not defined.')


class UniformInitialize(Initialization):
    """
        Uniform generation of candidate solutions (chromosomes).
    """
    @staticmethod
    def initialize(**kwargs) -> dict:
        """
            Create a population of solutions (chromosomes) for the RCGA

            Parameters
            ----------
            population_size: int
                                number of solutions (population) to be generated
            n_concepts: int
                        number of geners in the chromosome (number of concepts in the FCM)
            
            Return
            ------
            y: dict
                dictionary keys ---> soluation (array of chromosome), fitness ---> float (fitness of a given chromosome)
        """
        population_size = kwargs['population_size']
        n_concepts = kwargs['n_concepts']
        generations = {}
        for i in range(population_size):
            generations[i] = {'solution': np.random.uniform(low=-1, high=1, size=(n_concepts, n_concepts)), 'fitness' : 0}
        return generations