###########################################################################
##            Classes for recombination strategies                       ##
###########################################################################
import numpy as np
from abc import ABC
from abc import abstractmethod


class Recombination(ABC):
    """
        Interface for recombination classes for RCGA.
    """
    @abstractmethod
    def recombine(**kwargs):
        raise NotImplementedError('recombine method is not defined.')


class OnePointCrossover(Recombination):
    """
        One point crossover
    """
    def recombine(**kwargs)->tuple:
        """
            One-point crossover operation

            Parameters
            ----------
            parent1: np.ndarray
                        candidate solution
            
            parent2: np.ndarray
                        candidate solution
            
            Return
            ------
            y: tuple
                childOne, childTwo
        """
        parentOne = kwargs['parentOne']['solution'].copy().flatten()
        parentTwo = kwargs['parentTwo']['solution'].copy().flatten()

        childOne = {'solution': None, 'fitness' : 0}
        childTwo = {'solution': None, 'fitness' : 0}
        nConcepts = len(kwargs['parentOne']['solution'])

        # Determine the crossover point
        crossover_point = np.random.choice(range(nConcepts**2))

        childOne['solution'] = np.array(list(parentOne[:crossover_point]) + list(parentTwo[crossover_point:])).reshape(nConcepts, nConcepts)
        childTwo['solution'] = np.array(list(parentOne[crossover_point:]) + list(parentTwo[:crossover_point])).reshape(nConcepts, nConcepts)

        return childOne, childTwo
        

class TwoPointCrossover(Recombination):
    def recombine(**kwargs):
        # for future
        pass