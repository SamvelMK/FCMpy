###########################################################################
##                    Classes of replacement strategies                  ##
###########################################################################
import numpy as np
from scipy.spatial import distance
from abc import ABC
from abc import abstractmethod
import copy

class Replacement(ABC):
    """
        Interface for replacement classes
    """
    @abstractmethod
    def replace(**kwargs):
        raise NotImplementedError('replace method is not defined.')


class CdrwReplacement(Replacement):
    """
        Replacement based on contribution to diversity and replace worst strategy
    """
    @staticmethod
    def __diversity(chromosome1, chromosome2):
        """
            Calculate the diversity between a chromosome and the elements in the population based on Euclidean distance

            Parameters
            ----------
            chromosome: dict
                            chromosome whose diversity should be calculated

            population: dict
                        population of chromosomes

            Return
            --------
            y: float, 
                diversity score for a chromosome
        """
        divScore = distance.euclidean(chromosome1['solution'].flatten(), chromosome2['solution'].flatten())
        return divScore
        
    @staticmethod
    def __lessFitPop(chromosome, population):
        """
            Find a subpopulation whose elements have less fit than the passed chromosome.

            Parameters
            ----------
            chromosome: dict
                            child chromosome

            population: list of dict 
                            population of chromosomes

            Return
            --------
            y: list of dict 
                a subpopulation P
        """
        p = {}
        for i in population:
            if chromosome['fitness'] > population[i]['fitness']:
                p[i] = copy.deepcopy(population[i])
        lessFitPop = dict(sorted(p.items(), key=lambda k: k[1]['fitness']))
        
        return lessFitPop

    def replace(**kwargs):
        """
            Replacement based on contribution to diversity and replace worst.

            Parameters
            ----------
            child: dict
                        offspring 
                                
            population: dict
                        population of chromosomes

            Return
            --------
            y: list, 
                new population of chromosomes.
        """
        pop = kwargs['population']
        child = kwargs['child']
        # find the less fit chromosomes in the population for childOne and childTwo
        # if such subpopulation does not exist then the child does not survive
        childLessFitPop = CdrwReplacement.__lessFitPop(chromosome=child, population=pop)
        
        if childLessFitPop:
            childDiv = []
            for i in childLessFitPop:
                childDiv.append(CdrwReplacement.__diversity(chromosome1=child, chromosome2=childLessFitPop[i]))
            childDiv = min(childDiv)
            dvLessFitPop = {}
            for i in childLessFitPop:
                for y in pop:
                    if i != y:
                        dvLessFitPop[i] = CdrwReplacement.__diversity(chromosome1=childLessFitPop[i], chromosome2=pop[y])
            if childDiv > min(list(dvLessFitPop.values())):
                cmin = {k: v for k, v in dvLessFitPop.items() if v == min(dvLessFitPop.values())}
                pop[list(cmin.keys())[0]] = child
                return pop
            else:
                # replace worst
                pop[list(childLessFitPop.keys())[0]] = child
                return pop
        else: # if no replacement was done (i.e., child did not survive)
            return pop