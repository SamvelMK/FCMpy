###########################################################################
##                    Classes for selection strategies                   ##
###########################################################################
import numpy as np
from abc import ABC
from abc import abstractmethod


class Selection(ABC):
    """
        Interface for selection classes for RCGA.
    """
    @abstractmethod
    def select(**kwargs):
         raise NotImplementedError('select method is not defined.')


class Tournament(Selection):
    """
        Tournament selection of candidate solutions
    """
    @staticmethod
    def select(**kwargs)->dict:
        """
            Tournament selection.

            Parameters
            ----------            
            n_participants: int
                            number of participants in the tournament
            
            population: int
                        population to be selected from
            
            Return
            ------
            y: dict
                selected chromosomes
        """
        n_participants = kwargs['n_participants']
        population = kwargs['population']
        selected_indexes = np.random.choice(list(population.keys()), size = n_participants)
        selected = {}

        s = sorted([population[i] for i in population if i in selected_indexes.flatten()], key = lambda i: i['fitness'], reverse=True)     
        selected[0] = s[0].copy()
        selected[1] = s[1].copy()

        return selected


class RouletteWheel(Selection):
    """
        Roulette wheel selection of candidate solutions
    """
    @staticmethod
    def select(**kwargs)->dict:
        """
            Roulette wheel selection

            Parameters
            ----------
            population: dict
                        population to be selected
            
            size: int
                    number of candidates (chromosomes) to be selected
            
            population: int
                        population to be selected from
            
            Return
            ------
            y: dict
                selected chromosomes 
        """
        population = kwargs['population']
        size = kwargs['size']
        f_score = [population[i]['fitness'] for i in population]
        total_f = np.sum(f_score)
        p = [i/total_f for i in f_score]

        selected_indexes = np.random.choice(list(population.keys()), size = size, p=p, replace=False)
        selected_solutions = {k:population.get(k,None).copy() for k in selected_indexes}

        return selected_solutions