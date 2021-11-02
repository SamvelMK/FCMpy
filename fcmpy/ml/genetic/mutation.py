###########################################################################
##                   Classes for mutation operations                     ##
###########################################################################
import numpy as np
from abc import ABC
from abc import abstractmethod
from fcmpy.expert_fcm.input_validator import type_check


class Mutation(ABC):
    """
        Interface for mutation classes for RCGA.
    """
    @abstractmethod
    def mutate(**kwargs):
        raise NotImplementedError('mutate method is not defined.')


class RandomMutation(Mutation):
    """
        Random uniform mutation based on Michalewicz, 1992.
    """
    def mutate(**kwargs)->dict:
        """
            Random uniform mutation based on Michalewicz, 1992.

            Parameters
            ----------
            chromosome: dict
                        the candidate FCM solution (chromosome). key --> solution,
                        fitness, value --> np.ndarray, float
            
            p_mutation: float
                            the mutation rate

            Return
            ------
            y: dict
                the mutated chromosome
        """
        chromosome = kwargs['chromosome'].copy()
        p_mutation = kwargs['p_mutation']

        for i in range(len(chromosome['solution'])):
            for j in range(len(chromosome['solution'])):
                change = np.random.choice([False, True], p=[1 - p_mutation, p_mutation]) # Decide whether the given gene should be mutated with p = p_mutation.
                if change:
                    chromosome['solution'][i,j] = np.random.uniform(-1,1)

        return chromosome


class NonUniformMutation(Mutation):
    """
        Non-uniform mutation based on Michalewicz, 1992.
    """
    @staticmethod
    @type_check
    def __nonUniCoef(t:int, T:int, y:float, b:int):
        """
            Calculate a coefficient for non uniform mutation

            Parameters
            ----------
            t: int
                current iteration

            T: int
                total number of iteration
            
            y: float

            b: int
                system parameter
            
            Return
            ------
            y: float
                a coefficient for the non uniform mutation (range [0,1])
        """
        r = np.random.uniform(0,1)
        mutationCoefficient = y*(1-r**((1-t/T)**b))

        return mutationCoefficient

    @staticmethod
    def mutate(**kwargs)->dict:
        """
            Non-uniform mutation based on Michalewicz, 1992.

            Parameters
            ----------
            chromosome: dict
                        the candidate FCM solution (chromosome). key -> solution,
                        fitness, value --> np.ndarray, float

            p_mutation: float
                            mutation rate

            max_generation: int
                            number of generations the RCGA should run

            n_iteration: int
                            the nth iteration of the RCGA

            b: int
                system parameter determining the degree of dependence
                of the mutation on the number of generation.

            Return
            ------
            y: dict
                the mutated chromosome
        """
        chromosome = kwargs['chromosome'].copy()
        p_mutation = kwargs['p_mutation']
        max_generations = kwargs['max_generations']
        nthIteration = kwargs['nth_Iteration']
        b = kwargs['b']

        for i in range(len(chromosome['solution'])):
            for j in range(len(chromosome['solution'])):
                change = np.random.choice([False, True], p=[1 - p_mutation, p_mutation]) # Decide whether the given gene should be mutated with p = p_mutation.
                if change:
                    m = np.random.randint(2, size=1)
                    if m > 0:
                        y = chromosome['solution'][i,j] + 1
                        # chromosome['solution'][i,j] = chromosome['solution'][i,j] - NonUniformMutation.__nonUniCoef(t=nthIteration, T=max_generations, y=y, b=b) 
                    else:
                        y = 1 - chromosome['solution'][i,j]
                        chromosome['solution'][i,j] = chromosome['solution'][i,j] + NonUniformMutation.__nonUniCoef(t=nthIteration, T=max_generations, y=y, b=b)

        return chromosome