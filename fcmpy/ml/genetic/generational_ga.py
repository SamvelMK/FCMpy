###########################################################################
##                    Implementation of the Generationa GA               ##
###########################################################################
import numpy as np
import pandas as pd
from tqdm import tqdm
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.ml.genetic.ga_interface import GA
from fcmpy.store.methodsStore import InitializationStore
from fcmpy.store.methodsStore import SelectionStore
from fcmpy.store.methodsStore import RecombinationStore
from fcmpy.store.methodsStore import MutationStore
from fcmpy.ml.genetic.evaluation import PopulationEvaluation
import copy


class GRCGA(GA):
    """
        Generational Real-Coded Algorithm for learning FCM connection matrices

        Parameters
        ----------
        data: pd.DataFrame
                longitudinal data
        
        init_type: str
                    type of initialization
                    default --> 'uniform'
        
        population_size: int
                            number of candidates to generate 
                            as an initial population
                            default --> 100
        
        fitness_type: str
                        fitness function
                        default --> 'stach_fitness'
        
        normalization_type: str
                            type of normalization for the matrix error
                            default --> 'L2'

        a: int
            coefficient for the auxiliary function
            default --> 100

        p: int
            coefficient for the matrix error
            default --> 2 (for L2 norm.)
        
        inference: str
                    inference type for the FCM update function
                    default --> 'mKosko'
        
        tranfer: str
                 type of transfer function to be applied during the FCM update
                 default --> 'sigmoid'
        
        l: int
            slope parameter for the sigmoid function
            default --> 1
    """
    @type_check
    def __init__(self, data:pd.DataFrame, init_type:str='uniform', population_size:int=100,
                    fitness_type:str='stach_fitness', normalization_type:str='L2', a:int=100, p:int=2,
                    inference:str='mKosko', transfer:str='sigmoid', l:int=1, **kwargs):
        # Check the parameter ranges
        assert population_size % 2 == 0 # check if population size is even
        assert p > 0
        assert a > 0

        self.__initialize = InitializationStore.get(init_type)
        self.__popSize = population_size
        self.__nConcepts = len(data.keys())
        self.__data = data
        self.__evaluation = PopulationEvaluation()
        self.__fitness_type = fitness_type
        self.__normalization_type = normalization_type
        self.__a = a
        self.__p = p
        self.__inference = inference
        self.__transfer = transfer
        self.__l=l

        # Step 1: Initialize the population
        population = self.__initialize.initialize(population_size = self.__popSize, 
                                                            n_concepts = self.__nConcepts, params=kwargs)
        # Evaluate the fitness of the candidate chromosomes in the population
        self.__population = self.__evaluation.evaluate(population=population, data=self.__data, l=self.__l, 
                                                            transfer=self.__transfer, inference=self.__inference,
                                                            fitness_type=self.__fitness_type, normalization_type=self.__normalization_type,
                                                            a=self.__a, p=self.__p)

    @type_check
    def run(self, n_iterations:int=100000, n_participants:int=5, recombination_type:str='one_point_crossover',
                p_recombination:float=0.9, p_mutation:float=0.5, b:int=5, threshold:float=0.9, **kwargs):
        """
            Run the RCGA learning based on the generational approach

            Parameters
            ----------
            n_participants: int
                            number of participants in the turnament selection
                            deault -> 5
            
            p_mutation: float
                            probability of mutation
                            default -> 0.5
            
            p_recombination: float 
                                probability of recombination
                                default -> 0.9
            
            threshold: float
                        threshold of error (criteria for stopping the learning process)
                        default -> 0.999
            
            Return
            ------
            y: dict
                identified solution
        """
        # Check the parameter ranges
        assert n_iterations > 0
        assert n_participants > 0
        assert 0 <= p_recombination <= 1
        assert 0 <= p_mutation <= 1
        assert b > 0
        assert 0 <= threshold <= 1

        # Set up the recombination method
        recombination = RecombinationStore.get(recombination_type)
        # Create a copy of the initial population
        old_population = copy.deepcopy(self.__population)
        # Set up an obj. for the best candidate
        best_candidate = {'solution':None, 'fitness':0}
        pbar = tqdm(range(n_iterations))

        for iteration in pbar:
            new_pop = {}
            for _ in range(len(old_population)//2): # to maintain the population size
                # Step 1: Selection
                select_method = np.random.choice(list(SelectionStore._SelectionStore__methods.keys()))
                selection = SelectionStore.get(method=select_method)
                selected = selection.select(population=old_population, size=2,
                                                n_participants=n_participants, params=kwargs)
                selected_indexes = [i for i in selected.keys()]

                # Step 2: Recombination
                change = np.random.choice([False, True], p=[1 - p_recombination, p_recombination]) 
                if change:
                    parent1 = selected[selected_indexes[0]].copy()
                    parent2 = selected[selected_indexes[1]].copy()
                    childOne, childTwo = recombination.recombine(parentOne=parent1, parentTwo=parent2)
                else:
                    childOne, childTwo = selected[selected_indexes[0]], selected[selected_indexes[1]]

                # Step 5: Mutation
                mutation_type =  np.random.choice(list(MutationStore._MutationStore__methods.keys())) 
                mutation = MutationStore.get(mutation_type)
                new_pop[len(new_pop)] = mutation.mutate(chromosome=childOne, p_mutation=p_mutation, 
                                                            max_generations=n_iterations, nth_Iteration=iteration, b=b).copy()
                new_pop[len(new_pop)] = mutation.mutate(chromosome=childTwo, p_mutation=p_mutation, 
                                                            max_generations=n_iterations, nth_Iteration=iteration, b=b).copy()
            
            # Step 6: Re-evaluate the population
            evaluated = self.__evaluation.evaluate(population=new_pop, data=self.__data, l=self.__l, transfer=self.__transfer,
                                                        inference=self.__inference, fitness_type=self.__fitness_type, 
                                                        normalization_type=self.__normalization_type, a=self.__a, p=self.__p)

            old_population = evaluated

            # Get the best candidate of the current generation
            _ = old_population[list(dict(sorted(old_population.items(), key=lambda k: k[1]['fitness'],reverse=True)).keys())[0]]
            # Update the best candidate if the best candidate of the current generation is better
            if _['fitness'] > best_candidate['fitness']:
                best_candidate = _
                pbar.set_postfix({'fitness': _['fitness']})

            # Step 7: Check termination
            if best_candidate['fitness'] >= threshold:
                print(f'RCGA identified a solution with fitness score <= {threshold}.')
                return best_candidate
        
        print(f'RCGA did not identify a solution with fitness score <= {threshold}. Returning a solution with the max fitness throughout the search process.')
        return best_candidate