###########################################################################
##                    Implementation of the Steady State GA              ##
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
from fcmpy.store.methodsStore import ReplacementStore
from fcmpy.ml.genetic.evaluation import ChromosomeEvaluation
from fcmpy.ml.genetic.evaluation import PopulationEvaluation
import copy


class SSGA(GA):
    """
        Steady State Real-Coded Algorithm for learning FCM connection matrices

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
                            type of normalization to apply onto the matrix error
                            normalization type --> "L1", "L2", "LInf"
                            default --> L2
        a: int
            coefficient for the auxiliary function
            default -> 100

        p: int
            coefficient for the matrix error
            default -> 2 (for L2 norm.)
        
        inference: str
                    inference method for the FCM update
                    inference method --> "kosko", "mKosko", "rescaled"
                    default -> 'mKosko'
        
        transfer: str
                    transfer function to be applied on the updated concept values
                    transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"
                    default --> "sigmoid"
        
        l: int
            slope parameter for the sigmoid transfer function
            default --> 1
    """
    @type_check
    def __init__(self, data:pd.DataFrame, init_type:str='uniform', population_size:int=100,
                    fitness_type:str='stach_fitness', normalization_type:str='L2', a:int=100, p:int=2,
                    inference:str='mKosko', transfer:str='sigmoid', l:int=1, **kwargs):
        
        self.__initialize = InitializationStore.get(init_type)
        self.__popSize = population_size
        self.__nConcepts = len(data.keys())
        self.__data = data
        __evaluation = PopulationEvaluation()
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
        self.__population = __evaluation.evaluate(population=population, data=self.__data, l=self.__l, 
                                                            transfer=self.__transfer, inference=self.__inference,
                                                            fitness_type=self.__fitness_type, normalization_type=self.__normalization_type,
                                                            a=self.__a, p=self.__p)
    
    @type_check
    def run(self, n_iterations:int=100000, n_participants:int=5, recombination_type:str='one_point_crossover', replacement_strategy='CRDW',
                p_recombination:float=0.9, p_mutation:float=0.5, b:int=5, threshold:float=0.9, **kwargs):
        
        recombination = RecombinationStore.get(recombination_type)
        replacement = ReplacementStore.get(replacement_strategy)
        ev = ChromosomeEvaluation()
        old_population = copy.deepcopy(self.__population)
        best_candidate = {'solution': None, 'fitness':0}
        pbar = tqdm(range(n_iterations))
        
        for iteration in pbar:
            # Step 1: Selection
            select_method = np.random.choice(list(SelectionStore._SelectionStore__methods.keys()))
            selection = SelectionStore.get(method=select_method)
            selected = selection.select(population=old_population, size=2,
                                            n_participants=n_participants, params=kwargs)
            selected_indexes = [i for i in selected.keys()]

            # Step 2: Recombination
            change = np.random.choice([False, True], p=[1 - p_recombination, p_recombination]) 
            if change:
                parent1 = selected[selected_indexes[0]]
                parent2 = selected[selected_indexes[1]]
                childOne, childTwo = recombination.recombine(parentOne=copy.deepcopy(parent1), parentTwo=copy.deepcopy(parent2))
            else:
                childOne, childTwo = selected[selected_indexes[0]], selected[selected_indexes[1]]

            # Step 5: Mutation
            mutation_type =  np.random.choice(list(MutationStore._MutationStore__methods.keys())) 
            mutation = MutationStore.get(mutation_type)
            childOne = mutation.mutate(chromosome=copy.deepcopy(childOne), p_mutation=p_mutation, 
                                                        max_generations=n_iterations, nth_Iteration=iteration, b=b)
            childTwo = mutation.mutate(chromosome=copy.deepcopy(childTwo), p_mutation=p_mutation, 
                                                        max_generations=n_iterations, nth_Iteration=iteration, b=b)
            # Step 6: Evaluate the fitness of each solution
            childOne['fitness'] = ev.evaluate(child=childOne, data=self.__data, l=self.__l, 
                                            transfer=self.__transfer, inference=self.__inference,
                                            fitness_type=self.__fitness_type, normalization_type=self.__normalization_type,
                                            a=self.__a, p=self.__p).copy()
            childTwo['fitness'] = ev.evaluate(child=childTwo, data=self.__data, l=self.__l, 
                                            transfer=self.__transfer, inference=self.__inference,
                                            fitness_type=self.__fitness_type, normalization_type=self.__normalization_type,
                                            a=self.__a, p=self.__p).copy()
            # Step 7: Replace
            old_population = replacement.replace(child=copy.deepcopy(childOne), population=copy.deepcopy(old_population))
            old_population = replacement.replace(child=copy.deepcopy(childTwo), population=copy.deepcopy(old_population))
            
            # Get the best candidate of the current generation
            _ = copy.deepcopy(old_population[list(dict(sorted(old_population.items(), key=lambda k: k[1]['fitness'],reverse=True)).keys())[0]])
            # Update the best candidate if the best candidate of the current generation is better
            if _['fitness'] > best_candidate['fitness']:
                pbar.set_postfix({'fitness': _['fitness']})
            best_candidate = _
            # Step 8: Check termination
            if best_candidate['fitness'] >= threshold:
                print(f'RCGA identified a solution with fitness score <= {threshold}.')
                return best_candidate

        print(f'RCGA did not identify a solution with fitness score <= {threshold}. Returning a solution with the max fitness of the last generation.')
        return  best_candidate