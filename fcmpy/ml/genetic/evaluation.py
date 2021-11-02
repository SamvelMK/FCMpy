###########################################################################
##       Classes for evaluating the solutions of candidate solutions     ##
###########################################################################
import pandas as pd
from abc import ABC
from abc import abstractmethod
from fcmpy.ml.hebbian.update_state import FcmUpdate
from fcmpy.store.methodsStore import FitnessStore
from fcmpy.expert_fcm.input_validator import type_check


class Evaluation(ABC):
    """
        Interface for evaluation classes for RCGA.
    """
    @abstractmethod
    def evaluate(**kwargs):
        raise NotImplementedError('evaluate method is not defined.')


class PopulationEvaluation(Evaluation):
    """
        Evaluate the fitness of the population
    """
    @staticmethod
    @type_check
    def gen_data(state_vector:dict, weight_matrix:pd.DataFrame, transfer:str, 
                    inference:str, iterations:int, l:int, **kwargs) -> pd.DataFrame:
        """
            Update the state vector according to the selected inference method.

            Parameters
            ----------
            state_vector: dict
                            state vector of the concepts
                            keys ---> concepts, values ---> value of the associated concept

            weight_matrix: pd.DataFrame
                            N*N weight matrix of the FCM.

            transfer: str
                        transfer function to be applied on the updated concept values
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

            inference: str
                        inference method for the FCM update
                        inference method --> "kosko", "mKosko", "rescaled"
            
            l: Union[float, int]
                    slop parameter for the sigmoid transfer function
                    default --> 0.98

            Return
            ------
            y: pd.DataFrame
                simulated data
        """
        sim = FcmUpdate()
        state_vector=state_vector
        res = [state_vector]
        for _ in range(iterations):
            s = sim.update(state_vector=state_vector, weight_matrix=weight_matrix, 
                            transfer=transfer, inference=inference, iterations=iterations, 
                            l=l, params=kwargs)
            res.append(s)
            state_vector=s
            
        return pd.DataFrame(res)

    @staticmethod
    def evaluate(**kwargs)->dict:
        """
            Evaluate the fitness of the population

            Parameters
            ----------
            population: dict
                        population of candidate chromosomes
            
            data: pd.DataFrame
                    longitudinal data (measurements of the concepts)
            
            inference: str
                        inference method for the FCM update
                        inference method --> "kosko", "mKosko", "rescaled"
                        default -> 'mKosko'
            
            transfer: str
                        transfer function to be applied on the updated concept values
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"
                        default --> 'sigmoid'
            
            l: int
                slop parameter for the sigmoid transfer function
                default --> 1
            
            fitness_type: str
                            type of fitness function to use
                            default -> 'stach_fitness'

            normalization_type: str
                                type of normalization to apply onto the matrix error
                                normalization type --> "L1", "L2", "LInf"
                                default --> L2
        
            Return
            ------
            y: dict
                population with the calculated fitness scores of each candidate solution
        """
        population = kwargs['population'].copy()
        data = kwargs['data'].copy()
        transfer = kwargs['transfer']
        inference = kwargs['inference']
        fitness_type = kwargs['fitness_type']
        normalization_type = kwargs['normalization_type']
        l = kwargs['l']
        a = kwargs['a']
        p = kwargs['p']
        
        # The first row of the data is the initial state
        initial_state = data.iloc[0].to_dict()
        iterations = len(data) - 1 # t-1 simulation steps

        # Select fitness method
        fitness = FitnessStore.get(method=fitness_type)
        
        # Evaluate the fitness of every candidate solution in the population
        for i in population.keys():
            # simulate a data with a length len(data)-1 based on the candidate solution
            res = PopulationEvaluation.gen_data(state_vector=initial_state, weight_matrix=pd.DataFrame(population[i]['solution'],
                                                    columns=initial_state.keys()), iterations=iterations, transfer=transfer, 
                                                    inference=inference, l=l, params=kwargs)
            # Calculate the fitness of the candidate solution based on the selected fitness method
            fit = fitness.calculate_fitness(data_simulated=res, data=data, normalization_type=normalization_type, p=p, a=a, params=kwargs)
            population[i]['fitness'] = fit
            
        return population


class ChromosomeEvaluation(Evaluation):
    """
        Evaluate the fitness of a single chromosome.
    """
    @staticmethod
    def evaluate(**kwargs)->dict:
        """
            Evaluate the fitness of a single chromosome.

            Parameters
            ----------
            child: dict
                    child chromosome to be evaluated
            
            data: pd.DataFrame
                    longitudinal data (measurements of the concepts)
            
            inference: str
                        inference method for the FCM update
                        inference method --> "kosko", "mKosko", "rescaled"
                        default -> 'mKosko'
            
            transfer: str
                        transfer function to be applied on the updated concept values
                        transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"
                        default --> 'sigmoid'
            
            l: int
                slop parameter for the sigmoid transfer function
                default --> 1
            
            fitness_type: str
                            type of fitness function to use
                            default --> 'stach_fitness'

            normalization_type: str
                                type of normalization to apply onto the matrix error
                                normalization type --> "L1", "L2", "LInf"
                                default --> L2
        
            Return
            ------
            y: dict
                evaluated chromosome
        """
        child = kwargs['child'].copy()
        data = kwargs['data'].copy()
        transfer = kwargs['transfer']
        inference = kwargs['inference']
        fitness_type = kwargs['fitness_type']
        normalization_type = kwargs['normalization_type']
        l = kwargs['l']
        a = kwargs['a']
        p = kwargs['p']
        
        eval = PopulationEvaluation()
        # The first row of the data is the initial state
        initial_state = data.iloc[0].to_dict()
        iterations = len(data) - 1 # t-1 simulation steps

        # Select fitness method
        fitness = FitnessStore.get(method=fitness_type)

        res = eval.gen_data(state_vector=initial_state, weight_matrix=pd.DataFrame(child['solution'],
                                                    columns=initial_state.keys()), iterations=iterations, transfer=transfer, 
                                                    inference=inference, l=l, params=kwargs)
        
        fit = fitness.calculate_fitness(data_simulated=res, data=data, normalization_type=normalization_type, p=p, a=a, params=kwargs)
        return fit