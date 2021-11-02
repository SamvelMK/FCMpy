###########################################################################
##                       RCGA fro FCMs                                   ##
###########################################################################
import pandas as pd
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.ml.genetic.ga_interface import GA
from fcmpy.store.methodsStore import GetMethod
from fcmpy.ml.genetic.generational_ga import GRCGA
from fcmpy.ml.genetic.ssga import SSGA


# Factory of genetic algorithms
class RcgaStore(GetMethod):
    """
        RCGA algorithms for learning FCMs.
    """
    __methods = {'generational':GRCGA, 'ssga' : SSGA}

    @staticmethod
    @type_check
    def get(method:str):
        """
            Get the respective RCGA method based on user input.
        """
        if method in RcgaStore.__methods.keys():
            return RcgaStore.__methods[method]
        else:
            raise ValueError('The RCGA type is not defined.')


class RCGA(GA):
    """
        Real Coded Genetic Algorithm (RCGA) for learning FCM connection matrices.

        Parameters
        ----------
        data: pd.DataFrame

        ga_type: str
                    type of genetic algorithm --> "generational", "ssga"
                    default --> "generational"
        
        init_type: str
                    type of initialization
                    default --> "uniform"
        
        population_size: int
                            number of chromosome to be generated in the population
                            default --> 100
        
        fitness_type: str
                        fitness function
                        default -> "stach_fitness"
        
        normalization_type: str
                            type of normalization to apply onto the matrix error
                            normalization type --> "L1", "L2", "LInf"
                            default --> L2
        
        a: int
            parameter for the fitness function
            default -> 100
        
        p: int
            parameter for the fitness function
            default --> 2

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
    def __init__(self, data:pd.DataFrame, ga_type:str='generational', init_type:str='uniform', population_size:int=100,
                    fitness_type:str='stach_fitness', normalization_type:str='L2', a:int=100, p:int=2,
                    inference:str='mKosko', transfer:str='sigmoid', l:int=1, **kwargs):
                
        self.__init_type = init_type
        self.__popSize = population_size
        self.__data = data
        self.__fitness_type = fitness_type
        self.__normalization_type = normalization_type
        self.__a = a
        self.__p = p
        self.__inference = inference
        self.__transfer = transfer
        self.__l=l
        self.__rcga = RcgaStore.get(ga_type)
        self.__kwargs = kwargs
        self.fitness = None
        self.solution = None

    def run(self, n_iterations:int=100000, n_participants:int=5, recombination_type:str='one_point_crossover',
                p_recombination:float=0.9, p_mutation:float=0.5, b:int=5, threshold:float=0.9, **params): 
        """
            Run the RCGA learning algorithm for learning FCM connection matrices

            Parameters
            ----------
            n_iterations: int
                            number of iterations to run
                            default -> 100000
            
            n_participants: int
                            number of participants in the selection process
            
            recombination_type: str
                                recombination type to use
                                default -> "one_point_crossover"
            
            p_recombination: float
                                probability of recombination
                                default -> 0.9
            
            p_mutation: float
                        probability of mutation
                        default -> 0.5
            
            b: int
                parameter for the fitness function
                default -> 5
            
            threshold: float
                        threshold of acceptable fitness
            
            Return
            ------
            y: pd.DataFrame
                optimized connection matrix
        """
        rcga = self.__rcga(data=self.__data, init_type=self.__init_type, population_size=self.__popSize,
                            fitness_type=self.__fitness_type, normalization_type=self.__normalization_type, a=self.__a, p=self.__p,
                            inference=self.__inference, transfer=self.__transfer, l=self.__l, kwargs=self.__kwargs)

        res = rcga.run(n_iterations=n_iterations, n_participants=n_participants, recombination_type=recombination_type,
                            p_recombination=p_recombination, p_mutation=p_mutation, b=b, threshold=threshold, params=params)  

        df = pd.DataFrame(res['solution'], columns=self.__data.keys(), index=self.__data.keys())
        
        self.fitness = res['fitness']
        self.solution = df
