import numpy as np 
import pandas as pd
import functools
import collections
from abc import ABC, abstractmethod
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.store.methodsStore import EntropyStore
from fcmpy.store.methodsStore import ReaderStore
from fcmpy.store.methodsStore import MembershipStore
from fcmpy.store.methodsStore import ImplicationStore
from fcmpy.store.methodsStore import AggregationStore
from fcmpy.store.methodsStore import DefuzzStore
from fcmpy.expert_fcm.transform import Transform


class FcmConstructor(ABC):
    @abstractmethod
    def read_data(file_path, **kwargs) -> collections.OrderedDict:
        raise NotImplementedError('read_data method is not defined!')

    @abstractmethod
    def build(data: collections.OrderedDict, implication_method:str, 
                    aggregation_method:str, defuzz_method:str) -> pd.DataFrame:
                    
        raise NotImplementedError('build method is not defined!')


class ExpertFcm(FcmConstructor):
    """
        Construct Expert FCMs based on qualitative input data.

        Methods:
            read_data(file_path, **kwargs)
            entropy(data: collections.OrderedDict, method = 'entropy', **kwargs)
            automf(method:str='trimf', **kwargs)
            fuzzy_implication(membership_function, weight, method:str='Mamdani', **kwargs)
            aggregate(x, y, method:str='fMax', **kwargs)
            defuzz(x, mfx, method:str='centroid', **kwargs)
            build(data: collections.OrderedDict, implication_method:str='Mamdani', 
                            aggregation_method:str='fMax', defuzz_method:str='centroid')
    """
    @type_check
    def __init__(self):
        self.__linguisticTerms = None
        self.__membership = None
        self.__universe = None

    @property
    def linguistic_terms(self):
        return self.__linguisticTerms

    @linguistic_terms.setter
    @type_check
    def linguistic_terms(self, terms: dict):
        """
            Linguistic terms and the associated parameters for generating fuzzy membership functions.

            Parameters
            ----------
            terms : dict,
                        keys are the linguistic terms and the values are 
                        lists with term parameters (see blow)

            Term parameters
            ----------------
            for trimf:
                the parameters 'abc' should be passed as keys (in a list)
                to the linguistic terms -> e.g., {'+VL': [0.25, 0.5, 0.75]}
            
            for gaussmf:
                the parameters 'mean' and 'sigma' should be passed as keys (in a list)
                to the linguistic terms -> e.g., {'+VL': [0.25, 0.1]}
            
            for trapmf:
                the parameters 'abcd' should be passed as keys to the linguistic terms
                e.g., {'+VL': [0, 0.25, 0.5, 0.75]}
        """
        termsLower = dict((k.lower(), v) for k, v in terms.items()) 
        self.__linguisticTerms = termsLower

    @property
    def fuzzy_membership(self):
        return self.__membership
    
    @fuzzy_membership.setter
    @type_check
    def fuzzy_membership(self, membership:dict):   
        """
            Parameters
            ----------
            membership : dict,
                            keys are the linguistic terms and the values are the associated
                            numpy arrays with membership values.
        """

        self.__membership = membership

    @property
    def universe(self):
        return self.__universe
    
    @universe.setter
    @type_check
    def universe(self, universe:np.ndarray):
        
        """
        Parameters
        ----------
        universe: np.ndarray
                    the universe of discourse. 
                    (Note that the generated arrays are automatically rounded to 2 digits)
        """

        self.__universe = universe.round(2)

    @type_check
    def read_data(self, file_path:str, **kwargs) -> collections.OrderedDict:
        
        """ 
            Read data from a file. Currently, the method supports 
            .csv, .xlsx and .json file formats.
            
            Parameters
            ----------
            file_path : str 
            
            Other Parameters
            ----------------
            for .csv files:

                **sep_concept: str,
                                separation symbol (e.g., '->') that separates the antecedent 
                                from the consequent in the column heads of a csv file
                                default ---> '->'

                **csv_sep: str,
                            separator of the csv file
                            default ---> ','
                            
            for .xlsx files:

                **check_consistency: Bool
                                    check the consistency of raitings across the experts
                                    default --> False

                **engine: str,
                            the engine for excel reader (read more in pd.read_excel)
                            default --> "openpyxl"

            for .json files:

                **check_consistency: Bool
                                    check the consistency of raitings across the experts.
                                    default --> False
            
            Return
            -------
            data: collections.OrderedDict
                    ordered dictionary with the formatted data.
        """

        if self.linguistic_terms is None:
            raise ValueError('Linguistic terms are not defined!')

        fileType = file_path.split('.')[-1]
        reader = ReaderStore.get(fileType)()
        data = reader.read(filePath=file_path, linguisticTerms = self.linguistic_terms,
                            params = kwargs)
        
        return data

    @type_check
    def entropy(self, data: collections.OrderedDict, 
                method:str = 'entropy', **kwargs) -> pd.DataFrame:
        """
            Calculate the entropy of the expert ratings.

            Parameters
            ----------
            data: collections.OrderedDict
                ordered dictionary with the expert inputs
            
            method: str
                    method for calculating entropy. At the moment only information entropy is available
                    default --> 'entropy'

            Return
            -------
            y: pandas.DataFrame,
                entropy of the concept pairs in expert ratings
        """

        entropy = EntropyStore.get(method)()
        return entropy.calculateEntropy(data=data, params=kwargs)
    
    @type_check
    def automf(self, method:str='trimf', **kwargs) -> dict:
        """
            Generate membership functions for a given set of linguistic terms.

            Parameters
            ----------
            method: str
                    type of membership function. At the moment three such types are available  
                    'trimf': triangular membership functions
                    'gaussmf': gaussian membership functions
                    'trapmf': trapezoidal membership functions
                    default ---> 'trimf'
            
            Return
            -------
            y: dict
                generated membership functions. The keys are the linguistic
                terms and the values are 1d arrays
        """
        if self.linguistic_terms is None:
            raise ValueError('Linguistic terms are not defined!')
        elif self.universe is None:
            raise ValueError('Universe of discourse is not defined!')
        
        membership = MembershipStore.get(method)()

        return membership.membershipFunction(linguistic_terms=self.linguistic_terms, 
                                            universe=self.universe, params = kwargs)
    
    @type_check
    def fuzzy_implication(self, membership_function, weight, 
                            method:str='Mamdani', **kwargs) -> np.ndarray:
        """
            Fuzzy implication rule.

            Parameters
            ----------
            membership_function: numpy.ndarray,
                                    membership function of a linguistic term (x)

            weight: float,
                        the weight at which the given membership function x should be "activated" 
                        (i.e., the cut point or the point at which the membership function should be rescaled)
            
            method: str,
                    implication rule; at the moment two such rules are available 
                    'Mamdani': minimum implication rule
                    'Larsen': product implication rule
                    default ---> 'Mamdani'

            Return
            -------
            y: numpy.ndarray
                the "activated" membership function
        """
        implication = ImplicationStore.get(method)()

        return implication.implication(mf_x = membership_function, 
                                        weight = weight, params = kwargs)

    @type_check
    def aggregate(self, x, y, method:str='fMax', **kwargs) -> np.ndarray:
        """
            Fuzzy aggregation rule.

            Parameters
            ----------
            x, y: numpy.ndarray,
                        "activated" membership functions of the linguistic 
                        terms that need to be aggregated
            
            method: str,
                    aggregation rule; at the moment four such rules are available
                    'fMax': family maximum,
                    'algSum': family Algebraic Sum,
                    'eSum': family Einstein Sum,
                    'hSum': family Hamacher Sum
                    default ---> 'fMax'

            Return
            -------
            y: numpy.ndarray
                an aggregated membership function
        """

        aggr = AggregationStore.get(method)()
        aggregated = aggr.aggregate(x=x, y=y, params = kwargs)
        return aggregated

    @type_check
    def defuzz(self, x, mfx, method:str='centroid', **kwargs) -> float:
        """
            Defuzzification of the aggregated membership functions.

            Parameters
            ----------
            x: numpy.ndarray
                universe of discourse 
            
            mfx: numpy.ndarray,
                        "aggregated" membership functions
            
            method: str,
                    defuzzification method; at the moment four such 
                    rules are available:
                        'centroid': Centroid,
                        'bisector': Bisector,
                        'mom': MeanOfMax,
                        'som': MinOfMax,
                        'lom' : MaxOfMax
                        default ---> 'centroid'

            Return
            -------
            y: float
                defuzzified value
        """

        defuzz = DefuzzStore.get(method)()
        defuzz_value = defuzz.defuzz(x=x, mfx=mfx, method=method, params=kwargs)

        return defuzz_value

    @type_check
    def build(self, data: collections.OrderedDict, implication_method:str='Mamdani', 
                    aggregation_method:str='fMax', defuzz_method:str='centroid') -> pd.DataFrame:
        """
            Build an FCM based on qualitative input data.

            Parameters
            ----------
            data: collections.OrderedDict
                    ordered dictionary with the qualitative input data.
            
            implication_method: str,
                                implication rule; at the moment two such
                                rules are available;
                                    'Mamdani': minimum implication rule
                                    'Larsen': product implication rule
                                    default ---> 'Mamdani'

            aggregation_method: str,
                                aggregation rule; at the moment four such
                                rules are available:
                                    'fMax': family maximum,
                                    'algSum': family Algebraic Sum,
                                    'eSum': family Einstein Sum,
                                    'hSum': family Hamacher Sum
                                    default ---> 'fMax'

            defuzz_method: str,            
                            defuzzification method; at the moment four such
                            rules are available:
                                'centroid': Centroid,
                                'bisector': Bisector,
                                'mom': MeanOfMax,
                                'som': MinOfMax,
                                'lom' : MaxOfMax
                                default ---> 'centroid'

            Return
            -------
            y: pd.DataFrame
                the connection matrix with the defuzzified values            
        """
        if self.fuzzy_membership is None:
            raise ValueError('Membership function is not defined!')

        if self.linguistic_terms is None:
            raise ValueError('linguistic_terms are not defined!')

        if self.universe is None:
            raise ValueError('Universe of discourse is not defined!')

        nExperts = len(data)

        # Drop the columns that should be omitted from the calculations (e.g., unsure)
        keep = [i.lower() for i in list(self.linguistic_terms.keys())]
        flat_data = Transform.flatData(data)
        flat_data = flat_data[keep]
        
        # Create an empty weight matrix
        cols = set(flat_data.index.get_level_values('to'))
        index = set(flat_data.index.get_level_values('from'))
        index = sorted(index.union(cols))
        weight_matrix = pd.DataFrame(0.0,columns=index, index=index, dtype=float)
        # main part for calculating the weights
        for concepts in set(flat_data.index):
            # for a given pair of concepts calculate the propostions (weights) for the
            # implication rules.
            activation_parameter = Transform.calculateProportions(data=flat_data, 
                                            conceptPair=concepts, nExperts=nExperts)
            activated = {}
            # for each linguistic term apply the implication rule
            for term in self.fuzzy_membership.keys():
                act = self.fuzzy_implication(membership_function=self.fuzzy_membership[term], 
                                            weight=activation_parameter[term], method=implication_method)
                activated[term] = act
            
            # if the 'activated' membership functions are not all zeros then aggregate 
            # them and defuzzify them.
            if not all(x==0 for x in activation_parameter.values()):
                # aggregate all the activated membership functions
                aggregated = functools.reduce(lambda x,y: self.aggregate(x=x, y=y, method=aggregation_method),
                                                [activated[i] for i in activated.keys()])

                # defuzzify the aggregated functions                                
                value = self.defuzz(x=self.universe, mfx=aggregated, method=defuzz_method)
                # populate the empty weigtht_matrix with the defuzzified value
                weight_matrix.loc[concepts] = value
        
        weight_matrix = weight_matrix.fillna(0.0)

        return weight_matrix