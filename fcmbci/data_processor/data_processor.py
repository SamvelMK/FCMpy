import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pandas as pd
import itertools
import numpy as np
import skfuzzy as fuzz
import skfuzzy
import functools
import json
import collections
import re
from data_processor.checkers import Checker
from data_processor.fuzzy_inference import FuzzyInference
from data_processor.fuzzy_membership import FuzzyMembership

class DataProcessor(FuzzyInference, FuzzyMembership):
    """
    A class of methods to derive causal weights for FCMs based on linguistic terms.
    The FcmDataProcessor object is initialized with a universe of discourse with a range [-1, 1].
    
    Methods:
            __init__(self, linguistic_terms, data = None, check_consistency=False)
            __flatData(self, data)
            __activationParameter(self, flat_data, conceptPair)
            __entropy(self, data)
            add_membership_func(self, func)
            remove_membership_func(self, func_name)
            add_fuzzy_inference_func(self, func)
            remove_fuzzy_inference_func(self, func_name)
            read_xlsx(self, filepath, check_consistency=False)
            read_json(self, filepath, check_consistency=False)
            automf(self)
            activate(self, activation_input, mf)
            aggregate(self, activated)
            defuzzify(self, aggregated, method = 'centroid')
            gen_weights(self, method = 'centroid')
    """
    
    def __init__(self, linguistic_terms, no_causality='No-Causality', data = None, check_consistency=False):
        
        """
        The FcmDataProcessor object is initialized with a universe of discourse with a range [-1, 1].

        Parameters
        ----------
        linguistic_terms: list
                            Note that the number of linguistic terms should be even. A narrow interval around 0 (for no causality option) is added automatically.
        no_causality: str
                        name of the column that expresses no causality
                        default ---> 'No-Causality'
        data: ordered dict
                qualitative expert inputs.
                default --> None
        check_consistency: Bool
                            check the consistency of raitings across the experts.
                            default --> False
        """
        FuzzyInference.__init__(self)
        FuzzyMembership.__init__(self)
        
        self.linguistic_terms = [i.lower() for i in linguistic_terms]
        self.universe = np.arange(-1, 1.05, 0.05)
        # add a zero to the center of the universe of discourse to make it even (necessary for symetric dist of membership functions.)
        self.universe = np.insert(self.universe, len(self.universe[ : int(len(self.universe)/2)]), 0)
        self.__noCausality = no_causality.lower()

        if data != None:
            Checker.columns_check(data=data) # check if the from ---> to columns exist.
            if check_consistency:
                Checker.consistency_check(data=data, column_names = self.linguistic_terms) # check the consistency of the data.
                self.data = data
                
                # calculate the entropy of the expert raitings.
                self.entropy = self.__entropy(self.data)
        else:
            self.data = pd.DataFrame()

    def __flatData(self, data):
        """
        Create a flat data from an ordered dictionary.

        Parameters
        ----------
        data: dict,
                keys ---> expertId, values ---> pandasDf
        
        Return
        ---------
        y: pandas.DataFrame
            data with all the expert inputs in one dataframe.
        """
        # Create a flat data with all of the experts' inputs.
        flat_data = pd.concat([data[i] for i in data], sort = False)
        flat_data.columns = [i.lower() for i in flat_data.columns]
        flat_data = flat_data.set_index(['from', 'to'])
        flat_data = flat_data.sort_index(level=['from','to'])

        return flat_data

    def __concept_parser(self, string, sepConcept):
        """
        Parse the csv file column names. Extract the antecedent, concequent pairs and the polarity of the causal relationship.

        Parameters
        ----------
        string: str
                the column names that need to be parsed
        
        sepConcept: str
                    the separation symbol (e.g., '->') that separates the antecedent from the concequent in the columns of a csv file
        
        Return
        ---------
        y: dict
            keys --> antecedent, concequent, polarity

        """
        dt = {}
        pattern = f'[a-zA-Z]+.[a-zA-Z]+.{sepConcept}.+.(\(\+\)|\(\-\))'
        patterMatch = bool(re.search(pattern, string))
        
        if patterMatch:
            dt['polarity'] = re.search(r'\((.*?)\)', string).group(1)
            concepts = string.split(sepConcept)
            dt['antecedent'] = re.sub(r'\([^)]*\)', '', concepts[0]).strip() 
            dt['concequent'] = re.sub(r'\([^)]*\)', '', concepts[1]).strip()
            return dt
        else:
            raise ValueError('The $antecedent$ $->$ $concequent (sign)$ format is not detected! Check the data format!') 

    def __extractExpertData(self, data, sepConcept, linguistic_terms, noCausality):
        """
        Convert csv data fromat to a dataframe with columns representing the linguistic terms (see more in the doc.).

        Parameters
        ----------        
        sepConcept: str
                    the separation symbol (e.g., '->') that separates the antecedent from the concequent in the columns of a csv file
        
        linguistic_terms: list
                            list of linguistic terms
        
        noCausality: str,
                        the term used to express noCausality

        Return
        ---------
        y: pandas.DataFrame
        """

        dict_data = []
        for i in data.keys():
            _ = {i: 0 for i in linguistic_terms}
            concepts = self.__concept_parser(i, sepConcept=sepConcept)
            _['From'] = concepts['antecedent']
            _['To'] = concepts['concequent']

            if data[i].lower() in self.linguistic_terms:
                # no causality cases
                if data[i].lower() == noCausality:
                    _[data[i].lower()] = 1
                else:        
                    if concepts['polarity'] == '+':
                        _[data[i].lower()] = 1
                    else:
                        _[str('-'+data[i].lower())] = 1
                dict_data.append(_)
        return pd.DataFrame(dict_data)

    def __activationParameter(self, flat_data, conceptPair):
        """
        Create an activation parameter based on the expert inputs.

        Parameters
        ----------
        flat_data: pandas.DataFrame
                    flat data file
        
        conceptPair: tuple,
                        concept pair for which the activation parameter should be constructed.

        Return
        ---------
        y: dict,
            keys ---> linguistic terms, values ---> proportion of expert raitings.
        """
        
        activation_parameter = {}
        activation_parameter = (flat_data.loc[conceptPair].sum()/len(self.data)).to_dict()
        return activation_parameter

    def __entropy(self, data):
        """
        Calculate the entropy of the expert raitings.

        Parameters
        ----------
        data: collections.OrderedDict
                qualitative expert inputs.
        
        Return
        ---------
        y: pandas.DataFrame,
            entropy of the concept pairs in expert raitings.
        """

        flat_data = self.__flatData(data)
        prop = {}
        for concepts in set(flat_data.index):
            activation_parameter = self.__activationParameter(flat_data, concepts) 
            prop[concepts] = activation_parameter

        entropy_concept = {}
        for concept in prop.keys():
            p = prop[concept].values()
            res = -sum([i*np.log2(i) for i in p if i != 0])
            if res == 0: # to avoide -0 reports.
                res = abs(res)
            entropy_concept[concept] = res
        
        # Prepare a formated dataframe
        entropy_concept = {k:[v] for k,v in entropy_concept.items()}
        entropy_concept = pd.DataFrame(entropy_concept).T
        entropy_concept.index.set_names(['From','To'], inplace=True)
        entropy_concept.columns = ['Entropy']
        entropy_concept = entropy_concept.sort_index(level=[0,1])

        return entropy_concept

    #### Read data            

    def read_xlsx(self, filepath, check_consistency=False):
        
        """ 
        Read data from an excel spreadsheet.
        
        Parameters
        ----------
        filepath : str, 
                    ExcelFile, xlrd.Book, path object or file-like object (read more in pd.read_excel)

        check_consistency: Bool
                            check the consistency of raitings across the experts.
                            default --> False
        """
        column_names = [i.lower() for i in self.linguistic_terms]
        data = pd.read_excel(filepath, sheet_name=None)

        # check the data
        Checker.columns_check(data=data) # check if From ---> To columns exist: raise error if otherwise.
        if check_consistency:
            Checker.consistency_check(data=data, column_names = column_names) # Checks whether the sign of the raitings across the experts are consistent.
        self.data = collections.OrderedDict(data)
        
        # calculate the entropy of the expert raitings.
        self.entropy = self.__entropy(self.data)           

    def read_json(self, filepath, check_consistency=False):
        """ 
        Read data from a json file.

        Parameters
        ----------
        filepath : str, path object or file-like object

        check_consistency: Bool
                            check the consistency of raitings across the experts.
                            default --> False
        """
        column_names = [i.lower() for i in self.linguistic_terms]
        f = open(filepath) 
        data = json.load(f)
        f.close()
        d = {}
        for i in data.keys():
            d[i] = data[i]
        od = collections.OrderedDict([(i, pd.DataFrame(d[i])) for i in d])
        # check the data
        Checker.columns_check(data=od)
        if check_consistency:
            Checker.consistency_check(data=od, column_names=column_names)
        self.data = od
        
        # calculate the entropy of the expert raitings.
        self.entropy = self.__entropy(self.data)

    def read_csv(self, filePath, sepConcept, csv_sep=',', check_consistency=False):
        """ 
        Read data from a csv file.

        Parameters
        ----------
        filepath : str, path object or file-like object

        sepConcept: str
                    the separation symbol (e.g., '->') that separates the antecedent from the concequent in the columns of a csv file
        
        linguistic_terms: list
                            list of linguistic terms
        
        noCausality: str,
                        the term used to express noCausality
        
        csv_sep: str,
                    separator of the csv file (read more in pandas.read_csv)

        check_consistency: Bool
                            check the consistency of raitings across the experts.
                            default --> False
        """
        data = pd.read_csv(filePath, sep=csv_sep)
        dataOd = collections.OrderedDict()
        for i in range(len(data)):
            _ = data.iloc[i].to_dict()

            expertData = self.__extractExpertData(data=_, sepConcept=sepConcept, linguistic_terms=self.linguistic_terms, noCausality=self.__noCausality)
            dataOd[f'Expert{i}'] = expertData
        
        self.data = dataOd

    #### Obtain (numerical) causal weights based on expert (linguistic) inputs.

    def automf(self, membership_function = 'trimf', **params):
        
        """ 
        Automatically generate membership functions based on the passed linguistic terms (in the init).
        This functions were taken and modified from scikit-fuzzy.
        
        Parameters
        ----------
        membership_function: str,
                                fuzzy membership function. --> "trimf" 

        Return
        ---------
        y: dict,
            Generated membership functions. The keys are the linguistic terms and the values are 1d arrays. 
        """
        np.set_printoptions(suppress=True) # not necessary (easier for debug.)

        mf = self.membership_func[membership_function]

        terms = mf(universe = self.universe, linguistic_terms=self.linguistic_terms, noCausality=self.__noCausality)
        
        return terms
        
    def activate(self, mf, activation_input, fuzzy_inference="mamdaniProduct", **params):
        
        """ 
        Activate the specified membership function based on the passed parameters (Mamdani).
        
        Parameters
        ----------
        activation_input: dict,
                            Membership function to apply the implication operation, 
                            where the key is the linguistic term and the value is the frequency of its occurrence
                            Example: parameters = {'H': 0.66, 'VH': 0.33}
        mf: dict,
            membership functions upon which the implication operator is applied. The key in the dict is the linguistic term, 
            and the value is a 1d array with the membership values
        
        fuzzy_inference: str,
                            fuzzy inference method. --> "mamdaniMin", "mamdaniProduct"
        
        Return
        ---------
        y: dict,
            activated membership functions, where the key is the linguistic term and 
            the value is a 1d array with the activated membership values. 
        """

        infer = self.fuzzy_inference_funcs[fuzzy_inference]

        activation_input = {k.lower(): v for k, v in activation_input.items()} # Make lower case.
        activated = {}
          
        for i in activation_input.keys():
            activated[i] = infer(mf_x=mf[i], weight=activation_input[i], **params)
        return activated
    
    def aggregate(self, activated):
        
        """ 
        Aggregate the activated membership function using fmax operator. 
        
        Parameters
        ----------
        activated: dict,
                    a dictionary with the activated membership values to be aggregated
        
        Return
        ---------
        y : 1d array,
            Aggregated membership function.
        """
        
        aggregated = functools.reduce(lambda x,y: np.fmax(x,y),
                             [activated[i] for i in activated.keys()])
        
        return aggregated
    
    def defuzzify(self, aggregated, method = 'centroid'):
        
        """ 
        Difuzzify the aggregated membership functions using centroid defuzzification method as a default.
        One can pass on another defuzzification method available in scikit-fuzzy library (e.g., bisector, mom, sig)
        Returns the defuzzified value.

        Parameters
        ----------
        aggregated: 1d array,
                        Aggregated membership function to be defuzzified.
        method: str, 
                    Defuzzification method, default --> 'centroid'. 
                    For other defuzzification methods check scikit-fuzzy library (e.g., bisector, mom, sig)

        Return
        ---------
        y : int,
            Defuzzified value.
        """
        
        defuzzified_value = fuzz.defuzz(self.universe, aggregated, method)
        
        return defuzzified_value           
        
    def gen_weights(self, method = 'centroid', membership_function='trimf', fuzzy_inference="mamdaniProduct", **params): 
        
        """ 
        Apply fuzzy logic to obtain edge weights of an FCM with qualitative inputs 
        (i.e., where the causal relationships are expressed in linguistic terms).

        method: str,
                    Defuzzification method;  default --> 'centroid'. 
                    For other defuzzification methods check scikit-fuzzy library (e.g., bisector, mom, sig)
        
        fuzzy_inference: str,
                            fuzzy inference method. --> "mamdaniMin", "mamdaniProduct"                    
        """

        # A dict to store the aggregated results for the visualization purposes. 
        self.aggregated = {}
        flat_data = self.__flatData(self.data)
        
        # weight matrix for the final results.
        cols = set([i[0] for i in set(flat_data.index)])
        weight_matrix = pd.DataFrame(columns=cols, index=cols)
        
        # Create the membership functions for the linguistic terms.
        self.terms_mf = self.automf(membership_function=membership_function, **params)
        
        for concepts in set(flat_data.index):
            activation_parameter = self.__activationParameter(flat_data, concepts)
            activated = self.activate(mf=self.terms_mf, activation_input=activation_parameter, fuzzy_inference=fuzzy_inference, **params)
            if not all(x==0 for x in activation_parameter.values()):
                aggr = self.aggregate(activated)
                self.aggregated[f'{concepts}'] = aggr
                value = self.defuzzify(aggr, method)
                weight_matrix.loc[concepts] = value
        self.causal_weights = weight_matrix.fillna(0)