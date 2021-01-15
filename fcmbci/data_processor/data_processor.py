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
from data_processor.checkers import Checker

class FcmDataProcessor:
    """
    A class of methods to derive causal weights for FCMs based on linguistic terms.
    The FcmDataProcessor object is initialized with a universe of discourse with a range [-1, 1].
    
    Methods:
            __init__(self, linguistic_terms, data = None, check_consistency=False)
            read_xlsx(self, filepath, check_consistency=False)
            read_json(self, filepath, check_consistency=False)
            automf(self)
            activate(self, activation_input, mf)
            aggregate(self, activated)
            defuzzify(self, aggregated, method = 'centroid')
            gen_weights(self, method = 'centroid')
    """
    
    def __init__(self, linguistic_terms, data = None, check_consistency=False):
        
        """
        The FcmDataProcessor object is initialized with a universe of discourse with a range [-1, 1].

        Parameters
        ----------
        linguistic_terms: list
                            Note that the number of linguistic terms should be even. A narrow interval around 0 (for no causality option) is added automatically.
        data: ordered dict
                qualitative expert inputs.
                default --> None
        check_consistency: Bool
                            check the consistency of raitings across the experts.
                            default --> False
        """
        self.linguistic_terms = [i.lower() for i in linguistic_terms]
        self.universe = np.arange(-1, 1.001, 0.001)
        
        if data != None:
            Checker.columns_check(data=data) # check if the from ---> to columns exist.
            if check_consistency:
                Checker.consistency_check(data=data, column_names = self.linguistic_terms) # check the consistency of the data.
                self.data = data
        else:
            self.data = pd.DataFrame()

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

    def read_json(self, filepath, check_consistency=False):
        """ 
        Read data from a json file.

        Parameters
        ----------
        filepath : str, path object or file-like object
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

    #### Obtain (numerical) causal weights based on expert (linguistic) inputs.
    
    def automf(self):
        
        """ 
        Automatically generate triangular membership functions based on the passed linguistic terms (in the init).
        This function was taken and modified from scikit-fuzzy.
        
        Return
        ---------
        y: dict,
            Generated membership functions. The keys are the linguistic terms and the values are 1d arrays. 
        """
        
        number = len(self.linguistic_terms)
        limits = [self.universe.min(), self.universe.max()]
        universe_range = (limits[1] - limits[0])/2
        widths = [universe_range / (((number/2) - 1) / 2.)] * int(number)
        
        
        # Create the centers of the mfs for each side of the x axis and then merge them together.
        centers_pos = np.linspace(0.001, 1, number//2)
        centers_neg = np.linspace(-1, -0.001, number//2)
        centers = list(centers_neg)+list(centers_pos)
        
        abcs = [[c - w / 2, c, c + w / 2] for c, w in zip(centers, widths)]
        
        abcs[number//2] = [0.001, 0.001, centers_pos[1]] # + Very low 
        abcs[((number//2) -1)] = [centers_neg[-2], -0.001, -0.001] # - Very Low
        
        terms = dict()

        # add a narrow interval for no causality.
        self.linguistic_terms.insert(len(self.linguistic_terms)//2, 'na')
        abcs.insert(len(abcs)//2, np.array([-0.001, 0, 0.001]))

        # Repopulate
        for term, abc in zip(self.linguistic_terms, abcs):
            terms[term] = skfuzzy.trimf(self.universe, abc)
        
        return terms
        
    def activate(self, activation_input, mf):
        
        """ 
        Activate the specified membership function based on the passed parameters.
        
        Parameters
        ----------
        activation_input: dict,
                            Membership function to apply the implication operation, 
                            where the key is the linguistic term and the value is the frequency of its occurrence
                            Example: parameters = {'H': 0.66, 'VH': 0.33}
        mf: dict,
            membership functions upon which the implication operator is applied. The key in the dict is the linguistic term, 
            and the value is a 1d array with the membership values
        
        Return
        ---------
        y: dict,
            activated membership functions, where the key is the linguistic term and 
            the value is a 1d array with the activated membership values. 
        """
        activation_input = {k.lower(): v for k, v in activation_input.items()} # Make lower case.

        activated = {}
        for i in activation_input.keys():
            activated[i] = np.fmin(activation_input[i], mf[i])
        
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
    
    def gen_weights(self, method = 'centroid'): 
        
        """ 
        Apply fuzzy logic to obtain edge weights of an FCM with qualitative inputs 
        (i.e., where the causal relationships are expressed in linguistic terms).

        method: str,
                    Defuzzification method;  default --> 'centroid'. 
                    For other defuzzification methods check scikit-fuzzy library (e.g., bisector, mom, sig)
                    
        """        
        # A dict to store the aggregated results for the visualization purposes. 
        self.aggregated = {}

        # Create a flat data with all of the experts' inputs.
        flat_data = pd.concat([self.data[i] for i in self.data], sort = False)
        flat_data.columns = [i.lower() for i in flat_data.columns]
        flat_data = flat_data.set_index(['from', 'to'])
        flat_data = flat_data.sort_index(level=['from','to'])

        # weight matrix for the final results.
        cols = set([i[0] for i in set(flat_data.index)])
        weight_matrix = pd.DataFrame(columns=cols, index=cols)
        
        # Create the membership functions for the linguistic terms.
        terms_mf = self.automf()
        self.terms_mf = terms_mf

        for concepts in set(flat_data.index):
            activation_parameter = {}
            activation_parameter = (flat_data.loc[concepts].sum()/len(self.data)).to_dict()
            activated = self.activate(activation_parameter, self.terms_mf)
            if not all(x==0 for x in activation_parameter.values()):
                aggr = self.aggregate(activated)
                self.aggregated[f'{concepts}'] = aggr
                value = self.defuzzify(aggr, method)
                weight_matrix.loc[concepts] = value

        self.causal_weights = weight_matrix.fillna(0)