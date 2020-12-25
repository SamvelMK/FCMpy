import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pandas as pd
import itertools
import numpy as np
import skfuzzy as fuzz
import skfuzzy
import networkx as nx
import functools
import json
import collections
from data_processor.checkers import Checker

class FcmDataProcessor:
    
    """
    A class of methods to derive causal weights for FCMs based on linguistic terms.
    The FcmDataProcessor object is initialized with a universe of discourse with a range [-1, 1].
    """
    
    def __init__(self, linguistic_terms, data = None, check_consistency=False):
        
        """
        Parameters
        ----------
        linguistic_terms: list
                            Note that the number of linguistic terms should be even. A narrow interval around 0 (for no causality option) is added automatically.
        data: ordered dict
                qualitative expert inputs.
                default --> None
        column_names: list
                        the column names of the pandas df in the ordered dictionary
                        default --> None
        """
        self.linguistic_terms = [i.lower() for i in linguistic_terms]
        self.universe = np.arange(-1, 1.001, 0.001)
        
        if data != None:
            if column_names != None:
                column_names = [i.lower() for i in column_names]
                Checker.columns_check(data=data) # check if the from ---> to columns exist.
                if check_consistency:
                    Checker.consistency_check(data=data, column_names = self.column_names) # check the consistency of the data.
                self.data = data
            else:
                raise ValueError('The column names are not specified!')
        else:
            self.data = pd.DataFrame()            

    def read_xlsx(self, filepath, check_consistency=False):
        
        """ 
        Reads an excel spreadsheet into the constructor.
        
        Parameters
        ----------
        filepath : str, 
                    ExcelFile, xlrd.Book, path object or file-like object (read more in pd.read_excel)
        column_names: list
                        the column names of the pandas df in the ordered dictionary
        """
        column_names = [i.lower() for i in self.linguistic_terms]
        data = pd.read_excel(filepath, sheet_name=None)

        # check the data
        Checker.columns_check(data=data) # check if From ---> To columns exist: raise error if otherwise.
        if check_consistency:
            Checker.consistency_check(data=data, column_names = column_names) # Checks whether the sign of the raitings across the experts are consistent.
        self.data = collections.OrderedDict(data)            

    def read_json(self, filepath):
        """ 
        Reads data from a json file

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
        Checker.consistency_check(data=od, column_names=column_names)

        self.data = od

    #### Obtaining (numerical) causal weights based on expert (linguistic) inputs.
    
    def automf(self):
        
        """ 
        Automatically generates triangular membership functions based on the passed
        Lingustic Terms. This function was taken and modified from scikit-fuzzy.
        
        Return
        ---------
        y : dict,
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
        activation_input : dict,
                            Membership function to apply the implication operation, 
                            where the key is the linguistic term and the value is the frequency of its occurrence.
                            Example: parameters = {'H': 0.66, 'VH': 0.33}
        mf : dict,
             membership functions upon which the implication operator is applied. The key in the dict is the linguistic term, 
             and the value is a 1d array with the membership values.
        
        Return
        ---------
        y : dict,
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
        activated : dict,
                    a dictionary with the activated membership values to be aggregated.
        
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
        aggregated : 1d array,
                        Aggregated membership function to be defuzzified.
        method : str, 
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
        (i.e., where the causal relationships are expressed in linguistic terms) in an edge list format data.

        method : str,
                    Defuzzification method;  default --> 'centroid'. 
                    For other defuzzification methods check scikit-fuzzy library (e.g., bisector, mom, sig)
                    
        """        
        # A dict to store the aggregated results for the visualization purposes. 
        self.aggregated = {}

        # Create a flat data with all of the experts' inputs.
        flat_data = pd.concat([self.data[i] for i in self.data], sort = False)
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
            
            activation_parameter = {('-'+k if v < 0 else k): abs(v) for (k,v) in activation_parameter.items()}
            activated = self.activate(activation_parameter, self.terms_mf)
            if not all(x==0 for x in activation_parameter.values()):
                aggr = self.aggregate(activated)
                self.aggregated[f'{concepts}'] = aggr
                value = self.defuzzify(aggr, method)
                weight_matrix.loc[concepts] = value
            
        self.causal_weights = weight_matrix.fillna(0)
        
    def create_system(self, causal_weights):
        
        """ 
        Creates a fuzzy system/network based on the generated causal weights.
        
        Parameters
        ----------
        causal_weights : dataframe,
                            dataframe with the causal wights where the columns and rows/index represent the concepts
                            and the rows represent the weights.
        
        Return
        ----------
        y : networkx object,
        """
        causal_weights = causal_weights

        # Creates a netwrokx instance.
        G = nx.from_numpy_matrix(causal_weights.values, parallel_edges=True, 
                         create_using=nx.MultiDiGraph())
        
        # Creates truncated labels.
        labels = {idx: label_gen(val) for idx, val in enumerate(causal_weights.columns)}
        G = nx.relabel_nodes(G, labels)
        
        self.system = G