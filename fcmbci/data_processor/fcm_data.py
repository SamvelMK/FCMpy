import pandas as pd
import numpy as np
import itertools
import functools
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import skfuzzy
import matplotlib.pyplot as plt
import re
from fcmbci.visualization.fcm_view import FcmVisualize

class FcmDataProcessor(FcmVisualize):
    ### Reading in files.
    def __init__(self):
        
        """ The FcmBci object initializes with a universe of discourse with a range [0,1].  """
        
        self.data = pd.DataFrame()
        self.universe = np.arange(0, 1.01, 0.01)
    ### Reading in files.
    
    def read_csv(self, file_name, sep = ','):
        '''Reads a csv file. Returns pandas data frame. 
        Note that the first column in the file is set to be the index.
        
        Parameters
        ----------
        file_name : str, 
                    ExcelFile, xlrd.Book, path object or file-like object (read more in pd.read_excel)
        sep : str, 
                default --> ','
        '''
        
        data = pd.read_csv(file_name, sep, index_col = 0)
        
        # Checks if the data meets the requirments. 
        for i in data:
            if len(data.columns) != len(data.index):
                raise ValueError("The number of columns != the number of rows. Check the data requirments!")
        
        self.data = data

    def read_xlsx(self, file_name):
        '''Reads an excel spreadsheet. Returns an ordered dictionary.
        Note that the first column in the file is set to be the index.
        
        Parameters
        ----------
        file_name : str, 
                    ExcelFile, xlrd.Book, path object or file-like object (read more in pd.read_excel)
        '''
        
        data = pd.read_excel(file_name, index_col = 0,
                                  sheet_name=None)
        
        # Checks if the data meets the requirments. 
        for i in data:
            if len(data[i].columns) != len(data[i].index):
                raise ValueError("The number of columns != the number of rows. Check the data requirments!")
        
        self.data = data
        
        
    #### Obtaining (numerical) causal weights based on expert (linguistic) inputs.
    
    def automf(self, linguistic_terms = ['VL','L', 'M', 'H', 'VH']):
        
        """ Automatically generates triangular membership functions based on the passed
        Lingustic Terms. This function was taken and modified from scikit-fuzzy.
        
        Parameters
        ----------
        linguistic_terms : lsit, 
                            default --> ['VL','L', 'M', 'H', 'VH']
                            The passed terms should be from low to high. 
        
        Return
        ---------
        y : dict,
            Generated membership functions. The key is the linguistic term and the value is a 1d array. 
        """
        
        number = len(linguistic_terms)
        limits = [self.universe.min(), self.universe.max()]
        universe_range = limits[1] - limits[0]
        widths = [universe_range / ((number - 1) / 2.)] * int(number)
        centers = np.linspace(limits[0], limits[1], number)

        abcs = [[c - w / 2, c, c + w / 2] for c, w in zip(centers, widths)]

        terms = dict()

        # Repopulate
        for term, abc in zip(linguistic_terms, abcs):
            terms[term] = skfuzzy.trimf(self.universe, abc)
        
        return terms
    
    def activate(self, activation_input, mf):
        
        """ This function is to activate the specified membership function based on the passed parameters.
        
        Parameters
        ----------
        activation_input : dict,
                            Membership function to apply the implication operation, 
                            where the key is the linguistic term and the value is the frequency its occurence.
                            Example: parameters = {'H': 0.66, 'VH': 0.33}
        mf : dict,
             membership functions upon which the implication operator is applied. The key in the dict is the linguistic term, 
             and the value is a 1d array with the membership values.
        
        Return
        ---------
        y : dict,
            Activated membership functions, where the key is the linguistic term and 
            the value is a 1d array with the activated membership values. 
        """
        
        activated = {}
        for i in activation_input.keys():
            activated[i] = np.fmin(activation_input[i], mf[i])
        
        return activated
    
    def aggregate(self, activated):
        
        """ This function aggregates the activated membership function usiing fmax operator. 
        
        Parameters
        ----------
        activated : dict,
                    A dictionary with the activated membership values to be aggregated.
        
        Return
        ---------
        y : 1d array,
            Aggregated membership function.
        """
        
        aggregated = functools.reduce(lambda x,y: np.fmax(x,y),
                             [activated[i] for i in activated.keys()])
        
        return aggregated
    
    def defuzzify(self, universe, aggregated, method = 'centroid'):
        
        """ This function defuzzifies the aggregated membership functions using centroid defuzzification method as a default.
        One can pass on another defuzzification method available in scikit-fuzzy library (e.g., bisector, mom, sig)
        The function returns the defuzzified value.

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
        
        defuzzified_value = fuzz.defuzz(universe, aggregated, method)
        
        return defuzzified_value
    
    def valence_check(self, linguistic_term):
        
        """ This function checks the valence (i.e., the sign) of the causal weight (in a linguistic Term).
        The function returns -1 if the linguistic term is negative or +1 if otherwise.
        
        Parameters
        ----------
        linguistic_term : str,
                         A string of the linguistic term --> '±H'
        
        Return
        ----------
        y : int,
            - 1 if the Linguistic Term is negative and +1 if otherwise.
        """
        
        if '-' in linguistic_term:
            return -1
        else:
            return +1
        
    def generate_edge_weights(self,
                              linguistic_terms = ['VL','L', 'M', 'H', 'VH'],
                              method = 'centroid'):
                
        """ This function applies fuzzy logic to obtain edge weights from FCM with qualitative inputs (i.e., where the 
        causal relationships are expressed in linguistic terms).
        
        Parameters
        ----------
        linguistic_terms : list,
                            A list of Linguistic Terms; default --> ['VL','L', 'M', 'H', 'VH']
                            The linguistic terms should be from low to high. 
        method : str,
                    Defuzzification method;  default --> 'centroid'. 
                    For other defuzzification methods check scikit-fuzzy library (e.g., bisector, mom, sig)
        """
        
        full_df = pd.concat([self.data[i] for i in self.data], sort = False)
        self.expert_data = full_df.copy()

        # This is to avoid SettingWithCopyWarning. We want to modify the original full_df instead of the copy of it.
        pd.options.mode.chained_assignment = None 

        for antecedent in full_df:
            # Calculates the frequency of responses for each linguistic term
            crostab = pd.crosstab(full_df[antecedent], full_df.index)/len(self.data.keys()) 
            crostab_dic = crostab.copy().to_dict() # Changes the dataframe to a dictionary.
            
            for consequent in crostab_dic.keys():
                # This creates the activation parameter. We need to clear the unnecessary characters and convert back to a dict format.
                activation_parameter = eval(str(crostab_dic[consequent]).replace('+', '').replace('-','').replace('"', '')) 
                sign = self.valence_check(list(crostab_dic[consequent].keys())[0]) # Sign of the edge calculated edge value.
                terms = self.automf(linguistic_terms)
                activated = self.activate(activation_parameter, terms)
                aggregated = self.aggregate(activated)
                value = self.defuzzify(self.universe, aggregated, method)
                full_df[antecedent][consequent] = value * sign # This repopulates the original df
        
        # Removes the redundent dulicate concepts from the df and sets the nan to 0. 
        self.causal_weights = full_df.loc[~full_df.index.duplicated(keep='first')].fillna(0)