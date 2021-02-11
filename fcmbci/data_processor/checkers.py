###########################################################################
##              Class of methods to check the input data                 ##
###########################################################################

import numpy as np
import pandas as pd
from datetime import date 
from tqdm import tqdm
import collections
from data_processor.input_validator import type_check
from typing import Union

class Checker:

    """
    The class includes function input checks.

    Methods:
            consistency_check(data, column_names)
            column_check(data)
            check_lt(linguistic_terms)
            check_data(data, linguistic_terms)
            input_check(initial_state, weights)
            check_matrix(matrix: np.ndarray)
    """

    @staticmethod
    @type_check
    def consistency_check(data: dict, column_names: list):

        """
        Extract inconsistent ratings for the given linguistic terms in the supplied data.
        The method writes out an excel file with the inconsistencies and raises a ValueError if inconsistencies were identified.
        
        Parameters
        ----------
        data : dict,

        column_names: list
                        the column names (linguistic terms) of the pandas df in the ordered dictionary
        """
        current_date=date.today()

        flat_data = pd.concat([data[i] for i in data], sort = False)
        flat_data.columns = [x.lower() for x in flat_data.columns]
        flat_data = flat_data.set_index(['from', 'to'])
        flat_data = flat_data[column_names]
        pairs = set(flat_data.index) # a set of all concept pairs.
        
        incon = {}
        for pair in tqdm(pairs):
            val = {}
            for expert in data.keys():
                dat = data[expert].copy(deep=True)
                dat.columns = [x.lower() for x in dat.columns]
                dat = dat.set_index(['from', 'to']).replace(r'', np.nan)
                dat['na'] = np.nan
                dat[[i for i in dat if '-' in i]] = dat[[i for i in dat if '-' in i]] * -1
                v = dat.loc[pair].values[np.logical_not(np.isnan(dat.loc[pair].values))]
                if len(v) > 0:
                    val[expert] = int(v)
            if len(set(list(val.values()))) > 1:
                incon[pair] = val
        if incon:
            res = pd.DataFrame(incon).T
            res.index.set_names(['from', 'to'], inplace = True)
            res.to_excel(f'inconsistentRatings_{current_date.day}_{current_date.month}_{current_date.year}.xlsx', na_rep='NA')
            print(f'{list(res.index)} pairs of concepts were raited inconsistently across the experts. For more information check the inconsistentRatings_{current_date.day}_{current_date.month}_{current_date.year}.xlsx')

    @staticmethod
    @type_check
    def columns_check(data: dict):

        """
        Checks whether the dataframe includes From ---> To column. It raises an error, if the columns are not found. 
        
        Parameters
        ----------
        data : dict
        """
        for expert in data.keys():
            if ('from' not in [x.lower() for x in data[expert].columns]) | ('to' not in [x.lower() for x in data[expert].columns]):
                raise ValueError('Columns From --> To were not found. Check the data!')

    @staticmethod
    @type_check
    def check_lt(linguistic_terms: list):

        """
        Check the input of the linguistic terms against the following criteria:
            R1: should be even
            R2: should be in a list format
            R3: each item in a list should be a string
            R4: should be no douplicates

        Parameters
        ----------
        linguistic_terms: list
        """

        # R1: should be even
        if len(linguistic_terms) % 2 != 0:
            raise ValueError("You passed an odd number of linguistic terms. There should be even number of linguistic terms!")
        # R3: each item in a list should be a string     
        elif sum([type(i) != type(str()) for i in linguistic_terms]):
            raise ValueError("The linguistic terms should be strings")
        # R4: should be no douplicates 
        elif len(set(linguistic_terms)) != len(linguistic_terms):
            raise ValueError('There are douplicate linguistic terms.')

    @staticmethod
    @type_check
    def check_data(data: collections.OrderedDict, linguistic_terms: list):

        """
        Check the input data against the following criteria:
            R1: data.values shoud be a pandas.DataFrame
            R3: data.values.columns shoud include all the linguistic terms.
        
        Parameters
        ----------
        data: collections.OrderedDict

        linguistic_terms: list
        """

        # R1: data.values shoud be a pandas.DataFrame
        if sum([type(data[i]) != type(pd.DataFrame()) for i in data]) > 0:
            raise ValueError('The values in the ordered dict should be in a pandas.DataFrame format.')
        # R3: data.values.columns shoud include all the linguistic terms.
        elif sum([term not in [i.lower() for i in data[i].columns] for term in linguistic_terms for i in data]) > 0:
            raise ValueError('The columns of the dataframe should include all the linguistic terms.')

    @staticmethod
    @type_check
    def input_check(initial_state: dict, weight_matrix: Union[pd.DataFrame, np.ndarray]):

        """
        Check the inputs for simulations.
        
        Parameters
        ----------
        initial_state: str
                name of the intervention

        weight_matrix: pd.DataFrame, np.ndarray
                        causal weights between concepts
        """
        
        if len(initial_state) != weight_matrix.shape[0]:
            raise ValueError('The length of the initial_state.values() must == the length of the weights')
        
        if  type(weight_matrix) != np.ndarray:
    
            if (min(initial_state.values()) < -1) | (max(initial_state.values()) > 1):
                raise ValueError('The values in the initial_state vector are out of the input domain (-1, 1)')
            
            elif (weight_matrix.values.min() < -1) | (weight_matrix.values.max() > 1):
                raise ValueError('The values in the weight_df are out of the input domain (-1, 1)')
        else:
            if (weight_matrix.min() < -1) | (weight_matrix.max() > 1):
                raise ValueError('The values in the weight_df are out of the input domain (-1, 1)')

    @staticmethod
    @type_check
    def check_matrix(matrix: np.ndarray):
        """
        Check if a matrix is symetric.

        Parameters
        ----------
        matrix: np.ndarray
        """

        if matrix.shape[0] != matrix.shape[0]:
            raise ValueError('The matrix should be symetric!')
