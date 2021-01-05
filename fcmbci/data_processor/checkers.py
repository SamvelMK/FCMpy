###########################################################################
##              Class of methods to check the imput data                 ##
###########################################################################

import numpy as np
import pandas as pd
from datetime import date 
from tqdm import tqdm

class Checker:

    @staticmethod
    def consistency_check(data, column_names):
        """
        Extract inconsistent ratings for the given linguistic terms in the supplied data.
        
        Parameters
        ----------
        data : OrderedDict,
        column_names: list
                        the column names (linguistic terms) of the pandas df in the ordered dictionary
        Return
        ----------
        Writes out an excel file with the inconsistencies and raises a ValueError if inconsistencies were identified.
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
    def columns_check(data):
        """
        Checks whether the dataframe includes From ---> To column. It raises an error, if the columns are not found. 
        
        Parameters
        ----------
        data : OrderedDict
        """
        for expert in data.keys():
            if ('from' not in [x.lower() for x in data[expert].columns]) | ('to' not in [x.lower() for x in data[expert].columns]):
                raise ValueError('Columns From --> To were not found. Check the data!')
    
    @staticmethod
    def input_check(initial_state, weights_df):
        if len(initial_state) != len(weights_df.columns):
            raise ValueError('The length of the initial_state.values() must == the length of the weights_df.columns')

        elif min(initial_state.values()) < -1 & max(initial_state.values()) > 1:
            raise ValueError('The values in the initial_state vector are out of the input domain (-1, 1)')
        
        elif weights_df.values.min() < -1 & weights_df.values.max() > 1:
            raise ValueError('The values in the weight_df are out of the input domain (-1, 1)')