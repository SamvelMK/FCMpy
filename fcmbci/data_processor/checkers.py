###########################################################################
##              Class of methods to check the imput data                 ##
###########################################################################

import numpy as np
import pandas as pd
from datetime import date 

class Checker:

    @staticmethod
    def consistency_check(data, linguistic_terms):
        """
        Extract inconsistent ratings for the given linguistic terms
        
        Parameters
        ----------
        data : OrderedDict,
        linguistic_terms: list
                            list of linguistic terms to be evaluated
        Return
        ----------
        Writes out an excel file with the inconsistencies and raises a ValueError if inconsistencies were identified
        """
        current_date=date.today()

        flat_data = pd.concat([data[i] for i in data], sort = False)
        flat_data.columns = [x.lower() for x in flat_data.columns]
        flat_data = flat_data.set_index(['from', 'to'])
        columns = set([i.lower().replace(r'-', '') for i in linguistic_terms])
        flat_data = flat_data[columns]
        pairs = set(flat_data.index) # a set of all concept pairs.
        
        incon = {}
        for pair in pairs:
            val = {}
            for expert in data.keys():
                dat=data[expert]
                dat.columns = [x.lower() for x in dat.columns]
                dat = dat.set_index(['from', 'to'])[columns].replace(r'', np.nan)
                v = dat.loc[pair].values[np.logical_not(np.isnan(dat.loc[pair].values))]
                if len(v) > 0:
                    val[expert] = int(v)
            if len(set(list(val.values()))) > 1:
                incon[pair] = val
        if incon:
            res = pd.DataFrame(incon).T
            res.index.set_names(['from', 'to'], inplace = True)
            res.to_excel(f'inconsistentRatings_{current_date.day}_{current_date.month}_{current_date.year}.xlsx', na_rep='NA')
            raise ValueError(f'{list(res.index)} pairs of concepts were raited inconsistently across the experts. For more information check the inconsistentRatings_{current_date.day}_{current_date.month}_{current_date.year}.xlsx')

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