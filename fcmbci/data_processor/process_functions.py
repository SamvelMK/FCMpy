#################################################################################################
##                                  External Functions                                         ##                                                                      
#################################################################################################

import pandas as pd
import itertools

def valence_check(linguistic_term):
        
        """ This function checks the valence (i.e., the sign) of the causal weight (in a linguistic Term).
        The function returns -1 if the linguistic term is negative or +1 if otherwise.
        
        Parameters
        ----------
        linguistic_term : str,
                            A string of the linguistic term --> '±H'
        
        Return
        ----------
        y : int,
                - 1 if the Linguistic Term is negative, +1 if its positive and 0 if otherwise.
        """
        if linguistic_term != 0:
            if '-' in linguistic_term:
                return -1
            else:
                return +1
        else:
            return 0

def consistency_check(data, dtype):
    
    """
    Checks whether the sign of the raitings the paris of the concepts
    are consistent across all the experts.
    
    Parameters
    ----------
    data : OrderedDict,
    dtype : str,
            Data format: 'Matrix' or 'List'.
    """
    # Checks whether the dtype is correctly specified.
    if dtype.lower() not in ['matrix', 'list']:
        raise ValueError(f'Unrecognized data format "{dtype}"! Check the spelling or the data type!')

    if dtype.lower() == 'matrix':
        ### Check the consistency of raitings of a matrix_like data.
        # Create a flat data with all the expert inputs and create a list of all the pairs of concepts.
        flat_data = pd.concat([data[i] for i in data], sort = False)
        indexes = [x for x in itertools.permutations(list(set(flat_data.index)), 2)]

        # for each pair of concepts; 1) Create a list of expert raitings, 2) Check the valence of each linguistic term.
        #                            3) Check if the max and min are equal.

        for pair in indexes:
            f = []
            for expert in data:
                d = data[expert]
                if len(set(pair) & set(list(d.index))) == len(set(pair)): # In case if the concept is not present in one of the expert's map.
                    l = d.loc[pair]
                    if not pd.isna(l):
                        v = valence_check(l)
                        f.append(float(v))
            if len(f) > 0:
                if min(f) != max(f):
                    raise ValueError(f'{pair} were raited inconsistently across the experts. Check the data! {f}')

    else:
        # Obtain the pairs of concepts. 1) create a flat data file with all the expert inputs. 
        #                               2) set the index of the data From, To.
        flat_data = pd.concat([data[i] for i in data], sort = False)
        flat_data = flat_data.set_index(['From', 'To'])
        indexes = set(flat_data.index) # a set of all concept pairs.

        # For each pair of concepts 1) select the expert inputs 
        #                           2) check whether the max of the list == to the min of the list
        for pair in indexes:
            f = []
            for expert in data:
                d = data[expert].set_index(['From', 'To'])
                if pair in list(d.index): # In case if the concept is not present in one of the expert's map. (Note Multiindex).
                    l = d.loc[pair].dropna()
                    if len(l) != 0:
                        f.append(float(l.values))
            if len(f) > 0:
                if min(f) != max(f):
                    raise ValueError(f'{pair} were raited inconsistently across the experts. Check the data! {f}')

def check_column(data):
    """
    Checks whether the dataframe includes From ---> To column. It raises an error, if the columns are not found. 
    
    Parameters
    ----------
    data : OrderedDict,
    """
    columns = pd.concat([data[i] for i in data], sort = False).columns
    for expert in data.keys():
        if ('from' not in [x.lower() for x in data[expert].columns]) | ('to' not in [x.lower() for x in data[expert].columns]):
            raise ValueError('Columns From --> To were not found. Check the data!')
