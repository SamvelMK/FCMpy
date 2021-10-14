import numpy as np
import pandas as pd 
import re
import collections
import json
from abc import ABC, abstractmethod
from fcmpy.expert_fcm.input_validator import type_check
from fcmpy.expert_fcm.checkers import ConsistencyCheck
from fcmpy.expert_fcm.checkers import ColumnsCheck


class ReadData(ABC):
    """
        Class of methods for reading in data.
    """
    @abstractmethod
    def read() -> collections.OrderedDict:
        raise NotImplementedError('read method is not defined!')


class CSV(ReadData):
    """
        Read data from a .csv file.
    """
    def __init__(self):
        pass

    @type_check
    def __conceptParser(self, string: str, sepConcept: str) -> dict:
        """
            Parse the csv file column names. Extract the antecedent -> 
            consequent pairs and the polarity of the causal relationship.

            Parameters
            ----------
            string: str
                    the column name that need to be parsed
            
            sepConcept: str
                        the separation symbol (e.g., '->') that separates 
                        the antecedent from the consequent in the columns of a csv file
            
            Return
            -------
            y: dict
                keys --> antecedent, consequent, polarity
        """
        dt = {}
        pattern = f'[a-zA-Z]+.+->.+.(\(\+\)|\(\-\))' # detect the column head pattern: antesedent -> consequent (polarity)
        patterMatch = bool(re.search(pattern, string))
        
        if patterMatch:
            dt['polarity'] = re.search(r'\((.*?)\)', string).group(1)
            concepts = string.split(sepConcept)
            dt['antecedent'] = re.sub(r'\([^)]*\)', '', concepts[0]).strip() 
            dt['consequent'] = re.sub(r'\([^)]*\)', '', concepts[1]).strip()
            return dt
        else:
            raise ValueError('The $antecedent$ $->$ $consequent (sign)$ \
                                format is not detected! Check the data format!')
    
    @type_check
    def __extractRowData(self, data: dict, sepConcept: str, 
                            linguisticTerms: list) -> pd.DataFrame:
        """
            Convert csv data fromat to a dataframe with columns representing the linguistic terms
            and rows representing the participants inputs (i.e., expert inputs).

            Parameters
            ----------
            data: dict
                    data to be extracted
            sepConcept: str
                        the separation symbol (e.g., '->') that separates the antecedent from the
                        consequent in the columns of a csv file
            
            linguisticTerms: list
                                list of linguistic terms
            
            Return
            -------
            y: pandas.DataFrame
        """
        ltValence = [i.lower().strip('+-') for i in linguisticTerms if i.startswith(('-', '+'), 0, 1)] # get the terms that have valence

        # Columns of the dfs (add the no causality term)
        dict_data = []
        for i in data.keys():
            _ = {i: 0 for i in linguisticTerms}
            conceptsParsed = self.__conceptParser(string=i, sepConcept=sepConcept)
            _['From'] = conceptsParsed['antecedent']
            _['To'] = conceptsParsed['consequent']

            # no causality cases
            if data[i].lower() in ltValence:
                if conceptsParsed['polarity'] == '+':
                    _[str('+'+data[i].lower())] = 1
                else:
                    _[str('-'+data[i].lower())] = 1
            else:
                _[data[i].lower()] = 1
            
            dict_data.append(_)
        
        return pd.DataFrame(dict_data).fillna(0)
    
    @type_check
    def read(self, **kwargs) -> collections.OrderedDict:
        """ 
            Read data from a csv file.

            Other Parameters
            ----------------
            **filepath : str

            **linguisticTerms: dictionary
                                dictionary of linguistic terms used to express causality between concepts.

            **sepConcept: str
                        the separation symbol (e.g., '->') that separates the antecedent from the consequent in the columns of a csv file

            **csvSep: str,
                        separator of the csv file (read more in pandas.read_csv)

            Return
            -------
            data: collections.OrderedDict
                    ordered dictionary with the formatted data.
        """
        filePath = kwargs['filePath']
        linguisticTerms = kwargs['linguisticTerms']
        linguisticTerms = [i.lower() for i in list(linguisticTerms.keys())] # Create a list

        try:
            sepConcept = kwargs['params']['sep_concept']
        except:
            sepConcept = '->'
        
        try:
            csvSep = kwargs['params']['csv_sep']
        except: # set the default csv sep to ','
            csvSep = ','

        data = pd.read_csv(filePath, sep=csvSep)
        dataOd = collections.OrderedDict()
        for i in range(len(data)):
            _ = data.iloc[i].to_dict()
            expertData = self.__extractRowData(data=_, sepConcept=sepConcept, linguisticTerms=linguisticTerms)
            dataOd[f'Expert{i}'] = expertData
        
        data = dataOd

        ColumnsCheck.checkColumns(data=data) # check whether From -> To columns exist.

        return data


class XLSX(ReadData):
    """
        Read data from .xlsx file.
    """
    @staticmethod
    @type_check
    def read(**kwargs) -> collections.OrderedDict:
        """ 
            Read data from .xlsx file.
            
            Other Parameters
            ----------------
            **filePath : str

            **checkConsistency: Bool
                                check the consistency of ratings across the experts.
                                default --> False
            
            **engine: str,
                        the engine for excel reader (read more in pd.read_excel)
                        default --> "openpyxl"
            
            Return
            -------
            data: collections.OrderedDict
                    ordered dictionary with the formatted data.
        """
        filePath = kwargs['filePath']

        try:
            checkConsistency = kwargs['params']['check_consistency']
        except:
            checkConsistency = False

        try:
            engine = kwargs['params']['engine']
        except:
            engine = 'openpyxl'

        data = pd.read_excel(filePath, sheet_name=None,  engine=engine)
        data = collections.OrderedDict(data)

        ColumnsCheck.checkColumns(data=data) # check whether From -> To columns exist.

        if checkConsistency:
            ConsistencyCheck.checkConsistency(data=data) # check the consistency of the linguistic ratings.
        
        return data


class JSON(ReadData):
    """
    Read data from .json file.
    """
    @staticmethod
    @type_check
    def read(**kwargs) -> collections.OrderedDict:
        """ 
            Read data from a .json file.

            Other Parameters
            ----------------
            **filePath : str, path object or file-like object

            **checkConsistency: Bool
                                check the consistency of ratings across the experts.
                                default --> False
            
            Return
            -------
            data: collections.OrderedDict
                    ordered dictionary with the formatted data.
        """
        filePath = kwargs['filePath']

        try:
            checkConsistency = kwargs['params']['check_consistency']
        except:
            checkConsistency = False

        f = open(filePath) 
        dataJson = json.load(f)
        f.close()
        d = {}
        for i in dataJson.keys():
            d[i] = dataJson[i]
        data = collections.OrderedDict([(i, pd.DataFrame(d[i]).replace(r'^\s*$', np.nan, regex=True)) for i in d])

        ColumnsCheck.checkColumns(data=data) # check whether From -> To columns exist.

        if checkConsistency:
            ConsistencyCheck.checkConsistency(data=data) # check the consistency of the linguistic ratings.

        return data