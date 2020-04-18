# Methods for deriving FCM edge weights from qualitative inputs.

## Constructing a Fuzzy Cognitive Map (Retrived from {})

<div align='justify'>

Once the structure of the FCM is available, we proceed with the second model-building step in which we need to identify the value of each edge. This is often done by providing a questionnaire to participants (Gray, Hilsberg, McFall, & Arlinghaus, 2015; Firmansyah, Supangkat, Arman, & Giabbanelli, 2019; Giabbanelli & Crutzen, 2014; Giabbanelli, Torsney-Weir, & Mago, 2012), who occasionally consist of (or are supplemented by) researchers leading the study (Mago, et al., 2013; Papada, Katsoulakos, Doulos, Kaliampakos, & Damigos, 2019). 
Then, a common practice consists of counting the fraction of respondents for each membership function (e.g. 3 out of 6 participants say “low”), clipping the function accordingly (i.e. truncate the triangle “low” after 3/6 of its height), and combining the functions and finding their center. We now detail this process on the example illustrated in Figure 1. <br>
<br>
<img src="figures\figure_1.jpg" alt="figure not found" style="float: left; margin-right: 10px;" />
<em>Figure 1:</em> Four steps process to obtain the quantitative value of an edge’s causal strength from the qualitative evaluation of a panel of participants.<br>
<br>
First, a questionnaire is provided, which asks participant to choose one linguistic variable for each relation (or subset of relations with which the participant is familiar). For instance, participants must evaluate the perceived causal impact of “perceived norms about sex and condom use” onto “belief that condoms must be used during sexual intercourse”. Once participants have completed the evaluation, their answers are aggregated. Here, we consider 6 participants: three chose ‘Medium’, one chose ‘Strong’, and two chose ‘Very Strong’. Then, we can use fuzzy logic per se. We define a membership function for each linguistic variable, allowing for overlaps between functions. As shown with triangular membership functions in Figure 3, there is a small possibility that a participant saying “medium” may think the same as one stating “high”. Fuzzy implications are used to take each membership function to the extent in which it was endorsed by participants. For instance, if 3 out of 6 participants (i.e. half) stated “medium”, then we project from the y-axis at 0.5 onto the membership function and preserve only the part under 0.5. This defines an implication as the minimum of the function that is triggered. After each membership function has been dealt with, we combine them to represent the judgment from the collective of participants. The combination consists of aggregating the function (i.e. taking the maximum) and using the centroid as representative value. Finally, as exemplified in Figure 4, the centroid is projected onto the fuzzy range of [0, 1] to obtain the strength of causation.

Although the fuzzy range produced by the process is always [0, 1], causal connections are not necessarily positive: there are also negative causations in which case some determinants decrease the value of others (Figure 3; red edges). The polarity of the causal connection is applied at the end of this process. For instance, the questionnaire in Figure 4 asked participants to predict the strength of the increase (Figure 1; Step 1) hence the value will be positive. Conversely, if participants had to predict the strength of the decrease, the result would be made negative (i.e. multiplied by -1). 
Note that participants may be given options such as “Non-existent” (Figure 1) or “I am not sure” in the questionnaire. Answers equivalent to “I am not sure” are skipped and thus not counted as part of the total number of respondents when computing the ratio of answers (Step 2→3, Figure 1). Answers equivalent to “Non-existent” can be mapped to 0 or a very narrow membership function (e.g. peaking at 0 and minimum at 0.001). The effect of introducing such an option is that some edges may turn out to carry a very small weight and may be dropped from the model.

This conversion process from qualitative linguistic terms into quantitative outputs is grounded in Fuzzy Set Theory, hence the name of ‘Fuzzy’ cognitive maps. As stated by Li and Zhang (emphases added), “Fuzzy set theory resembles human reasoning under approximate information and inaccurate data to generate decisions under uncertain environments. It is designed to mathematically represent uncertainty and vagueness, and to provide formalized tools for dealing with imprecision in real-world problems.” (Li & Zhang, 2007). The use of Fuzzy Set Theory distinguishes FCMs from conceptual maps (e.g. mind maps, concept maps) which either do not have causal weights or assign them through processes that are not mathematically designed to handle uncertainty. 

Given the use of fuzzy logic, the result from the process is a number in the interval [-1, 1] where -1 is the strongest negative causation and 1 the strongest positive causation. For instance, Figure 1 depicts a strong positive causation.
</div>

## FcmDataProcessor

To create an instance of FcmDataProcessor class you can either pass a dataframe that contains the data directly to the constructor 

```
fcm = FcmDataProcessor(df)
```
or use the [read_xlsx](#read_xlsx) method after creating the instance with no argument. 

The instance is initialized with a universe of discourse with a range of [-1, 1].

## Methods

<div align='justify'>

The methods presented in this section are used to derive the edge weights based on qualitative inputs as described above.

- [read_xlsx](#read_xlsx)
- [automf](#automf)
- [activate](#activate)
- [aggregate](#aggregate)
- [defuzzify](#defuzzify)
- [gen_weights_mat](#gen_weights_mat)
- [gen_weights_list](#gen_weights_list)
- [create_system](#create_system)

</div>

## read_xlsx()
<div align='justify'>
The read xlsx function takes the same argument as pd.read_excel function. 

```
read_xlsx(file_name, dtype)

Parameters
----------        
file_name : str, 
            ExcelFile, xlrd.Book, path object or file-like object (read more in pd.read_excel)

dtype : str,
        'List', 'Matrix'
```
However, the data that it expects should be in a specific shape. In the current version, the function can take either matrix-like or edge list formats.

<img src="figures\figure_2.PNG" alt="figure not found" style="float: left; margin-right: 10px;" />
<em>Figure 2:</em> Matrix like format. <br>
<br>
<img src="figures\figure_2_1.PNG" alt="figure not found" style="float: left; margin-right: 10px;" />
<em>Figure 2.1:</em> Matrix like format.

Example:

```
fcm.read_xlsx('sample.xlsx', 'Matrix')
fcm.read_xlsx('list_format.xlsx', 'List')
```
The data is read into the constructor and can be accessed in the following way:

```
fcm.data
```
```
Output1:
OrderedDict([('Expert_1',             C1    C2  C3    C4
                                C1    NaN  "VH" NaN  NaN
                                C2  "-VH"   NaN NaN  NaN
                                C3   "VH"   NaN NaN  "L"
                                C4    NaN   NaN NaN  NaN), 
('Expert_2',                          C1    C2  C3   C4
                                C1    NaN  "VH" NaN  NaN
                                C2  "-VH"   NaN NaN  NaN
                                C3    "M"   NaN NaN  "L"
                                C4    NaN   NaN NaN  NaN),
```
```
Output2:

OrderedDict([('Expert_1',      From  To  VL    L   M   H   VH
                            0    C1  C1 NaN  NaN NaN NaN  NaN
                            1    C1  C2 NaN  NaN NaN NaN -1.0
                            2    C1  C3 NaN  NaN NaN NaN  NaN
                            3    C1  C4 NaN  NaN NaN NaN  NaN
                            4    C2  C1 NaN  NaN NaN NaN  1.0
                            5    C2  C2 NaN  NaN NaN NaN  NaN
                            6    C2  C3 NaN  NaN NaN NaN  NaN
                            7    C2  C4 NaN  NaN NaN NaN  NaN
                            8    C3  C1 NaN  NaN NaN NaN  1.0
                            9    C3  C2 NaN  NaN NaN NaN  NaN
                            10   C3  C3 NaN  NaN NaN NaN  NaN
                            11   C3  C4 NaN  1.0 NaN NaN  NaN
                            12   C4  C1 NaN  NaN NaN NaN  NaN
                            13   C4  C2 NaN  NaN NaN NaN  NaN
                            14   C4  C3 NaN  NaN NaN NaN  NaN
                            15   C4  C4 NaN  NaN NaN NaN  NaN), 
('Expert_2',                   From  To  VL    L    M   H   VH
                            0    C1  C1 NaN  NaN  NaN NaN  NaN
                            1    C1  C2 NaN  NaN  NaN NaN -1.0
                            2    C1  C3 NaN  NaN  NaN NaN  NaN
                            3    C1  C4 NaN  NaN  NaN NaN  NaN
                            4    C2  C1 NaN  NaN  NaN NaN  1.0
                            5    C2  C2 NaN  NaN  NaN NaN  NaN
                            6    C2  C3 NaN  NaN  NaN NaN  NaN
                            7    C2  C4 NaN  NaN  NaN NaN  NaN
                            8    C3  C1 NaN  NaN  1.0 NaN  NaN
                            9    C3  C2 NaN  NaN  NaN NaN  NaN
                            10   C3  C3 NaN  NaN  NaN NaN  NaN
                            11   C3  C4 NaN  1.0  NaN NaN  NaN
                            12   C4  C1 NaN  NaN  NaN NaN  NaN
                            13   C4  C2 NaN  NaN  NaN NaN  NaN
                            14   C4  C3 NaN  NaN  NaN NaN  NaN
                            15   C4  C4 NaN  NaN  NaN NaN  NaN),
```
The read_xlsx function returns an ordered dictionary where keys are the experts (the names of the excel sheets) and the values are the expert inputs.

</div>

## automf()
<div align='justify'>

Automatically generates triangular membership functions based on the passed Lingustic Terms. This function was taken and modified from scikit-fuzzy.

```
automf(universe, 
            linguistic_terms = ['-VH', '-H', '-M', '-L','-VL', 'VL','L', 'M', 'H', 'VH'])

Parameters
---------- 
universe : 1d array,
                    The universe of discourse.
                    
linguistic_terms : lsit, 
                    default --> ['-VH', '-H', '-M', '-L', '-VL', 'VL', 'L', 'M', 'H', 'VH']
                    Note that the number of linguistic terms should be even. A narrow interval around 0 is added automatically.
        
Return
---------
y : dict,
    Generated membership functions. The key is the linguistic term and the value is a 1d array. 
```

The function returns a dictionary with the linguistic terms as the keys and the corresponding numerical intervals (in 1d arrays) as the values.

Example:

```
import numpy as np

universe = np.arange(-1, 1.001, 0.001)
lt = automf(universe, ['-VH', '-H', '-M', '-L', 'L', 'M', 'H', 'VH'])
```
```
Output:

{'-VH': array([1.   , 0.997, 0.994, ..., 0.   , 0.   , 0.   ]), '-H': array([0.001, 0.004, 0.007, ..., 0.   , 0.   , 0.   ]), '-M': array([0., 0., 0., ..., 0., 0., 0.]), '-L': array([0., 0., 0., ..., 0., 0., 0.]), 'L': array([0., 0., 0., ..., 0., 0., 0.]), 'M': array([0., 0., 0., ..., 0., 0., 0.]), 'H': array([0.   , 0.   , 
0.   , ..., 0.007, 0.004, 0.001]), 'VH': array([0.   , 0.   , 0.   , ..., 0.994, 0.997, 1.   ])}
```

You can visualize this with the mf_view() method.

<img src="figures\figure_3.png" alt="figure not found" style="float: left; margin-right: 10px;" />
<em>Figure 3:</em> Automatically generated triangular membership function.

</div>

## activate()
activate(activation_input, mf)

## aggregate()
aggregate(activated)

## defuzzify()
defuzzify(universe, aggregated, method = 'centroid')

## gen_weights_mat()
gen_weights_mat(data = None,
                    linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'],
                    method = 'centroid')

## gen_weights_list()
gen_weights_list(data = None,
                    linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'],
                    method = 'centroid') 

## create_system()
create_system(causal_weights = None)