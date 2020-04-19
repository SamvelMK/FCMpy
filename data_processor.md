# Methods for deriving FCM edge weights from qualitative inputs.

## Constructing a Fuzzy Cognitive Map (Retrived from {})

<div align='justify'>

Once the structure of the FCM is available, we proceed with the second model-building step in which we need to identify the value of each edge. This is often done by providing a questionnaire to participants (Gray, Hilsberg, McFall, & Arlinghaus, 2015; Firmansyah, Supangkat, Arman, & Giabbanelli, 2019; Giabbanelli & Crutzen, 2014; Giabbanelli, Torsney-Weir, & Mago, 2012), who occasionally consist of (or are supplemented by) researchers leading the study (Mago, et al., 2013; Papada, Katsoulakos, Doulos, Kaliampakos, & Damigos, 2019). 
Then, a common practice consists of counting the fraction of respondents for each membership function (e.g. 3 out of 6 participants say “low”), clipping the function accordingly (i.e. truncate the triangle “low” after 3/6 of its height), and combining the functions and finding their center. We now detail this process on the example illustrated in Figure 1. <br>
<br>
<img src="figures\figure_1.jpg" alt="figure not found" style="float: center; margin-right: 10px;" />
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

<img src="figures\figure_2.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 2:</em> Matrix like format. <br>
<br>
<img src="figures\figure_2_1.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
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
OrderedDict([('Expert_1',         C1    C2  C3   C4
                            C1    NaN  "VH" NaN  NaN
                            C2  "-VH"   NaN NaN  NaN
                            C3   "VH"   NaN NaN  "L"
                            C4    NaN   NaN NaN  NaN), 
              
              ('Expert_2',        C1    C2  C3   C4
                            C1    NaN  "VH" NaN  NaN
                            C2  "-VH"   NaN NaN  NaN
                            C3    "M"   NaN NaN  "L"
```
```
Output2:

OrderedDict([('Expert_1',    From  To  VL    L   M   H   VH
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
             ('Expert_2',    From  To  VL    L    M   H   VH
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
The read_xlsx function stores the data in an ordered dictionary where <em>keys</em> are the experts (the names of the excel sheets) and the <em>values</em> are the expert inputs.

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
mf = fcm.automf(universe, ['-VH', '-H', '-M', '-L', '-VL', 'VL', 'L', 'M', 'H', 'VH'])

```
```
Output:

{'-VH': array([1.   , 0.996, 0.992, ..., 0.   , 0.   , 0.   ]),
 '-H': array([0.001, 0.005, 0.009, ..., 0.   , 0.   , 0.   ]),
 '-M': array([0., 0., 0., ..., 0., 0., 0.]),
 '-L': array([0., 0., 0., ..., 0., 0., 0.]),
 '-VL': array([0., 0., 0., ..., 0., 0., 0.]),
 'VL': array([0., 0., 0., ..., 0., 0., 0.]),
 'L': array([0., 0., 0., ..., 0., 0., 0.]),
 'M': array([0., 0., 0., ..., 0., 0., 0.]),
 'H': array([0.   , 0.   , 0.   , ..., 0.009, 0.005, 0.001]),
 'VH': array([0.   , 0.   , 0.   , ..., 0.992, 0.996, 1.   ])}
 
```

You can visualize this with the mf_view() method.

<img src="figures\figure_3.png" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 3:</em> Automatically generated triangular membership function.

</div>

## activate()
<div align='justify'>

Activate the specified membership function based on the passed parameters.

```
activate(activation_input, mf)

Parameters
----------
activation_input : dict,
                    Membership function to apply the implication operation, 
                    where the key is the linguistic term and the value is the frequency its occurence .
                    Example: parameters = {'H': 0.66, 'VH': 0.33}
mf : dict,
        membership functions upon which the implication operator is applied. The key in the dict is the linguistic term, 
        and the value is a 1d array with the membership values.
        
Return
---------
y : dict,
        Activated membership functions, where the key is the linguistic term and 
        the value is a 1d array with the activated membership values. 
```
To activate the membership functions we need to pass in the activation input along with the membership function. The activation input is a dictionary with the keys as the linguistic terms and the values as the frequency of the occurance. As in the example presented in the background section, let's suppose that 1/6 experts expressed the causality between a given pair of concepts as Medium, 3/6 experts expressed it as High and 2/6 expressed it as VH. The respective activation input would look like this: {'M': 0.16, 'H': 0.5, 'VH': 0.33}.
The function returns a dictionary with the activated membership function.

```
Example:
    
act = fcm.activate({'M': 0.16, 'H': 0.5, 'VH': 0.33}, mf)
```
```
Output:

{'M': array([0., 0., 0., ..., 0., 0., 0.]),
 'H': array([0.   , 0.   , 0.   , ..., 0.009, 0.005, 0.001]),
 'VH': array([0.  , 0.  , 0.  , ..., 0.33, 0.33, 0.33])}
```
Essentially, the values in the activation input determine the point at which the membership function of each linguistic term is going to be cut. You can visualize it with the fcm.mf_view() method (Figure 4). 

```
fcm.mf_view(act)
```
<img src="figures\figure_4.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 4:</em> The activated membership function.

</div>

## aggregate()
<div align='justify'>
Now that we have activated the respective membership functions, we need to aggregate them before we can derive the precise causal weight through the difuzzification process.

```
aggregate(activated)

Parameters
----------
activated : dict,
                A dictionary with the activated membership values to be aggregated.
        
Return
---------
y : 1d array,
        Aggregated membership function.
```
Example:

```
aggr = fcm.aggregate(act)
```

You can visualize it as follows:

```
import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 5))
axes = plt.axes()
axes.plot(fcmbci.universe, aggr, linewidth=0.4)
axes.fill_between(fcmbci.universe, aggr, alpha=0.5)

        
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
```


<img src="figures\figure_5.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 5:</em> The aggregated membership function.

</div>

## defuzzify()
<div align='justify'>

After aggregating the activated membership functions we can derive derive the causal weight between the given two concepts. The defuzzify() method uses the defuzz() method from python scikit-fuzzy module. 

```
defuzzify(universe, aggregated, method = 'centroid')

Parameters
----------
universe : 1d array,
            The universe of discourse.

aggregated : 1d array,
                Aggregated membership function to be defuzzified.
method : str, 
            Defuzzification method, default --> 'centroid'. 
            For other defuzzification methods check scikit-fuzzy library (e.g., bisector, mom, sig)
        
Return
---------
y : int,
    Defuzzified value.
```

Example:

```
defuzz = fcm.defuzzify(universe, aggr)
```
```
Output:

0.703888431055232
```
This can be visualized as follows (The code is adjusted from skit-fuzzy tutorial):

```
import matplotlib.pyplot as plt
import skfuzzy as fuzz

fig = plt.figure(figsize=(10, 5))
axes = plt.axes()
legend_anchor=(0.95, 0.6)
        
y_activation = fuzz.interp_membership(universe, aggr, defuzz)  # for plot
out = np.zeros_like(universe) 

for i in mf:
    axes.plot(universe, mf[i], linewidth=0.3, label=str(i)) # plots all the mfs. 
    axes.fill_between(universe, mf[i], alpha=0.4)
                
    axes.fill_between(universe, out, 
                             aggr, facecolor='#36568B', alpha=1)
            
    axes.plot([defuzz, defuzz], [0, y_activation], 'k', linewidth=1.5, alpha=1)

    axes.legend(bbox_to_anchor=legend_anchor)
    axes.text(defuzz-0.05, y_activation+0.02, f'{round(defuzz, 2)}',
                     bbox=dict(facecolor='red', alpha=0.1))
            
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
```

<img src="figures\figure_6.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 6:</em> Derived edge weight (defuzzification).

</div>

## gen_weights_mat(), gen_weights_list()
<div align='justify'>

The methods described above are wrapped in gen_weights_mat() and gen_weights_list() methods to allow for automatic generation of weight matrices based on qualitative inputs. One can either pass the data to the method, or the data can be taken from the constructor. 
```
gen_weights_mat(data = None,
                    linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'],
                    method = 'centroid')
```
```
gen_weights_list(data = None,
                    linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'],
                    method = 'centroid')

```
Example:

```
Read in the data from matrix
fcm = FcmDataProcessor()
fcm.read_xlsx('sample.xlsx', 'Matrix')

Read in the data from list
fcml = FcmDataProcessor()
fcm.read_xlsx('sample.xlsx', 'List')
```
```
fcm.gen_weights_mat()
fcml.gen_weights_list()
```

The weight matrix can be inspected as follows:

```
fcm.causal_weights
```
```
Output:

        C1        C2    C3      C4
C1  0.000000  0.702205   0  0.000000
C2 -0.610698  0.000000   0  0.000000
C3  0.556908  0.000000   0  0.230423
C4  0.000000  0.000000   0  0.000000
```
One can inspect the frequency of the raitings each linguistic term for a given pair of concepts with the term_freq_hist() method.

```
fcm.term_freq_hist('C1', 'C2')
```
<img src="figures\figure_7.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 7:</em> Expert's raitings of causal trength between C1 and C2.

One can also visually inspect the deffuzification of the activated membership functions between a pair of concepts with the defuzz_view() method.

```
fcm.defuzz_view('C1', 'C2')
```

<img src="figures\figure_8.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 8:</em> Deffuzification of the aggregated membership functions of concepts C1 and C2.

</div>

## create_system()
<div align='justify'>

create_system(causal_weights = None)
</div>
