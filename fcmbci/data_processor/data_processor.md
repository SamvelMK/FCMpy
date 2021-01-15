# Methods for deriving FCM edge weights from qualitative inputs.

## Constructing a Fuzzy Cognitive Map (Retrived from Mkhitaryan, Giabbanelli, Vries & Crutzen, 2020)

<div align='justify'>

Once the structure of the FCM is available, we proceed with the second model-building step in which we need to identify the value of each edge. This is often done by providing a questionnaire to participants (Gray, Hilsberg, McFall, & Arlinghaus, 2015; Firmansyah, Supangkat, Arman, & Giabbanelli, 2019; Giabbanelli & Crutzen, 2014; Giabbanelli, Torsney-Weir, & Mago, 2012), who occasionally consist of (or are supplemented by) researchers leading the study (Mago, et al., 2013; Papada, Katsoulakos, Doulos, Kaliampakos, & Damigos, 2019). 
Then, a common practice consists of counting the fraction of respondents for each membership function (e.g. 3 out of 6 participants say “low”), clipping the function accordingly (i.e. truncate the triangle “low” after 3/6 of its height), and combining the functions and finding their center. We now detail this process on the example illustrated in Figure 1. <br>
<br>
<img src="..\..\figures\figure_1.jpg" alt="figure not found" style="float: center; margin-right: 10px;" />
<em>Figure 1:</em> Four steps process to obtain the quantitative value of an edge’s causal strength from the qualitative evaluation of a panel of participants.<br>
<br>
First, a questionnaire is provided, which asks participant to choose one linguistic variable for each relation (or subset of relations with which the participant is familiar). For instance, participants must evaluate the perceived causal impact of “perceived norms about sex and condom use” onto “belief that condoms must be used during sexual intercourse”. Once participants have completed the evaluation, their answers are aggregated. Here, we consider 6 participants: three chose ‘Medium’, one chose ‘Strong’, and two chose ‘Very Strong’. Then, we can use fuzzy logic per se. We define a membership function for each linguistic variable, allowing for overlaps between functions. As shown with triangular membership functions in Figure 3, there is a small possibility that a participant saying “medium” may think the same as one stating “high”. Fuzzy implications are used to take each membership function to the extent in which it was endorsed by participants. For instance, if 3 out of 6 participants (i.e. half) stated “medium”, then we project from the y-axis at 0.5 onto the membership function and preserve only the part under 0.5. This defines an implication as the minimum of the function that is triggered. After each membership function has been dealt with, we combine them to represent the judgment from the collective of participants. The combination consists of aggregating the function (i.e. taking the maximum) and using the centroid as representative value. Finally, as exemplified in Figure 4, the centroid is projected onto the fuzzy range of [0, 1] to obtain the strength of causation.

Although the fuzzy range produced by the process is always [0, 1], causal connections are not necessarily positive: there are also negative causations in which case some determinants decrease the value of others (Figure 3; red edges). The polarity of the causal connection is applied at the end of this process. For instance, the questionnaire in Figure 4 asked participants to predict the strength of the increase (Figure 1; Step 1) hence the value will be positive. Conversely, if participants had to predict the strength of the decrease, the result would be made negative (i.e. multiplied by -1). 
Note that participants may be given options such as “Non-existent” (Figure 1) or “I am not sure” in the questionnaire. Answers equivalent to “I am not sure” are skipped and thus not counted as part of the total number of respondents when computing the ratio of answers (Step 2→3, Figure 1). Answers equivalent to “Non-existent” can be mapped to 0 or a very narrow membership function (e.g. peaking at 0 and minimum at 0.001). The effect of introducing such an option is that some edges may turn out to carry a very small weight and may be dropped from the model.

This conversion process from qualitative linguistic terms into quantitative outputs is grounded in Fuzzy Set Theory, hence the name of ‘Fuzzy’ cognitive maps. As stated by Li and Zhang (emphases added), “Fuzzy set theory resembles human reasoning under approximate information and inaccurate data to generate decisions under uncertain environments. It is designed to mathematically represent uncertainty and vagueness, and to provide formalized tools for dealing with imprecision in real-world problems.” (Li & Zhang, 2007). The use of Fuzzy Set Theory distinguishes FCMs from conceptual maps (e.g. mind maps, concept maps) which either do not have causal weights or assign them through processes that are not mathematically designed to handle uncertainty. 

Given the use of fuzzy logic, the result from the process is a number in the interval [-1, 1] where -1 is the strongest negative causation and 1 the strongest positive causation. For instance, Figure 1 depicts a strong positive causation.
</div>

## DataProcessor

<div align='justify'>

To create an instance of the DataProcessor class you can either pass the data (collections.OrderedDict where the key is the expert ID and the values are panda's dataframes) to the constructor or create an instance with no data and then read the data by using one of the methods of this class (i.e., [read_xlsx](#read_xlsx) or [read_json](#read_json)).

To instantiate the FcmDataProcessor class one needs to pass a list of linguistic terms that need to be converted to numerical weights. 

```
from fcmbci import DataProcessor

lt = ['-VH', '-H', '-M', '-L','-VL', 'VL','L', 'M', 'H', 'VH']

fcm = DataProcessor(linguistic_terms=lt)
```
Note that the class is automatically instantiated with a universe of discourse with a range of [-1, 1].

When supplying the data one can also specify whether there is a need to check for the consistency in the data. The <em>consistency_check</em> argument checks the consistency of the raitings (mainly the valency of the raitings) of each pari of concepts across all the experts. If inconsistencies are identified then an inconsistencies_current_date.xlsx file is generated. In this file you can find the pair of concepts that were rated inconsistently across the experts.
<div>

## Methods

<div align='justify'>

The methods presented in this section are used to derive the edge weights based on qualitative inputs as described above.

- [read_xlsx](#read_xlsx)
- [read_json](#read_json)
- [automf](#automf)
- [activate](#activate)
- [aggregate](#aggregate)
- [defuzzify](#defuzzify)
- [gen_weights_mat](#gen_weights)

</div>

## read_xlsx()
<div align='justify'>

The read xlsx method takes two arguments: <em> filepath </em>, <em> check_consistency </em>.

```
read_xlsx(filepath, check_consistency=False)

Parameters
----------
filepath : str, 
                ExcelFile, xlrd.Book, path object or file-like object (read more in pd.read_excel)
check_consistency: Bool
                        check the consistency of raitings across the experts.
                        default --> False
```
The data that it expects should be in a specific shape. The data must have the following columns:
* From: Representing antecedents
* To: Representing consequents
* L(n): Representing linguistic term(s) n (e.g., VL, L, M etc.).

<img src="..\..\figures\figure_2.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 2:</em> Sample data structure.

Example:

```
fcm.read_xlsx(filepath, check_consistency=True)
```

The data can be accessed in the following way:

```
fcm.data
```

```
Output1:

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
The read_xlsx() method stores the data in an ordered dictionary where the <em>keys</em> are the expert ids (i.e., the names of the excel sheets) and the <em>values</em> are panda's dataframes of the expert inputs.

</div>

## read_json()
<div align='justify'>

In certain cases one might have to read the data from json files. The json file should have the following general structure:

```
{"ExpertId": [
                {"from": 1, "to": 1, "NA": "", "VL": "", "L": "", "M": 1, "H": "", "VH": ""}, 
                {"from": 1, "to": 2, "NA": "", "VL": 1, "L": "", "M": "", "H": "", "VH": ""}
             ]
}
```
The read_json method takes the filepath and a list of keys as arguments. The keys argument are the keys in the json file that represent the linguistic terms. 

```
read_json(filepath, keys)

Parameters
----------
filepath : str, path object or file-like object
keys: list
        the column names of the pandas df in the ordered dictionary       
```

Example:

```
keys = ['VL', 'L', 'M', 'H', 'VH']
fcm.read_json(data='data.json', keys = keys)
```

</div>

## automf()
<div align='justify'>

This method automatically generates triangular membership functions based on the passed linguistic terms (during the instantiation). The method was taken and modified from <em>scikit-fuzzy</em> package.

```
automf()
        
Return
---------
y : dict,
        Generated membership functions. The keys are the linguistic terms and the values are 1d arrays. 
```

The method returns a dictionary with the linguistic terms as the keys and the corresponding numerical intervals (in 1d arrays) as the values.

Example:

```
import numpy as np

mf = fcm.automf()

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

You can visualize this as follows:

```
import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 5))
axes = plt.axes()

for i in mf:
    axes.plot(fcm.universe, mf[i], linewidth=0.4, label=str(i))
    axes.fill_between(fcm.universe, mf[i], alpha=0.5)

axes.legend(bbox_to_anchor=(0.95, 0.6))

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
plt.tight_layout()
plt.show()
```

<img src="..\..\figures\figure_3.png" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
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
```
To activate the membership functions, we need to pass in the activation input along with the membership function. The activation input is a dictionary with the keys as the linguistic terms and the values as the frequency of the occurrences of the terms. As in the example presented in the background section, let's suppose that 1/6 of the experts expressed the causality between a given pair of concepts as Medium, 3/6 of the experts expressed it as High and 2/6 of the experts expressed it as VH. The respective activation input would look like this: {'M': 0.16, 'H': 0.5, 'VH': 0.33}.
The method returns a dictionary with the activated membership function.

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
Essentially, the values in the activation input determine the point at which the membership function of each linguistic term is going to be cut. 
You can visualize it as follows (Figure 4). 

```
fig = plt.figure(figsize= (10, 5))
axes = plt.axes()

for i in act:
    axes.plot(fcm.universe, act[i], linewidth=0.4, label=str(i))
    axes.fill_between(fcm.universe, act[i], alpha=0.5)

axes.legend(bbox_to_anchor=(0.95, 0.6))

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
axes.set_ylim([0,1])
plt.tight_layout()
plt.show()
```

<img src="..\..\figures\figure_4.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 4:</em> The activated membership function.

</div>

## aggregate()
<div align='justify'>
Now that we have activated the respective membership functions, we need to aggregate them before we can derive the causal weights through the defuzzification process.

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
axes.plot(fcm.universe, aggr, linewidth=0.4)
axes.fill_between(fcm.universe, aggr, alpha=0.5)

        
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
plt.ylim(0,1)

plt.tight_layout()
plt.show()
```

<img src="..\..\figures\figure_5.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 5:</em> The aggregated membership function.

</div>

## defuzzify()
<div align='justify'>

After aggregating the activated membership functions we can derive the causal weight between the given two concepts. The defuzzify() method uses the defuzz() method from python <em>scikit-fuzzy</em> package. 

```
defuzzify(universe, aggregated, method = 'centroid')

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
```

Example:

```
defuzz = fcm.defuzzify(aggr)
```
```
Output:

0.704
```
This can be visualized as follows (The code was taken and adjusted from  <a href="https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html">scikit-fuzzy tutorial)</a>:

```
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

<img src="..\..\figures\figure_6.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 6:</em> Derived edge weight (defuzzification).

</div>

## gen_weights()
<div align='justify'>

The methods described above are wrapped in gen_weights() method to allow for automatic generation of weight matrices based on qualitative inputs.

```
gen_weights(method = 'centroid')

Parameters
----------
method : str,
        Defuzzification method;  default --> 'centroid'. 
        For other defuzzification methods check scikit-fuzzy library (e.g., bisector, mom, sig)

```

Example:

```
fcm.gen_weights()
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

To use visualization methods developed for this module we first need to create an instance of FcmVisualize class and instantiate it with the fcm object we created earlier.

Example:

```
from fcmbci import FcmVisualize

vis = FcmVisualize(fcm)
```

One can inspect the frequency of the ratings of each linguistic term for a given pair of concepts with the term_freq_hist() method.

```
vis.term_freq_hist('C1', 'C2')
```
<img src="..\..\figures\figure_7.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 7:</em> Expert's ratings of causal strength between C1 and C2.

One can also visually inspect the deffuzification of the activated membership functions between a pair of concepts with the defuzz_view() method.

```
vis.defuzz_view('C1', 'C2')
```

<img src="..\..\figures\figure_8.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 8:</em> Deffuzification of the aggregated membership functions of concepts C1 and C2.

</div>
