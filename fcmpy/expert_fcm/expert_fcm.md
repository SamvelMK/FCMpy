# Methods for calculating numerical values based on qualitative input data using Fuzzy Logic.

## ExpertFcm: Constructing Expert-based Fuzzy Cognitive Maps

<div align='justify'>
The construction of the FCMs based on qualitative input data is a four step process. In the first step, one must define the fuzzy membership functions associated with the linguistic terms used to express causal relationships between the antecedent and consequent concepts in a given system. Fuzzy membership functions map the given linguistic terms to numerical values in a defined interval (a.k.a., the universe of discourse). In the second step, based on the frequency of the used linguistic terms, for a given pair of antecedent-consequent concepts one applies a fuzzy implication rule to "activate" the associated membership function. In the third step, one must aggregate all of the "activated" membership functions. Lastly, one applies a defuzzification method to derive a numerical value for the given antecedent-consequent pair.

We first present the implementation of the methods for each of the above mentioned steps. Then we conclude with a defined <em>build()</em> method that algorithmically applies these methods to a given dataset.
We will use the following general imports for the code examples:

```python
import numpy as np
import functools
import os

# for the visualizations
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
```


<b>Creating an instance of the ExpertFcm module</b>

We shall begin by creating an instance of the ExpertFcm class.

```python
from fcmpy import ExpertFcm

fcm = ExpertFcm()
```

<b>Step1: Fuzzy Membership Functions</b>

To generate the fuzzy membership functions, we must first define the universe of discourse (UD) that the fuzzy membership functions should map the associated linguistic terms to. For this demonstration we define the UD in the range of [-1, 1] with evenly spaced intervals of 0.05. One can also use a smaller spacing for higher resolution. We will use the ExpertFcm.universe() setter to add the corresponding output to the fcm object.

```python
fcm.universe = np.arange(-1, 1.05, .05)
```

```python
Output[1]:

array([-1.  , -0.95, -0.9 , -0.85, -0.8 , -0.75, -0.7 , -0.65, -0.6 ,
       -0.55, -0.5 , -0.45, -0.4 , -0.35, -0.3 , -0.25, -0.2 , -0.15,
       -0.1 , -0.05,  0.  ,  0.05,  0.1 ,  0.15,  0.2 ,  0.25,  0.3 ,
        0.35,  0.4 ,  0.45,  0.5 ,  0.55,  0.6 ,  0.65,  0.7 ,  0.75,
        0.8 ,  0.85,  0.9 ,  0.95,  1.  ])
```

Now that we have defined the UD, we can specify the linguistic terms that should be mapped to it. To do that, we need to decide on a geometric shape that would best represent the linguistic terms. In many applications, a triangular membership function is used. The triangular membership function requires one to specify the lower and the upper bounds of the triangle (i.e., where the meaning of the given linguistic term is represented the least) and the center of the triangle (i.e., where the meaning of the given linguistic term is fully expressed).

We will use the <em>ExpertFcm.linguistic_terms()</em> setter to set the linguistic terms and the associated parameters for the triangular membership function.

```python
fcm.linguistic_terms = {
                        '-VH': [-1, -1, -0.75],
                        '-H': [-1, -0.75, -0.50],
                        '-M': [-0.75, -0.5, -0.25], 
                        '-L': [-0.5, -0.25, 0],
                        '-VL': [-0.25, 0, 0],
                        'NA': [-0.001, 0, 0.001],
                        '+VL': [0, 0, 0.25],
                        '+L': [0, 0.25, 0.50],
                        '+M': [0.25, 0.5, 0.75],
                        '+H': [0.5, 0.75, 1],
                        '+VH': [0.75, 1, 1]
                        }
```

The keys in the above dictionary represent the linguistic terms and the values are lists that contain the parameters  for the triangular membership function (i.e., the lower bound, the center and the upper bound). In case if one decides to use a different membership function one would need to specify a list with the parameters required for a given function. 

Now that we know what type membership function we wish to use and have set the linguistic terms and the corresponding parameters, we can proceed to generating the membership functions. To do that we can use the <em>ExpertFcm.automf()</em> method and use the ExpertFcm.fuzzy_membership() setter to add the output to the fcm object.

```python
fcm.fuzzy_membership = fcm.automf(method='trimf')
```

```python
Output[2]:

{'-vh': array([1., 1., 1., ..., 0., 0., 0.]),
 '-h': array([0., 0., 0., ..., 0., 0., 0.]),
 '-m': array([0., 0., 0., ..., 0., 0., 0.]),
 '-l': array([0., 0., 0., ..., 0., 0., 0.]),
 '-vl': array([0., 0., 0., ..., 0., 0., 0.]),
 'na': array([0., 0., 0., ..., 0., 0., 0.]),
 '+vl': array([0., 0., 0., ..., 0., 0., 0.]),
 '+l': array([0., 0., 0., ..., 0., 0., 0.]),
 '+m': array([0., 0., 0., ..., 0., 0., 0.]),
 '+h': array([0., 0., 0., ..., 0., 0., 0.]),
 '+vh': array([0., 0., 0., ..., 1., 1., 1.])}
```

We can visualize the generated membership functions as follows:

```python
mfs = fcm.fuzzy_membership

fig = plt.figure(figsize= (10, 5))
axes = plt.axes()

for i in mfs:
    axes.plot(fcm.universe, mfs[i], linewidth=0.4, label=str(i))
    axes.fill_between(fcm.universe, mfs[i], alpha=0.5)

axes.legend(bbox_to_anchor=(0.95, 0.6))

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
plt.tight_layout()
plt.show()
```

![png](/figures/mfs.png)


In addition to the triangular membership functions, the ExpertFcm.automf() also implements gaussian membership functions ('gaussmf') and trapezoidal membership functions ('trapmf') (based on sci-kit fuzzy module in python).

```python

# Gaussian/ [mean, sigma]
fcm.linguistic_terms = {
                        '-VH': [-1, 0.1],
                        '-H': [-0.75, 0.1],
                        '-M': [-0.5, 0.1], 
                        '-L': [-0.25, 0.1],
                        '-VL': [0, 0.1],
                        'NA': [0, 0.001],
                        '+VL': [0, 0.1],
                        '+L': [0.25, 0.1],
                        '+M': [0.5, 0.1],
                        '+H': [0.75, 0.1],
                        '+VH': [1, 0.1]
                        }

mfs = fcm.automf(method='gaussmf')       
```

```python

fig = plt.figure(figsize= (10, 5))
axes = plt.axes()

for i in mfs:
    axes.plot(fcm.universe, mfs[i], linewidth=0.4, label=str(i))
    axes.fill_between(fcm.universe, mfs[i], alpha=0.5)

axes.legend(bbox_to_anchor=(0.95, 0.6))

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
plt.tight_layout()
plt.show()
```


<b>Step 2: Fuzzy Implications</b>

To determine the level of activation of the linguistic terms for a given pair of concepts one must first identify the level of endorsement of the given terms by the participants. This is done by calculating the proportion of the answers to each linguistic term for a given antecedent-consequent. Consider, a case where 50% of the participants (e.g., domain experts) rated the causal impact of an antecedent on the consequent as Positive High, 33% rated it as Positive Very High and the 16% rated it as Positive Medium. Subsequently, a fuzzy implication rule is used to "activate" the corresponding membership functions. Two such rules are often used, namely Mamdani's minimum and Larsen's product implication rule. Two such rules are often used, namely Mamdani's minimum and Larsen's product implication rule. 

The <a href="https://link.springer.com/chapter/10.1007%2F978-3-642-25859-6_4">Mamdani minimum</a> fuzzy implication rule is expressed as:

$$
\mu_{R}(x,y)= min \left \lfloor \mu_{A}(x), \mu_{B}(y) \right \rfloor
$$ 

We can use the <em>ExpertFcm.fuzzy_implication()</em> method to apply the implication rules.

```python
mfs = fcm.fuzzy_membership

act_pvh = fcm.fuzzy_implication(mfs['+VH'], weight= 0.33, method ='Mamdani')
act_ph = fcm.fuzzy_implication(mfs['+H'], weight=0.5, method ='Mamdani')
act_pm = fcm.fuzzy_implication(mfs['+M'], weight=0.16, method ='Mamdani')

activatedMamdani = {'+vh' : act_pvh, '+h' : act_ph, '+m' : act_pm}
```

```python
Output[3]

{'+vh': array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.2 , 0.33, 0.33, 0.33, 0.33]),
 '+h': array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. , 0. , 0. , 0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4,
        0.2, 0. ]),
 '+m': array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  , 0.  , 0.  , 0.  , 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
        0.16, 0.16, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])}
```

The results can be visualized as follows:

```python
fig = plt.figure(figsize= (10, 5))
axes = plt.axes()

for i in activatedMamdani:
    axes.plot(fcm.universe, activatedMamdani[i], linewidth=0.4, label=str(i))
    axes.fill_between(fcm.universe, activatedMamdani[i], alpha=0.5)

axes.legend(bbox_to_anchor=(0.95, 0.6))

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
axes.set_ylim([0,1])
```

![png](/figures/mamdani.png)


As you can see in the figure above, the corresponding membership functions were cut at the level of endorsement. In contrast, the Larsen's product rule rescales the corresponding membership functions based on the level of endorsement of each term. <a href="https://link.springer.com/chapter/10.1007%2F978-3-642-25859-6_4"> Larsen's</a> product rule can be expressed as:

$$
\mu_{R}(x,y)= \mu_{A}(x)\cdot \mu_{B}(y)
$$ 

```python
mfs = fcm.fuzzy_membership

act_pvh = fcm.fuzzy_implication(mfs['+VH'], weight= 0.33, method ='Larsen')
act_ph = fcm.fuzzy_implication(mfs['+H'], weight=0.5, method ='Larsen')
act_pm = fcm.fuzzy_implication(mfs['+M'], weight=0.16, method ='Larsen')

activatedLarsen = {'+vh' : act_pvh, '+h' : act_ph, '+m' : act_pm}
```

```python
Output[4]

{'+vh': array([0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.066, 0.132, 0.198, 0.264, 0.33 ]),
 '+h': array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. , 0. , 0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2,
        0.1, 0. ]),
 '+m': array([0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.032,
        0.064, 0.096, 0.128, 0.16 , 0.128, 0.096, 0.064, 0.032, 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   ])}
```

```python
fig = plt.figure(figsize= (10, 5))
axes = plt.axes()

for i in activatedLarsen:
    axes.plot(fcm.universe, activatedLarsen[i], linewidth=0.4, label=str(i))
    axes.fill_between(fcm.universe, activatedLarsen[i], alpha=0.5)

axes.legend(bbox_to_anchor=(0.95, 0.6))

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
axes.set_ylim([0,1])
```

![png](/figures/larsen.png)


<b>Step 3: Aggregate the Activated Membership Functions</b>

Now that we have activated the respective membership functions, we need to aggregate them before we can derive the causal weights through the defuzzification process. There are several aggregation methods, namely, $f(x, y)=x + y - x \times y$, the family Einstein Sum $f(x, y) = (x + y)/(1 + x \times y)$ and the family Hamacher Sum $f(x, y) = (x + y - 2 \times x \times y)/(1 - x \times y)$. To aggregate the activated membership functions, we can use the <em>ExpertFcm.aggregate()</em> method.

```python
aggregated = functools.reduce(lambda x,y: fcm.aggregate(x=x, y=y, method='fMax'),
                        [activatedLarsen[i] for i in activatedLarsen.keys()])
```

```python
Output[5]

array([0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.032,
       0.064, 0.096, 0.128, 0.16 , 0.128, 0.2  , 0.3  , 0.4  , 0.5  ,
       0.4  , 0.3  , 0.2  , 0.264, 0.33 ])
```

The results can be visualized as follows:

```python
fig = plt.figure(figsize= (10, 5))
axes = plt.axes()
axes.plot(fcm.universe, aggregated, linewidth=0.4)
axes.fill_between(fcm.universe, aggregated, alpha=0.5)

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
plt.ylim(0,1)
plt.tight_layout()
```

![png](/figures/aggregated.png)

<b> Step 4: Defuzzification</b>

Now that we have aggregated the activated membership functions we can apply one of the defuzzification methods to derive the numerical value (i.e., crisp value) for the casual impact of the given antecedent-consequent pair. Among the available defuzzification methods (e.g., bisector, mean of maximum etc.) the most commonly used method is the centroid method (a.k.a. center of gravity). The centroid method for continuous membership functions can be expressed mathematically as:

$$
x^{*} = \frac{\int\mu_{B}(x) \cdot xdx}{\int\mu_{B}(x)dx}
$$

We can apply the dedicated <em>ExpertFcm.defuzz()</em> method to derive the crisp value.

```python
dfuz = fcm.defuzz(x=fcm.universe, mfx=aggregated, method='centroid')
```

```python
Output[5]

0.72...
```

We can visualize all the results in one graph as follows:

```python
fig = plt.figure(figsize=(10, 5))
axes = plt.axes()
legend_anchor=(0.95, 0.6)

import skfuzzy as fuzz

mfs = fcm.fuzzy_membership

# for plot
y_activation = fuzz.interp_membership(fcm.universe, aggregated, dfuz)  
out = np.zeros_like(fcm.universe) 

for i in mfs:
    axes.plot(fcm.universe, mfs[i], linewidth=0.3, label=str(i)) # plots all the mfs. 
    axes.fill_between(fcm.universe, mfs[i], alpha=0.4)

    axes.fill_between(fcm.universe, out, 
                            aggregated, facecolor='#36568B', alpha=1)

    axes.plot([dfuz, dfuz], [0, y_activation], 'k', linewidth=1.5, alpha=1)

    axes.legend(bbox_to_anchor=legend_anchor)
    axes.text(dfuz-0.05, y_activation+0.02, f'{round(dfuz, 2)}',
                    bbox=dict(facecolor='red', alpha=0.1))

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().tick_left()
plt.tight_layout()
```

![png](/figures/overall.png)


<b>Build FCMs Based on Qualitative Datasets</b>

Now that we have explored how each part of the FCM construction works, we can define the steps one would use to construct FCMs based on qualitative datasets. Data on FCMs are often collected via the means of surveys. During these surveys, the participants are asked to identify the antecedent consequent concepts involved in the system and then rate the causal relationship between these concepts. Consequently, the data includes the concept pairs and the associated linguistic ratings of the participants. Since the data may come in different file formats. The ExpertFcm class provides methods for reading data from .csv, .xlsx and .json files. The corresponding files should follow certain requirements. We will describe the requirements for each file type separately.

<b>CSV</b>

The csv file should have the following general structure:

![png](/figures/csv.png)

Each <em>column</em> represents a pair of connected concepts. The column heads should follow the following format: antecedent sepConcept consequent (polarity) (e.g., 'C1 -> C2 (+)').
If the pattern is not detected the method will throw an error.
Each <em>row</em> in the file represents the inputs of an expert and each cell of the file represents a linguistic term expressing causality between the respective concept pairs.

<b>XLSX</b>

The .xlsx files must have the following columns:
* From: Representing antecedents
* To: Representing consequent
* L(n): Representing linguistic term(s) n (e.g., VL, L, M etc.).

![png](/figures/xlsx.png)

<b>JSON</b>

The .json file must have the following general structure:

```python
{"Expert_1": [
    {"from": "C1", "to": "C1", "NA": "", "VL": "", "L": "", "M": 1, "H": "", "VH": "", "-VL": "", "-L": "", "-M": "", "-H": "", "-VH": ""}, 
    {"from": "C1", "to": "C2", "NA": "", "VL": "", "L": "", "M": "", "H": "", "VH": "", "-VL": 1, "-L": "", "-M": "", "-H": "", "-VH": ""}
 ],
 "Expert_2": [
   {"from": "C1", "to": "C1", "NA": "", "VL": "", "L": 1, "M": "", "H": "", "VH": "", "-VL": "", "-L": "", "-M": "", "-H": "", "-VH": ""}, 
   {"from": "C1", "to": "C2", "NA": "", "VL": 1, "L": "", "M": "", "H": "", "VH": "", "-VL": "", "-L": "", "-M": "", "-H": "", "-VH": ""}
 ]
}
```

We can use the dedicated <em>ExpertFcm.read_data()</em> method to read the data. Depending on the file extension one must specify the following arguments:

```python    
Parameters
----------
file_path : str 

Other Parameters
----------------
for .csv files:

        **sep_concept: str,
                        separation symbol (e.g., '->') that separates the antecedent from the consequent in the column heads of a csv file
                        default ---> '->'

        **csv_sep: str,
                separator of the csv file
                default ---> ','
                
for .xlsx files:

        **check_consistency: Bool
                        check the consistency of ratings across the experts
                        default --> False

        **engine: str,
                the engine for excel reader (read more in pd.read_excel)
                default --> "openpyxl"

for .json files:

        **check_consistency: Bool
                                check the consistency of ratings across the experts.
                                default --> False

Return
-------
data: collections.OrderedDict
        ordered dictionary with the formatted data.
```

For the .xlsx and .json file there is an option to specify the <em>check_consistency</em> argument. If set to True then the algorithm checks whether the experts rated the causal impact of the antecedent-consequent pairs consistently in terms of the valence of the causal impact (i.e., positive or negative causality). It writes out a respective .xlsx file if such inconsistencies are identified.

The ExpertFcm.read_data() method returns an ordered dictionary where the <em>keys</em> are the expert ids (i.e., the names of the excel sheets) and the <em>values</em> are panda's dataframes of the expert inputs.

```python
data = fcm.read_data(file_path= os.path.abspath('unittests/test_cases/data_test.csv'), 
                      sep_concept='->', csv_sep=';')
```

```python
Output[6]

OrderedDict([('Expert0',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To
              0    1   0   0   0    0   0    0   0   0   0    0   C1  C2
              1    0   0   0   0    0   0    0   0   0   1    0   C2  C1
              2    0   0   0   0    0   0    0   0   0   1    0   C3  C1
              3    0   0   0   0    0   0    1   0   0   0    0   C3  C4),
             ....
             ('Expert5',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To  no causality
              0    0   0   1   0    0   0    0   0   0   0    0   C1  C2           0.0
              1    0   0   0   0    0   0    0   0   1   0    0   C2  C1           0.0
              2    0   0   0   0    0   0    0   0   0   0    0   C3  C1           1.0
              3    0   0   0   0    0   0    0   0   0   0    0   C3  C4           1.0)])

```

It is often interesting to check the extent to which the participants agree on their opinions. It is often done by calculating the information entropy. The entropy for a given pair of concepts is expressed as:

$$
R=-\sum_{i=1}^{n}p_i log_2(p_i)
$$

Where $p_i$ is the proportion of the answers (per linguistic term) about the causal relationship between a given antecedent-consequent pair. For this, one can use the dedicated <em>ExpertFcm.entropy()</em> method.

```python
entropy = fcm.entropy(data)
```

```python
Output[7]

            Entropy
From	To	
C1	    C2	1.459148
C2	    C1	1.459148
C3	    C1	1.251629
        C4	1.459148
```

Before using the ExpertFcm.build() method to compute the FCM connection matrix we must follow the first step described previously. Namely, define the UD, supply the linguistic terms, and generate the fuzzy membership functions.

The ExpertFcm.build() method takes the following arguments:

```python
Parameters
----------
data: collections.OrderedDict
        ordered dictionary with the qualitative input data.

implication_method: str,
                        implication rule; at the moment two such rules are available;
                        'Mamdani': minimum implication rule
                        'Larsen': product implication rule
                        default ---> 'Mamdani'

aggregation_method: str,
                        aggregation rule; at the moment four such rules are available:
                        'fMax': family maximum,
                        'algSum': family Algebraic Sum,
                        'eSum': family Einstein Sum,
                        'hSum': family Hamacher Sum
                        default ---> 'fMax'

defuzz_method: str,            
                defuzzification method; at the moment four such rules are available:
                'centroid': Centroid,
                'bisector': Bisector,
                'mom': MeanOfMax,
                'som': MinOfMax,
                'lom' : MaxOfMax
                default ---> 'centroid'

Return
-------
y: pd.DataFrame
        the connection matrix with the defuzzified values
```

```python
weight_matrix = fcm.build(data=data, implication_method='Larsen')
```

```python
Output[8]

        C2      	C1	        C4
C2	0.000000	0.610116	0.000000
C3	0.000000	0.541304	0.130328
C1	-0.722442	0.000000	0.000000
```