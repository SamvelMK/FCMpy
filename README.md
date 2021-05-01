# FCMpy: A package for Constructing and Analysing Fuzzy Cognitive Maps in Python.
<div align = justify>

The fcmpy is Python package for automatically generating causal weights for fuzzy cognitive maps based on qualitative inputs (by using fuzzy logic), optimizing the FCM connection matrix via Machine Learning Algorithms and testing <em>what-if</em> scenarios. The package includes the following submodules:

* ExpertFcm
* Simulation
* Intervention
* ---> (ML TBA)

<a href="fcmpy\expert_based_fcm\expert_fcm.md"> The ExpertFcm module </a> includes methods for deriving causal weights of an FCM based on qualitative data. <br> 
<a href="fcmpy\simulator\simulator.md"> The FcmSimulator module </a> provides methods for runing simulations on top of a given FCM structure. <br>
<a href="fcmpy\intervention\intervention.md"> The FcmIntervention module </a> allows testing what-if scenarios on top of the specified FCMs. <br>

## Installation
FCMpy requires python >=3.8.1 and depends on:

* pandas>=1.0.3
* numpy>=numpy==1.18.2
* scikit-fuzzy>=0.4.2
* tqdm>=4.50.2
* openpyxl

and will soon be available on PyPi! The lastest version can be installed by:

```
pip install fcmpy
```

Alternatively, you can install it from source or develop this package, you can fork and clone this repository then install FCMpy by running:

```
py -m pip install --user --upgrade setuptools wheel
py setup.py sdist bdist_wheel
py -m pip install install e . 
```

You can run the unittest for the package as follows:

```
py -m unittest discover unittests
```

## Examples

Below we present the fast implementation of the library. More specifically, how to construct an FCM based on the supplied data using Fuzzy Logic, run simulations on a defined FCMs and test interventions on top of the defined FCM structure.

### Building FCMs based on qualitative inputs using Fuzzy Logic

<b>Step 1: Generate Fuzzy Membership Functions</b>

Here we generate triangular fuzzy membership functions for 11 linguistic terms.

```Python
from fcmpy import ExpertFcm, FcmSimulator, FcmIntervention 

fcm = ExpertFcm()

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

fcm.universe = np.arange(-1, 1.05, .05)

fcm.fuzzy_membership = fcm.automf(method='trimf')
```

Let's take a look at the generated membership functions.

```Python
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
```

![png](/figures/mfs.png)

<b>Step 2: Build FCMs based on qualitative input data using Fuzzy Logic</b>


* Read data from a csv file.

```Python
data = fcm.read_data(file_path= os.path.abspath('unittests/test_cases/data_test.csv'), 
                      sep_concept='->', csv_sep=';')
```

```Python
Output[1]

OrderedDict([('Expert0',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To
              0    1   0   0   0    0   0    0   0   0   0    0   C1  C2
              1    0   0   0   0    0   0    0   0   0   1    0   C2  C1
              2    0   0   0   0    0   0    0   0   0   1    0   C3  C1
              3    0   0   0   0    0   0    1   0   0   0    0   C3  C4),
             ('Expert1',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To
              0    1   0   0   0    0   0    0   0   0   0    0   C1  C2
              1    0   0   0   0    0   0    0   0   0   0    1   C2  C1
              2    0   0   0   0    0   0    0   0   1   0    0   C3  C1
              3    0   0   0   0    0   0    0   1   0   0    0   C3  C4),
             ('Expert2',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To
              0    0   1   0   0    0   0    0   0   0   0    0   C1  C2
              1    0   0   0   0    0   0    0   0   0   1    0   C2  C1
              2    0   0   0   0    0   0    0   0   1   0    0   C3  C1
              3    0   0   0   0    0   0    1   0   0   0    0   C3  C4),
             ('Expert3',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To
              0    0   1   0   0    0   0    0   0   0   0    0   C1  C2
              1    0   0   0   0    0   0    0   0   1   0    0   C2  C1
              2    0   0   0   0    0   0    0   0   1   0    0   C3  C1
              3    0   0   0   0    0   0    1   0   0   0    0   C3  C4),
             ('Expert4',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To  no causality
              0    0   1   0   0    0   0    0   0   0   0    0   C1  C2           0.0
              1    0   0   0   0    0   0    0   0   1   0    0   C2  C1           0.0
              2    0   0   0   0    0   0    0   0   1   0    0   C3  C1           0.0
              3    0   0   0   0    0   0    0   0   0   0    0   C3  C4           1.0),
             ('Expert5',
                 -vh  -h  -m  -l  -vl  na  +vl  +l  +m  +h  +vh From  To  no causality
              0    0   0   1   0    0   0    0   0   0   0    0   C1  C2           0.0
              1    0   0   0   0    0   0    0   0   1   0    0   C2  C1           0.0
              2    0   0   0   0    0   0    0   0   0   0    0   C3  C1           1.0
              3    0   0   0   0    0   0    0   0   0   0    0   C3  C4           1.0)])
```

* Calculate the entropy of the expert raitings.

```Python
entropy = fcm.entropy(data)
```

```Python
Output[2]

Entropy
From	To	
C1	    C2	1.459148
C2	    C1	1.459148
C3	    C1	1.251629
        C4	1.459148
```

* Build FCM connection matrix

Here we build FCM based on the qualitative input data using Larsen's implication method, family maximum aggregation method and the centropid difuzzification method.

```Python
weight_matrix = fcm.build(data=data, implication_method='Larsen')
```

```Python
Output[3]
        C2	        C1      	C4
C2	0.000000	0.610116	0.000000
C3	0.000000	0.541304	0.130328
C1	-0.722442	0.000000	0.000000
```

### Run simulations on top of a defined FCM structure

In this example we will replicate the case presented in the fcm inference package in R by Dikopoulou & Papageorgiou

* Instantiate and FcmSimulator class

```Python
sim = FcmSimulator()
```

* Define the FCM structure

```Python
import pandas as pd

C1 = [0.0, 0.0, 0.6, 0.9, 0.0, 0.0, 0.0, 0.8]
C2 = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5]
C3 = [0.0, 0.7, 0.0, 0.0, 0.9, 0.0, 0.4, 0.1]
C4 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
C5 = [0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0, 0.3]
C6 = [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
C7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.9]
C8 =[0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6, 0.0]

weight_matrix = pd.DataFrame([C1,C2, C3, C4, C5, C6, C7, C8], 
                    columns=['C1','C2','C3','C4','C5','C6','C7','C8'])
```

* Define the initial state vector

```Python
init_state = {'C1': 1, 'C2': 1, 'C3': 0, 'C4': 0, 'C5': 0,
                    'C6': 0, 'C7': 0, 'C8': 0}
```

* Simulate

Here we run a simulation on top of the defined FCM structure using the sigmoid transfer function and the modified kosko's inference method. The simulation will run $50$ iterations and will stop if the absoulte difference between the concept values between the simulation steps is $\leq 0.001$. The steepness parameter for the sigmoid function is set to $1$. 

```Python
res_mK = sim.simulate(initial_state=init_state, weight_matrix=weight_matrix, transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1)
```
```Python
Outout[4]

The values converged in the 7 state (e <= 0.001)
```

```Python
Output[5]

        C1	        C2          C3	        C4	        C5	        C6	        C7	        C8
0	1.000000	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
1	0.750260	0.731059	0.645656	0.710950	0.500000	0.500000	0.549834	0.785835
2	0.738141	0.765490	0.749475	0.799982	0.746700	0.769999	0.838315	0.921361
3	0.730236	0.784168	0.767163	0.812191	0.805531	0.829309	0.898379	0.950172
4	0.727059	0.789378	0.769467	0.812967	0.816974	0.838759	0.908173	0.954927
5	0.726125	0.790510	0.769538	0.812650	0.818986	0.839860	0.909707	0.955666
6	0.725885	0.790706	0.769451	0.812473	0.819294	0.839901	0.909940	0.955774
```

```Python
plt.figure()
res_mK.plot(figsize=(15, 10))
plt.legend(bbox_to_anchor=(0.97, 0.94))

plt.xlabel('Simulation Steps')
plt.ylabel('Initial States')
plt.show()
```

![png](/figures/simulations.png)

<em>Figure 1:</em> The results of the FCM simulation. <br>

### Test Interventions on top of the defined FCMs

Here we will use the same initial state and the weight matrix defined in the previous example.
Let's first create an instance of the FcmIntervention class. To do so we need to pass an fcmpy Simulator object.

```
inter = FcmIntervention(FcmSimulator)
```

Now we need to create a baseline for testing our interventions. We do so my using the FcmIntervention.initialize() method.

```Python
inter.initialize(initial_state=init_state, weight_matrix=weight_matrix, 
                        transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1)
```

```Python
Output[6]

The values converged in the 7 state (e <= 0.001)
```

This should already be familiar from the previous example. Here we just run a simulation on top of the a defined FCM (where no intervention exists) with a given vector of initial conditions. The baseline of comparison is the derived equilibrium states of the concepts in the FCM.

Now we can specify the interventions that we want to test.
Let's consider three such hypothetical interventions we wish to test in our FCM. The first intervention targets concepts (nodes) C1 and C2. It negatively impacts concept C1 (-.3) while positively impacting the concept C2 (.5). We consider a case where the intervention has maximum effectiveness (1). The other two interventions follow the same logic but impact other nodes (see below). 

```Python
inter.add_intervention('intervention_1', impact={'C1':-.3, 'C2' : .5}, effectiveness=1)
inter.add_intervention('intervention_2', impact={'C4':-.5}, effectiveness=1)
inter.add_intervention('intervention_3', impact={'C5':-1}, effectiveness=1)
```

Now we can use the FcmIntervention.test_intervention() method to test the intervention cases.

```Python
inter.test_intervention('intervention_1')
inter.test_intervention('intervention_2')
inter.test_intervention('intervention_3')
```

```Python
Output[6]

The values converged in the 6 state (e <= 0.001)
The values converged in the 6 state (e <= 0.001)
The values converged in the 6 state (e <= 0.001)
```

We can look at the results of the simulation runs of each intervention case as follows:

```Python
inter.test_results['intervention_1']
```

```Python
Output[7]

        C1	        C2	        C3	        C4	        C5	        C6	        C7	        C8	intervention
0	0.725885	0.790706	0.769451	0.812473	0.819294	0.839901	0.909940	0.955774	1.0
1	0.662298	0.861681	0.769410	0.812414	0.819328	0.839874	0.909973	0.955787	1.0
2	0.649547	0.869922	0.762564	0.803526	0.819327	0.839863	0.911132	0.955134	1.0
3	0.646000	0.870312	0.759929	0.800292	0.818413	0.838899	0.911143	0.954860	1.0
4	0.644962	0.870147	0.759059	0.799263	0.817925	0.838484	0.911052	0.954712	1.0
5	0.644651	0.870060	0.758786	0.798947	0.817735	0.838350	0.911004	0.954652	1.0
```

Now we can inspect the equilibrium states of the concepts in each intervention case.

```Python
inter.equilbriums
```

```Python
Ouput[8]

    baseline    intervention_1  intervention_2  intervention_3
C1  0.725885        0.644651        0.715704        0.723417
C2  0.790706        0.870060        0.790580        0.790708
C3  0.769451        0.758786        0.768132        0.769141
C4  0.812473        0.798947        0.699316        0.812073
C5  0.819294        0.817735        0.819160        0.563879
C6  0.839901        0.838350        0.823430        0.871834
C7  0.909940        0.911004        0.909917        0.909778
C8  0.955774        0.954652        0.955427        0.952199
```

Lastly, we can inspect the differences between the interventions in relative terms (i.e., % increase or decrease) compared to the baseline.

```Python
inter.comparison_table
```

```Python
Ouput[9]

    baseline    intervention_1  intervention_2  intervention_3
C1       0.0      -11.191083       -1.402511       -0.339981
C2       0.0       10.035821       -0.015968        0.000202
C3       0.0       -1.385998       -0.171325       -0.040271
C4       0.0       -1.664794      -13.927524       -0.049314
C5       0.0       -0.190233       -0.016379      -31.175022
C6       0.0       -0.184640       -1.960979        3.802010
C7       0.0        0.116873       -0.002543       -0.017806
C8       0.0       -0.117365       -0.036331       -0.374038
```

## License

Please read LICENSE.txt in this directory.

</div>