# Fuzzy Cognitive Maps for Behavior Change Interventions and Evaluation (FcmBci).

fcmbci is Python package for automatically generating causal weights for fuzzy cognitive maps based on qualitative inputs (by using fuzzy logic) and testing <em>what-if</em> scnearios. The package includes the following submodules:
* FcmDataProcessor
* FcmSimulator
* FcmVisualize

<a ref=fcmbci\data_processor\data_processor.md> FcmDatProcessor </a> includes methods for deriving causal weights of an FCM based on qualitative inputs from experts. <br> 
<a ref=fcmbci\simulator\simulator.md> FcmSimulator </a> allows runing simulations on top of the specified FCMs and test <em>what-if</em> scenarios. <br>
The FcmVisualize module provides methods for visualizing different components of data processing and simulations.

## Installation
Fcmbci requires python >=3.6 and depends on:

* pandas>=0.25.1
* numpy>=1.16.5
* scikit-fuzzy>=0.4.2
* networkx>=1.9.0
* matplotlib>=3.1.1

and will soon be available on PyPi! The lastest version can be installed by:

```
pip install fcmbci
```

Alternatively, you can install from source or develop this package, you can fork and clone this repository then install fcmbci by running:

```
python -m pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel
pip install e . # to install it in the current dir.
```

## Examples

Let's read in a sample input data and generate causal weights for the fcm and then run a simulation on top of this FCM.

```
from fcmbci import FcmDataProcessor, FcmVisualize, FcmSimulator
fcm = FcmDataProcessor()
```
Let's read in the data from unittests/test_cases.
```
fcm.read_xlsx('unittests/test_cases/sample_test.xlsx', 'Matrix')
```
we can access the data by fcm.data:
```
Output:
OrderedDict([('Expert_1',        C1    C2  C3   C4
                          C1    NaN   VH  NaN  NaN
                          C2    -VH   NaN NaN  NaN
                          C3     VH   NaN NaN   L
                          C4    NaN   NaN NaN  NaN), 
              ('Expert_2',       C1    C2  C3   C4
                          C1    NaN   VH   NaN  NaN
                          C2    -VH   NaN  NaN  NaN
                          C3     M    NaN   NaN  L
                          C4    NaN   NaN NaN  NaN)
```
Now let's automatically generate causal weights between the concepts using fuzzy logic.

```
fcm.gen_weights_mat()
```
Let's inspect the generated weights.

```
fcm.causal_weights
```
```
Output:
                    C1	          C2	C3	C4
      C1	0.000000	0.702205	0	0.000000
      C2	-0.610698	0.000000	0	0.000000
      C3	0.556908	0.000000	0	0.230423
      C4	0.000000	0.000000	0	0.000000
```

Now we can look take a look at the systems view. First we would need to create the system.

```
fcm.create_system()
```
Now we can use FcmVisualize submodule to inspect the system.

```
vis = FcmVisualize(fcm)
vis.system_view(target=['C1'])
```

<img src="figures\figure_9.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 1:</em> System view.

Now that we have defined our FCM, we are ready to run simulations on top of it. Let's first create the initial state vector.

```
init = {'C1' : 1, 'C2' : 1, 'C3' : 0, 'C4' : 0}
```
Now we are ready to create an instance of FcmSimulator class. We instantiate the FcmSimulator class by passing initial state vector and the causal weights as arguments.

```
sim = FcmSimulator(init, fcm.causal_weights)
```
```
Output:
      The values converged in the 7 state (e <= 0.001)
```
We can vizualize the results as follows.

```
vis.simulation_view(sim.scenarios, 'initial_state', network_view =True)
```

<img src="figures\figure_11.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 2:</em> Simulation results.

On the left hand side of the graph, you can see the network view of the FCM and on the left side you can see the results of the simulation steps. The color of the nodes in the network view indicates how active the node is (the more active the node is the darker is the node). When a hyperbolic tangent is used as a transfer function, the color scale is set to vary from blue to red indicating the range from [-1,1]. You can turn the network view off by setting the network_view parameter to False. 

Now, let's test a <em>what-if</em> scenario. To do this we will use test_scenario() method. We only need to specify which nodes are we going to turn on or off. The rest of the initial states will be taken from the final state of the simulation results we obtained when instantiating the FcmSimulator class. We also need to specify the name of the scenario that we are testing (this used is used to set the key in in the dictionary of the result). 

```
sim.test_scenario('scenario_1', {'C1' : 0, 'C2' : 1}, fcm.causal_weights)
```
```
Output:
      The values converged in the 7 state (e <= 0.001)
```
We can visualize this as follows:

```
vis.simulation_view(sim.scenarios, 'scenario_1', network_view =True, target=['C1','C2'])
```
<img src="figures\figure_12.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 10:</em> Simulation results (test scenario).

You can access the simulation results as follows:
```
sim.scenarios
```

```
Output:
{'initial_state':          C1        C2        C3        C4
                    0  1.000000  1.000000  0.000000  0.000000
                    1  0.596115  0.845823  0.500000  0.500000
                    2  0.588563  0.779786  0.622459  0.649129
                    3  0.612777  0.767291  0.650778  0.688380
                    4  0.624013  0.768095  0.657186  0.698114
                    5  0.627365  0.769640  0.658628  0.700472
                    6  0.628116  0.770331  0.658952  0.701036,
'scenario_1':              C1        C2        C3        C4
                    0  0.000000  1.000000  0.658952  0.701036
                    1  0.439369  0.731059  0.659025  0.701169
                    2  0.589023  0.738770  0.659041  0.701201
                    3  0.623606  0.759946  0.659045  0.701208
                    4  0.628675  0.768141  0.659046  0.701210
                    5  0.628690  0.770228  0.659046  0.701211
                    6  0.628396  0.770599  0.659046  0.701211}
```

## License

Please read LICENSE.txt in this directory.