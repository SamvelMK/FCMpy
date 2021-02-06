# Simulating an intervention using a Fuzzy Cognitive Map

## <a href="https://www.sciencedirect.com/science/article/pii/S2666521220300089?via%3Dihub"> Retrived from Mkhitaryan, Giabbanelli, Vries & Crutzen, 2020 </a>

<div align='justify'>
The existence of determinants and the weights of the causal edges are considered fixed or unchangeable: determinants cannot appear or cease to be relevant as an intervention unfolds, and the causal relation from one determinant to another is assumed to be constant during the intervention. In contrast, the values of nodes can change, thus reflecting how determinants in a case can be affected by the intervention. In other words, an FCM serves to update the value of the determinants under an intervention, based on a network structure. The results under the intervention are generally compared in relative terms to results produced by a no-intervention (i.e. status quo) situation. For example, by using FCM simulations of psychosocial determinants of obesity <a href="https://www.sciencedirect.com/science/article/abs/pii/S1568494612000634">[60]</a>, found that small interventions can lead to a 25% in obesity.

Once the update of an FCM has been determined (see <a src="..\..\simulator\simulator.md">documentation of the Simulator module</a>), the FCM can be used to compute the results of an intervention. Any intervention should be specified with respect to: (i) the baseline situation, that is, the value of all determinants and behaviors before the intervention takes place; and (ii) the design of the intervention, that is, what determinants and behaviors will directly be affected and with which causal strength. These two aspects can be handled by (i) providing the initial value of all nodes in the FCM, and (ii) adding the intervention as a new node of the FCM [28] or specifying which values would change at baseline because of the intervention.
</div>

Currently, the FcmBci package implements the testing of intervention cases by adding an intervention node to the FCM structure and specifying the causal impact it has on the target nodes (i.e., intervention targets). The intervention module inherits from the  <a src="..\..\simulator\simulator.md"><em>Simulator</em> module</a>. 

## Intervention

<div align='justify'>

To instantiate the Intervention class we need to pass on the initial state vector of the FCM concepts (<em>initial_state</em>), the connection matrix (<em>weights</em>) and specify the transfer function (<em>transfer</em>), the inference (<em>inference</em>) method, the error threshold (<em>thresh</em>) and the number of iterations for the simulation to run (<em>iterations</em>). One can also pass additional parameters that are required for the functions involved in the computations (e.g., l in the sigmoing function). 

```
Intervention(initial_state, weights, transfer, inference, thresh, iterations, **params)

    Parameters
    ----------
    initial_state: dict
                    keys ---> concepts, values ---> initial states of the associated concepts
    weights: panda.DataFrame
                causal weights between concepts
    transfer: str
                transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"
    inference: str
                inference method --> "kosko", "mKosko", "rescaled"
    thresh: float
                threshold for the error
    iterations: int
                    number of iterations
    params: additional parameters for the methods
```

Let's consider an FCM with eight concepts.

```
C1 = [0.0, 0.0, 0.6, 0.9, 0.0, 0.0, 0.0, 0.8]
C2 = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5]
C3 = [0.0, 0.7, 0.0, 0.0, 0.9, 0.0, 0.4, 0.1]
C4 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
C5 = [0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0, 0.3]
C6 = [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
C7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.9]
C8 =[0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.6, 0.0]

weight_mat = pd.DataFrame([C1,C2, C3, C4, C5, C6, C7, C8], 
                    columns=['C1','C2','C3','C4','C5','C6','C7','C8'])

init_state = {'C1': 1, 'C2': 1, 'C3': 0, 'C4': 0, 'C5': 0,
                    'C6': 0, 'C7': 0, 'C8': 0}
```

When we instantiate the Intervention class it automatically runs one cycle of simulation and stores the fixed point vector (if such is identified) as the baseline values for testing the intervention cases.

```
inter = Intervention(initial_state=init_state, weights=weights, transfer='sigmoid', inference='mKosko', 
               thresh=0.001, iterations=100, l=1)
```

```
Output:

The values converged in the 7 state (e <= 0.001)
```

One can inspect the basline vector in the test_results field as follows:

```
inter.test_results['baseline']
```

```
Output:

        C1        C2        C3        C4        C5        C6        C7        C8
0  1.000000  1.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
1  0.750260  0.731059  0.645656  0.710950  0.500000  0.500000  0.549834  0.785835
2  0.738141  0.765490  0.749475  0.799982  0.746700  0.769999  0.838315  0.921361
3  0.730236  0.784168  0.767163  0.812191  0.805531  0.829309  0.898379  0.950172
4  0.727059  0.789378  0.769467  0.812967  0.816974  0.838759  0.908173  0.954927
5  0.726125  0.790510  0.769538  0.812650  0.818986  0.839860  0.909707  0.955666
6  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774
```

</div>

## Methods

<div align='justify'>

The Intervention class implements the following methods to simulate interventions using FCMs.

- [add_intervention](#add_intervention)
- [remove_intervention](#remove_intervention)
- [test_intervention](#test_interventions)

</div>

<div align='justify'>

Now that we instantiated the Intervention class we can specify the inetrventions we would like to test by using the <em>add_intervention</em> method.

</dif>

## add_intervention()

<div align='justify'>

The add_intervention method requires one to specify the name of the intervention, the causal impact it has on the target nodes and the effectiveness of the intervention.

```
add_intervention(name, weights, effectiveness):

    Add an intervention node with the associated causal weights to the FCM.

    Parameters
    ----------
    name: str
            name of the intervention
    weights: dict
                keys ---> concepts the intervention impacts, value: the associated causal weight
    effectiveness: float
                    the degree to which the intervention was delivered (should be between [-1, 1])
```

Let's consider three such hypothetical intervention we wish to test in our FCM. The first intervention targets concepts (nodes) C1 and C2. It negatively impacts concept C1 (-.3) while positively impacting the concept C2 (.5). We consider a case where the intervention has maximum effectiveness (1). The other two interventions follow the same logic but impact other nodes (see below). 

```
inter.add_intervention('intervention_1', weights={'C1':-.3, 'C2' : .5}, effectiveness=1)
inter.add_intervention('intervention_2', weights={'C1':-.5}, effectiveness=1)
inter.add_intervention('intervention_3', weights={'C1':-1}, effectiveness=1)
```

One can also remove the added interventions by using the remove_intervention method.

</div>

## remove_intervention()

<div align='justify'>

To remove an intervention one has to pass on the name of the intervention that needs to be removed.

```
def remove_intervention(name):

    Remove intervention.

    Parameters
    ----------
    name: str
            name of the intervention
```

</div>

## test_interventions()

<div align='justify'>

Now that we have specified the interventions we can use the test_interventions method to test what will happen in each scenario.

```
test_intervention(name, iterations = None):
        
    Test an intervention case.

    Parameters
    ----------
    name: str
            name of the intervention
    iterations: number of iterations for the FCM simulation
                    default ---> the iterations specified in the init.
```
The test_intervention method requires the name of the intervention to be tested and the number of iterations it needs to run. If no argument is provided for the iterations the algorithm will use the iterations parameter specified when instantiating the Intervention class. 

```
inter.test_interventions('intervention_1')
inter.test_interventions('intervention_2')
inter.test_interventions('intervention_3')
```

```
Output:

The values converged in the 7 state (e <= 0.001)
The values converged in the 6 state (e <= 0.001)
The values converged in the 7 state (e <= 0.001)
The values converged in the 7 state (e <= 0.001)
```

The results of the tests can be accessed in the test_results field.

```
Output:

{'baseline':
        C1        C2        C3        C4        C5        C6        C7        C8
0  1.000000  1.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
1  0.750260  0.731059  0.645656  0.710950  0.500000  0.500000  0.549834  0.785835
2  0.738141  0.765490  0.749475  0.799982  0.746700  0.769999  0.838315  0.921361
3  0.730236  0.784168  0.767163  0.812191  0.805531  0.829309  0.898379  0.950172
4  0.727059  0.789378  0.769467  0.812967  0.816974  0.838759  0.908173  0.954927
5  0.726125  0.790510  0.769538  0.812650  0.818986  0.839860  0.909707  0.955666
6  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774, 

'intervention_1':
        C1        C2        C3        C4        C5        C6        C7        C8   intervention
0  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774           1.0
1  0.662298  0.861681  0.769410  0.812414  0.819328  0.839874  0.909973  0.955787           1.0
2  0.649547  0.869922  0.762564  0.803526  0.819327  0.839863  0.911132  0.955134           1.0
3  0.646000  0.870312  0.759929  0.800292  0.818413  0.838899  0.911143  0.954860           1.0
4  0.644962  0.870147  0.759059  0.799263  0.817925  0.838484  0.911052  0.954712           1.0
5  0.644651  0.870060  0.758786  0.798947  0.817735  0.838350  0.911004  0.954652           1.0, 

'intervention_2': 
        C1        C2        C3        C4        C5        C6        C7        C8   intervention
0  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774           1.0
1  0.616224  0.790728  0.769410  0.812414  0.819328  0.839874  0.909973  0.955787           1.0
2  0.589979  0.790727  0.757523  0.796898  0.819327  0.839863  0.909976  0.951931           1.0
3  0.582014  0.789347  0.752411  0.790490  0.817738  0.837922  0.909396  0.950725           1.0
4  0.579529  0.788521  0.750563  0.788232  0.816814  0.836988  0.909077  0.950265           1.0
5  0.578740  0.788168  0.749938  0.787481  0.816426  0.836656  0.908943  0.950094           1.0, 

'intervention_3':
        C1        C2        C3        C4        C5        C6        C7        C8   intervention
0  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774           1.0
1  0.493388  0.790728  0.769410  0.812414  0.819328  0.839874  0.909973  0.955787           1.0
2  0.435620  0.790727  0.743729  0.778417  0.819327  0.839863  0.909976  0.947229           1.0
3  0.417954  0.787737  0.732060  0.763231  0.815881  0.835586  0.908706  0.944288           1.0
4  0.412358  0.785865  0.727670  0.757568  0.813776  0.833355  0.907973  0.943124           1.0
5  0.410543  0.785032  0.726132  0.755597  0.812857  0.832501  0.907650  0.942677           1.0
6  0.409944  0.784709  0.725609  0.754931  0.812506  0.832208  0.907524  0.942513           1.0}
```

</div>