# Simulating an intervention using a Fuzzy Cognitive Map

## <a href="https://www.sciencedirect.com/science/article/pii/S2666521220300089?via%3Dihub"> Retrived from Mkhitaryan, Giabbanelli, Vries & Crutzen, 2020 </a>

<div align='justify'>
Once the update function of an FCM has been specified (see <a src="..\..\simulator\simulator.md">documentation of the FcmSimulator module</a>), the FCM can be used to compute the results of an intervention. Any intervention should be specified with respect to: (i) the baseline situation, that is, the value of all concepts of the FCM before the intervention takes place; and (ii) the design of the intervention (i.e., the structure of the FCM). These two aspects can be handled by (i) providing the initial values of all nodes in the FCM, and (ii) adding the intervention as a new node of the FCM [28].
</div>

Currently, the fcmpy package implements the testing of intervention cases by adding an intervention node to the FCM structure and specifying the causal impact it has on the target nodes (i.e., intervention targets).  

## Intervention

<div align='justify'>

To instantiate the FcmIntervention class we need to pass an fcmpy Simulator object. In the current version the fcmpy implements one type of simulation but in the future new simulation types could be added (e.g., fuzzy gray maps, fuzzy cognitive networks). 

```Python
from fcmpy import FcmSimulator, FcmIntervention

inter = FcmIntervention(FcmSimulator)
```

Let's consider an FCM with eight concepts.

```Python
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

init_state = {'C1': 1, 'C2': 1, 'C3': 0, 'C4': 0, 'C5': 0,
                    'C6': 0, 'C7': 0, 'C8': 0}
```
Before specifying intervention cases and running simulations for each scenario, we need to create the baseline for the comparison (i.e., run a simulation with baseline initial conditions and take the converged state vector). To do this one needs to call <em>FcmIntervention.initialize()</em> method

```Python
inter.initialize(initial_state=init_state, weight_matrix=weight_matrix, 
                        transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1)
```

```Python
Output[1]

The values converged in the 7 state (e <= 0.001)
```

One can inspect the results of the initial simulation run (i.e., 'baseline') in the test_results field as follows:

```Python
inter.test_results['baseline']
```

```Python
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

The FcmIntervention class implements the following methods.

- [add_intervention](#add_intervention)
- [remove_intervention](#remove_intervention)
- [test_intervention](#test_interventions)

</div>

<div align='justify'>

To specify an intervention case one need to pass a <em>label</em>, a dictionary (<em>weights</em>) where the keys represent the concepts the intervention has an impact on and the values represent the magnitude of that impact. Lastly, one must specify the effectiveness of the intervention (i.e., <em>effectiveness</em>).

</dif>

## add_intervention()

<div align='justify'>

The add_intervention method requires one to specify the name of the intervention, the causal impact it has on the target nodes and the effectiveness of the intervention.

```Python
add_intervention(name, weights, effectiveness):

    Add an intervention node with the associated causal weights to the FCM.

    Parameters
    ----------
    name: str
            name of the intervention
    impact: dict
                keys ---> concepts the intervention impacts, value: the associated causal weight
    effectiveness: float
                    the degree to which the intervention was delivered (should be between [-1, 1])
```

Let's consider three such hypothetical interventions we wish to test in our FCM. The first intervention targets concepts (nodes) C1 and C2. It negatively impacts concept C1 (-.3) while positively impacting the concept C2 (.5). We consider a case where the intervention has maximum effectiveness (1). The other two interventions follow the same logic but impact other nodes (see below). 

```Python
inter.add_intervention('intervention_1', weights={'C1':-.3, 'C2' : .5}, effectiveness=1)
inter.add_intervention('intervention_2', weights={'C4':-.5}, effectiveness=1)
inter.add_intervention('intervention_3', weights={'C5':-1}, effectiveness=1)
```

One can also remove the added interventions by using the [remove_intervention](#remove_intervention) method.

</div>

## remove_intervention()

<div align='justify'>

To remove an intervention one has to pass on the name of the intervention that needs to be removed.

```Python
def remove_intervention(name):

    Remove intervention.

    Parameters
    ----------
    name: str
            name of the intervention
```

</div>

## test_intervention()

<div align='justify'>

Now that we have specified the interventions we can use the test_interventions method to test what will happen in each scenario.

```Python
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

```Python
inter.test_intervention('intervention_1')
inter.test_intervention('intervention_2')
inter.test_intervention('intervention_3')
```

```Python
Output:

The values converged in the 7 state (e <= 0.001)
The values converged in the 6 state (e <= 0.001)
The values converged in the 7 state (e <= 0.001)
The values converged in the 7 state (e <= 0.001)
```

The results of the tests can be accessed in the <em>test_results</em> field.

```Python
Output:

{'baseline':          C1        C2        C3        C4        C5        C6        C7         C8
                0  1.000000  1.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
                1  0.750260  0.731059  0.645656  0.710950  0.500000  0.500000  0.549834  0.785835
                2  0.738141  0.765490  0.749475  0.799982  0.746700  0.769999  0.838315  0.921361
                3  0.730236  0.784168  0.767163  0.812191  0.805531  0.829309  0.898379  0.950172
                4  0.727059  0.789378  0.769467  0.812967  0.816974  0.838759  0.908173  0.954927
                5  0.726125  0.790510  0.769538  0.812650  0.818986  0.839860  0.909707  0.955666
                6  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774
 
 'intervention_1':              C1        C2        C3        C4        C5        C6        C7      C8    intervention
                        1  0.662298  0.861681  0.769410  0.812414  0.819328  0.839874  0.909973  0.955774     1.0
                        0  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955787     1.0
                        2  0.649547  0.869922  0.762564  0.803526  0.819327  0.839863  0.911132  0.955134     1.0
                        3  0.646000  0.870312  0.759929  0.800292  0.818413  0.838899  0.911143  0.954860     1.0
                        4  0.644962  0.870147  0.759059  0.799263  0.817925  0.838484  0.911052  0.954712     1.0
                        5  0.644651  0.870060  0.758786  0.798947  0.817735  0.838350  0.911004  0.954652     1.0,
 
 'intervention_2':          C1        C2        C3        C4        C5        C6        C7          C8    intervention
                        0  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774     1.0
                        1  0.725827  0.790728  0.769410  0.724276  0.819328  0.839874  0.909973  0.955787     1.0
                        2  0.718741  0.790727  0.769396  0.706308  0.819327  0.828905  0.909976  0.955787     1.0
                        3  0.716516  0.790726  0.768639  0.701233  0.819325  0.825022  0.909976  0.955547     1.0
                        4  0.715883  0.790638  0.768266  0.699748  0.819224  0.823795  0.909939  0.955458     1.0
                        5  0.715704  0.790580  0.768132  0.699316  0.819160  0.823430  0.909917  0.955427     1.0,
 
 'intervention_3':          C1        C2        C3        C4        C5        C6        C7          C8    intervention
                        0  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774     1.0  
                        1  0.725827  0.790728  0.769410  0.812414  0.625228  0.839874  0.909973  0.955787     1.0  
                        2  0.725813  0.790727  0.769396  0.812397  0.578763  0.861988  0.909976  0.953260     1.0  
                        3  0.724436  0.790726  0.769393  0.812392  0.567392  0.869389  0.909852  0.952521     1.0  
                        4  0.723702  0.790725  0.769245  0.812202  0.564598  0.871358  0.909801  0.952278     1.0  
                        5  0.723417  0.790708  0.769141  0.812073  0.563879  0.871834  0.909778  0.952199     1.0}
```
</div>