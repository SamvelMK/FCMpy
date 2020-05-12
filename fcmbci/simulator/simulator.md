# Methods for simulating FCM's.
<div align = justify>
When creating an FCM, the emphasis is on fuzzy logic as shown in the earlier section on FCMs. However, when using an FCM, the emphasis shifts to mathematics that are similar to neural networks. Indeed, an FCM can also be seen as a “recurrent artificial neural network that creates models as collections of concepts/neurons and the various causal relations that exist between them” (Tsadiras, 2008). Inferring how a determinant quantitatively impacts another is thus analogous to computing how the firing of a neuron propagates throughout a network (Dickerson & Kosko, 1994). Consequently, the equation governing the update of an FCM is often known as ‘activation function’, using the same vocabulary as in neural networks.

In the current version, the fcmbci module implements three types of activation functions; Kosko's, modified Kosko's and Rescaled function. 
</div>

Kosko (Stach et al., 2010):
<div class=container, align=center>

![{d}_i^{s+1}=f(\sum_{j=1} \mathbf{d}_j^s * \mathbf{C}_{ij})](https://render.githubusercontent.com/render/math?math=%7Bd%7D_i%5E%7Bs%2B1%7D%3Df(%5Csum_%7Bj%3D1%7D%20%5Cmathbf%7Bd%7D_j%5Es%20*%20%5Cmathbf%7BC%7D_%7Bij%7D))

</div>

Modified Kosko (Papageorgiou, 2011):
<div class=container, align=center>

![\mathbf{d}_i^{s+1}= f(\mathbf{d}_i+\sum_{j=1} \mathbf{d}_j^s * \mathbf{C}_{ij})\\](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bd%7D_i%5E%7Bs%2B1%7D%3D%20f(%5Cmathbf%7Bd%7D_i%2B%5Csum_%7Bj%3D1%7D%20%5Cmathbf%7Bd%7D_j%5Es%20*%20%5Cmathbf%7BC%7D_%7Bij%7D)%5C%5C)

</div>

Rescaled (Papageorgiou, 2011):
<div class=container, align=center>

![\mathbf{d}_i^{s+1} = f((2\mathbf{d}_i -1)  +\sum_{j=1} (2\mathbf{d}_j^s -1) * \mathbf{C}_{ij})\\](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bd%7D_i%5E%7Bs%2B1%7D%20%3D%20f((2%5Cmathbf%7Bd%7D_i%20-1)%20%20%2B%5Csum_%7Bj%3D1%7D%20(2%5Cmathbf%7Bd%7D_j%5Es%20-1)%20*%20%5Cmathbf%7BC%7D_%7Bij%7D)%5C%5C)

</div>

<div align = justify>

Note that a (transfer) function <em>f</em> is applied to the result. As shown in equations above, this function is necessary to keep values within a certain range (e.g., [0,1] for sigmoid function or [-1,1] for hyperbolic tangent). In the current version, four such functions are implemented:

* Sigmoid:

<div class=container, align=center>

![f(x)=\frac{1}{1+e^{-\lambda x}}, x\in\mathbb{R}; \[0,1\]](https://render.githubusercontent.com/render/math?math=f(x)%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Clambda%20x%7D%7D%2C%20x%5Cin%5Cmathbb%7BR%7D%3B%20%5B0%2C1%5D)

</div>

* Hyperbolic Tangent:

<div class=container, align=center>

![f(x)=tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^{2x}-1}{e^{2x}+1}, x\in\mathbb{R}; \[-1,1\]\\](https://render.githubusercontent.com/render/math?math=f(x)%3Dtanh(x)%3D%5Cfrac%7Bsinh(x)%7D%7Bcosh(x)%7D%3D%5Cfrac%7Be%5E%7B2x%7D-1%7D%7Be%5E%7B2x%7D%2B1%7D%2C%20x%5Cin%5Cmathbb%7BR%7D%3B%20%5B-1%2C1%5D%5C%5C)

</div>

* Bivalent:

<div class=container, align=center>

![f(x)=\begin{Bmatrix}  1, & x > 0\\   0, & x\leq 0 \end{Bmatrix}](https://render.githubusercontent.com/render/math?math=f(x)%3D%5Cbegin%7BBmatrix%7D%20%201%2C%20%26%20x%20%3E%200%5C%5C%20%20%200%2C%20%26%20x%5Cleq%200%20%5Cend%7BBmatrix%7D)

</div>

* Trivalent:

<div class=container, align=center>

![f(x)=\begin{Bmatrix} 1, & x > 0 \\  0, & x= 0\\  -1, & x < 0 \\ \end{Bmatrix}](https://render.githubusercontent.com/render/math?math=f(x)%3D%5Cbegin%7BBmatrix%7D%201%2C%20%26%20x%20%3E%200%20%5C%5C%20%200%2C%20%26%20x%3D%200%5C%5C%20%20-1%2C%20%26%20x%20%3C%200%20%5C%5C%20%5Cend%7BBmatrix%7D)

</div>

<div align = justify>

The FcmSimulator module includes methods for runing simulations on an FCM and testing <em>what-if</em> scenarios. In the current version, it includes two methods: [simulate()](#simulate) and [test_scenario()](#test_scenario). You can create an FcmSimulator instance by either supplying an initial state vector and a weight matrix or leaving it empty.

Example:

```
sim = FcmSimulator(init, weight)
```
In the first case, the module automatically runs an FCM simulation on the supplied initial state vector and finds the equilibrium state and stores it in the constructor to be used for testing of scenarios.

If one is not interested in testing scenarios, one could leave it empty.

```
sim = FcmSimulator()
```

## simulate()

The simulate method allows one to run simulations on the defined FCM. The method takes two parameters state and weights. The state is the initial state vector of the concepts in the FCM and the weights is the matrix with the edge weights. The optional arguments include the iteration steps, inference method, transfer function, lambda parameter and the threshold parameter.  

```
simulate(state, weights, iterations = 50, inference = 'mk', 
                 transfer = 's', l = 1, thresh = 0.001)

Parameters
----------
State : dict,
            A dictionary of Concepts as keys and their states. ---> {'C1': 0.5, 'C2' : 0.4}.
            The states take only values in the range of [0,1] for the sigmoid transfer function and [-1,1] for the hperbolic tangent.

weights : Data frame with the causal weights.

iterations : int,
                Number of itterations to run in case if the system doesn't converge.
inference : str,
            default --> 'mk' -> modified kosko; available options: 'k' -> Kosko, 'r' -> Rescale.
            Method of inference.
                    
transfer : str,
            default --> 's' -> sigmoid; available options: 'h' -> hyperbolic tangent; 'b' -> bivalent; 't' trivalent. 
            transfer function.
l : int,
    A parameter that determines the steepness of the sigmoid function. 
        
thresh : float,
            default -->  0.001,
            a thershold for convergence of the values.

Return
----------
y : dataframe,
    dataframe with the results of the simulation steps.

```

The simulation is itterated until either of the two conditions are met: 1) output (A) converges to a fixed point attractor (delta(T) <= 0.001); or 2) maximum number of itterations passed to the function is reached. The latter indicates that either a cyclic or a chaotic behavior of the system (Napoles et al., 2020).

Example:

In the example below, we first generate the causal weights based on the examples shown in the data_processor.md. 

```
from fcmbci import FcmDataProcessor
from fcmbci import FcmSimulator

fcm = FcmDataProcessor()
fcm.read_xlsx(os.path.abspath('unittests/test_cases/sample_test.xlsx'), 'Matrix')
fcm.gen_weights_mat()
fcm.create_system() # create system for visualizations.

init = {'C1' : 1, 'C2' : 1, 'C3' : 0, 'C4' : 0} # create the initial state vector
sim = FcmSimulator(init, fcm.causal_weights) # instantiate the FcmSimulator object with initial state vector and the causal weights.
```
```
Output:

The values converged in the 7 state (e <= 0.001)
```

One can visualize the simulation steps with simulation_view() static method. 

```
from fcmbci import FcmVisualize as vis
vis.simulation_view(fcm.system, sim.scenarios, 'initial_state', network_view =False)
```
<img src="..\..\figures\figure_10.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 10:</em> Simulation results.

One could also look at the simulation results and the system simultaneously.

```
vis.simulation_view(sim.scenarios, 'initial_state', network_view =True)
```
<img src="..\..\figures\figure_11.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 11:</em> Simulation results and the System's view.

Here the intensity of the color of the nodes indicate the extent to which the nodes are active (the darker the node the more active the node is).

## test_scenario
The test_scenario() method allows runing FCM simulations of what-if scenarios by supplying/changing the state vectors. Changing the state vector implies turning on/off certain concepts in the system. Most of the default parameters are the same as in simulate method described above. The scenario name is the name of the scenario to be tested. The state vector is a dictionary with the keys of the concepts to be changed in the state vector that has been supplied in when creating an FcmSimulator instance.

```
test_scenario(scenario_name, state_vector, weights, 
                iterations = 50, inference = 'mk', transfer = 's', l = 1, thresh = 0.001):

Parameters
----------
scenario_name : str,
                name of the scenario.
state_vector : dict,
                A dictionary of target concepts as keys and their states. ---> {'C1': 0, 'C2' : 1}.
                The states take only values in the range of [0,1] for the sigmoid transfer function and [-1,1] for the hperbolic tangent.

weights : Data frame with the causal weights.

iterations : int,
                Number of itterations to run in case if the system doesn't converge.
inference : str,
            default --> 'mk' -> modified kosko; available options: 'k' -> Kosko, 'r' -> Rescale.
            Method of inference.
                    
transfer : str,
            default --> 's' -> sigmoid; available options: 'h' -> hyperbolic tangent; 'b' -> bivalent; 't' trivalent. 
            transfer function.
l : int,
    A parameter that determines the steepness of the sigmoid and hyperbolic tangent function at values around 0. 
        
thresh : float,
            default -->  0.001,
            a thershold for convergence of the values.
```


Example:

Now that we instantiated the FcmSimulator object we can test <em> what-if </em> scenarios. The test_scenario() method uses the state vector derived from the fixed point equilibrium of the system state computed during the instantiation of the FcmSimulator object. Let's suppose we deactivated node C1 and activated node C2.

```
sim.test_scenario('scenario_1', {'C1' : 0, 'C2' : 1}, fcm.causal_weights)
```
```
Output: 

The values converged in the 7 state (e <= 0.001)
```
We can visualize it with the simulation_view() method. Now that we have target nodes we can include it as an argument in the method.

```
vis.simulation_view(sim.scenarios, 'scenario_1', network_view =True, target = ['C1', 'C2'])
```

<img src="..\..\figures\figure_12.PNG" alt="figure not found" style="float: center; margin-right: 10px;" /><br>
<em>Figure 12:</em> Simulation results for Scenario 1.

</div>


