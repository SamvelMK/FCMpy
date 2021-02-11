# Fuzzy Cognitive Maps for Behavior Change Interventions and Evaluation (FcmBci).
<div align = justify>

The fcmbci is Python package for automatically generating causal weights for fuzzy cognitive maps based on qualitative inputs (by using fuzzy logic), optimizing the FCM connection matrix via Machine Learning Algorithms and testing <em>what-if</em> scenarios. The package includes the following submodules:

* DataProcessor
* Simulation
* Intervention
* ---> (ML TBA)

<a href="fcmbci\data_processor\data_processor.md"> The DataProcessor module </a> includes methods for deriving causal weights of an FCM based on qualitative inputs from experts. <br> 
<a href="fcmbci\simulator\simulator.md"> The Simulator module </a> provides methods for runing simulations on top of a given FCM structure. <br>
<a href="fcmbci\intervention\intervention.md"> The Intervention module </a> allows testing what-if scenarios on top of the specified FCMs. <br>

## Installation
Fcmbci requires python >=3.8.1 and depends on:

* pandas>=1.0.3
* numpy>=numpy==1.18.2
* scikit-fuzzy>=0.4.2
* tqdm>=4.50.2

and will soon be available on PyPi! The lastest version can be installed by:

```
pip install fcmbci
```

Alternatively, you can install from source or develop this package, you can fork and clone this repository then install fcmbci by running:

```
python -m pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel
pip install e . 
```

You can run the unittest for the package as follows:

```
python -m unittest discover unittests
```

## Examples

The DataProcessor provides the utilities to construct an FCM (the connection/weight matrix) based on the qualitative inputs (by using fuzzy logic). One can either supply the data directly when creating an instance of a class:

```
from fcmbci import DataProcessor

lt = ['-VH', '-H', '-M', '-L','-VL', 'VL','L', 'M', 'H', 'VH']

fcm = DataProcessor(linguistic_terms=lt, data)
```

or read the data in by uising one of the following methods: <em>read_xlsx</em>, <em>read_json</em>, <em>read_csv</em> (see more details in the <a href="fcmbci\data_processor\data_processor.md"> documentation </a>).

```
from fcmbci import DataProcessor

lt = ['-VH', '-H', '-M', '-L','-VL', 'VL', 'L', 'M', 'H', 'VH']

fcm = DataProcessor(linguistic_terms=lt)

fcm.read_xlsx(filepath, check_consistency=True)
```

If the <em>check_consistency</em> is set to <em>True</em> then the algorithm searches for inconsistencies in the expert raitings. After data is provided to the fcmbci object, the algorithm automatically calculates the entropy of the experts' raitings of the concepts. The data and the entropy scores can be accessed as follows:

```
fcm.data
fcm.entropy
```
After supplying the data to the fcmbci object, one can use the <em>gen_weights</em> method to construct the connection matrix (see more details on the process of deriving the weight matrix in the <a href="fcmbci\data_processor\data_processor.md"> documentation </a>). The <em>gen_weights</em> method by default uses triangular membership functions, Mamdani product fuzzy inference method and centroid defuzzification method (see more details on the methods in the <a href="fcmbci\data_processor\data_processor.md"> documentation</a>).

```
gen_weights(method = 'centroid', membership_function='trimf', fuzzy_inference="mamdaniProduct", **params)
```

The connection matrix can be accessed/inspected as follows:
```
fcm.weight_matrix
```

Once we constructed the weight matrix we can use the <em>Simulate</em> module to run simulations on top of the defined FCM structure. After creating an instance of the Simulation class one can use the <em>simulate</em> method to run the FCM simulations. The sumulate method requires one to pass the initial state vector (<em>initial_state</em>), connection matrix (<em>weight_matrix</em>), the transfer (<em>transfer</em>) and inference methods (<em>inference</em>), the threshold (<em>thresh</em>) for stoping the simulation and the number of iterations to run (<em>iterations</em> ) (see more details on the methods in the <a href="fcmbci\simulator\simulator.md">documentation</a>).

```
from fcmbci import Simulator

sim = Simulator()

res = sim.simulate(initial_state=init_state, weight_matrix=weight_matrix, 
                    transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50)
```

If one wants to test intervention scenarios (what-if scenarios) one can use the utilities provided in the <em>Intervention</em> module. To create an instance of the Intervention class, as in the case of the simulations one needs to pass the initial state vector, connection matrix, the transfer and inference methods, the threshold for stoping the simulation and the number of iterations to run. Additionally, if one uses the sigmoid transfer function, one needs to specify the <em>l</em> (i.e., lambda) parameter for the sigmioid function (see more details on the methods in the <a href="fcmbci\intervention\intervention.md">documentation</a>).

```
inter = Intervention(initial_state=init_state, weight_matrix=weight_matrix, transfer='sigmoid', 
                        inference='mKosko', thresh=0.001, iterations=100, l=1)
```

After instantiating the Intervention class, one can add intervention cases. One can do that by using the <em>add_intervention</em> method. The method requires one to pass the name of the intervention (<em>name</em>), the weights (a dictionary where the keys are the concepts the intervention impacts and the values are the associated causal impact) (<em>weights</em>), and the effectiveness of the intervention (<em>effectiveness</em>).

```
inter.add_intervention(name='intervention_1', weights={'C1':-.3, 'C2' : .5}, effectiveness=1)
```

Once the intervention cases are specified, one can use the <em>test_intervention</em> method to run the simulations for a given scenario.

```
inter.test_intervention('intervention_1')
```
The results can be accessed as follows:

```
test_results['intervention_1']
```

## License

Please read LICENSE.txt in this directory.

</div>