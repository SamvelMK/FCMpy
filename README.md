# Fuzzy Cognitive Maps for Behavior Change Interventions and Evaluation (FcmBci).

The fcmbci is Python package for automatically generating causal weights for fuzzy cognitive maps based on qualitative inputs (by using fuzzy logic), optimizing the FCM connection matrix via Machine Learning Algorithms and testing <em>what-if</em> scenarios. The package includes the following submodules:

* DataProcessor
* Intervention
* Visualize
* ---> (ML TBA)

<a ref=fcmbci\data_processor\data_processor.md> The DatProcessor module </a> includes methods for deriving causal weights of an FCM based on qualitative inputs from experts. <br> 
<a ref=fcmbci\intervention\FcmIntervention.md> Intervention module </a> allows runing simulations on top of the specified FCMs and test <em>what-if</em> scenarios. <br>
<a ref=fcmbci\vizualization\FcmVisualize.md> The Visualize module </a> provides methods for visualizing different components of data processing and simulations. <br>

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

You can run the unittest for the package as follows:

```
python -m unittest discover unittests
```

## Examples



## License

Please read LICENSE.txt in this directory.