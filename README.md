# Fuzzy Cognitive Maps for Behavior Change Interventions and Evaluation (FcmBci).

fcmbci is Python package for constructing fuzzy cognitive maps and testing <em>what-if</em> scnearios. The package includes the following submodules:
* FcmDataProcessor
* FcmSimulator
* FcmVisualize

The <a ref=fcmbci\data_processor\data_processor.md> FcmDatProcessor </a> includes methods for deriving causal weights of an FCM based on qualitative inputs from experts. 
The <a ref=fcmbci\simulator\simulator.md> FcmSimulator </a> allows runing simulations on top of the specified FCMs and test <em>what-if</em> scenarios.
The FcmVisualize module provides methods for visualizing different components of data processing and simulations.

## Installation
Fcmbci requires python >=3.6 and depends on:

pandas>=0.25.1
numpy>=1.16.5
scikit-fuzzy>=0.4.2
networkx>=1.9.0
matplotlib>=3.1.1

and will soon be available on PyPi! The lastest version can be installed by:

```
pip install fcmbci
```

Alternatively, you can install from source or develop this package, you can fork and clone this repository then install fcmbci by running:

```
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
```

## Examples

## License

Please read LICENSE.txt in this directory.