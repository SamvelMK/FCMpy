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
python -m pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel
python -m pip install install e . 
```

You can run the unittest for the package as follows:

```
python -m unittest discover unittests
```

## Examples

TBA

## License

Please read LICENSE.txt in this directory.

</div>