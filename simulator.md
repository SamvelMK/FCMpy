# Methods for simulating FCM's.

When creating an FCM, the emphasis is on fuzzy logic as shown in the earlier section on FCMs. However, when using an FCM, the emphasis shifts to mathematics that are similar to neural networks. Indeed, an FCM can also be seen as a “recurrent artificial neural network that creates models as collections of concepts/neurons and the various causal relations that exist between them” (Tsadiras, 2008). Inferring how a determinant quantitatively impacts another is thus analogous to computing how the firing of a neuron propagates throughout a network (Dickerson & Kosko, 1994). Consequently, the equation governing the update of an FCM is often known as ‘activation function’, using the same vocabulary as in neural networks.

In the current version, the fcmbci module implements three types of activation functions; Kosko's, modified Kosko's and Rescaled function. 

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Cmathbf%7Bd%7D_i%5E%7Bs&plus;1%7D%20%26%20%3D%20f%28%5Csum_%7Bj%3D1%7D%20%5Cmathbf%7Bd%7D_j%5Es%20*%20%5Cmathbf%7BC%7D_%7Bij%7D%29%5C%5C%20%5Cend%7Balign%7D)
</div>