# Methods for simulating FCM's.

When creating an FCM, the emphasis is on fuzzy logic as shown in the earlier section on FCMs. However, when using an FCM, the emphasis shifts to mathematics that are similar to neural networks. Indeed, an FCM can also be seen as a “recurrent artificial neural network that creates models as collections of concepts/neurons and the various causal relations that exist between them” (Tsadiras, 2008). Inferring how a determinant quantitatively impacts another is thus analogous to computing how the firing of a neuron propagates throughout a network (Dickerson & Kosko, 1994). Consequently, the equation governing the update of an FCM is often known as ‘activation function’, using the same vocabulary as in neural networks.

In the current version, the fcmbci module implements three types of activation functions; Kosko's, modified Kosko's and Rescaled function. 

</div>
<img src="https://render.githubusercontent.com/render/math?math={d}_i^{s+1}=f(\sum_{j=1} \mathbf{d}_j^s * \mathbf{C}_{ij})">

