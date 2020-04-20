# Methods for simulating FCM's.

When creating an FCM, the emphasis is on fuzzy logic as shown in the earlier section on FCMs. However, when using an FCM, the emphasis shifts to mathematics that are similar to neural networks. Indeed, an FCM can also be seen as a “recurrent artificial neural network that creates models as collections of concepts/neurons and the various causal relations that exist between them” (Tsadiras, 2008). Inferring how a determinant quantitatively impacts another is thus analogous to computing how the firing of a neuron propagates throughout a network (Dickerson & Kosko, 1994). Consequently, the equation governing the update of an FCM is often known as ‘activation function’, using the same vocabulary as in neural networks.

In the current version, the fcmbci module implements three types of activation functions; Kosko's, modified Kosko's and Rescaled function. 

Kosko (Stach et al., 2010):
<div class=container, align=center>

![{d}_i^{s+1}=f(\sum_{j=1} \mathbf{d}_j^s * \mathbf{C}_{ij})](https://render.githubusercontent.com/render/math?math=%7Bd%7D_i%5E%7Bs%2B1%7D%3Df(%5Csum_%7Bj%3D1%7D%20%5Cmathbf%7Bd%7D_j%5Es%20*%20%5Cmathbf%7BC%7D_%7Bij%7D))

</div>

Modified Kosko (Papageorgiou, 2011):
<div class=container, align=center>

![\mathbf{d}_i^{s+1} = f(\sum_{j=1} \mathbf{d}_j^s * \mathbf{C}_{ij})\\](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bd%7D_i%5E%7Bs%2B1%7D%20%3D%20f(%5Csum_%7Bj%3D1%7D%20%5Cmathbf%7Bd%7D_j%5Es%20*%20%5Cmathbf%7BC%7D_%7Bij%7D)%5C%5C)

</div>

Rescaled (Papageorgiou, 2011):
<div class=container, align=center>

![\mathbf{d}_i^{s+1} = f((2\mathbf{d}_i -1)  +\sum_{j=1} (2\mathbf{d}_j^s -1) * \mathbf{C}_{ij})\\](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bd%7D_i%5E%7Bs%2B1%7D%20%3D%20f((2%5Cmathbf%7Bd%7D_i%20-1)%20%20%2B%5Csum_%7Bj%3D1%7D%20(2%5Cmathbf%7Bd%7D_j%5Es%20-1)%20*%20%5Cmathbf%7BC%7D_%7Bij%7D)%5C%5C)

</div>

</div>


