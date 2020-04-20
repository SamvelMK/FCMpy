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

Note that a (transfer) function f is applied to the result. As shown in equations above, this function is necessary to keep values within the range [-1, 1]. In the current version, four such functions are implemented:

* Sigmoid:

<div class=container, align=center>

![f(x)=\frac{1}{1+e^{-\lambda x}}, x\epsilon \mathbb{R}; \[0,1\]\\](https://render.githubusercontent.com/render/math?math=f(x)%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Clambda%20x%7D%7D%2C%20x%5Cepsilon%20%5Cmathbb%7BR%7D%3B%20%5B0%2C1%5D%5C%5C)

</div>

* Hyperbolic Tangent:

<div class=container, align=center>

![f(x)=tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^{2x}-1}{e^{2x}+1}, x\epsilon \mathbb{R}; \[-1,1\]\\](https://render.githubusercontent.com/render/math?math=f(x)%3Dtanh(x)%3D%5Cfrac%7Bsinh(x)%7D%7Bcosh(x)%7D%3D%5Cfrac%7Be%5E%7B2x%7D-1%7D%7Be%5E%7B2x%7D%2B1%7D%2C%20x%5Cepsilon%20%5Cmathbb%7BR%7D%3B%20%5B-1%2C1%5D%5C%5C)

</div>

* Bivalent

<div class=container, align=center>

![f(x)=\begin{Bmatrix}  1, & x > 0\\   0, & x\leq 0 \end{Bmatrix}](https://render.githubusercontent.com/render/math?math=f(x)%3D%5Cbegin%7BBmatrix%7D%20%201%2C%20%26%20x%20%3E%200%5C%5C%20%20%200%2C%20%26%20x%5Cleq%200%20%5Cend%7BBmatrix%7D)

</div>

* Trivalent

<div class=container, align=center>

![f(x)=\begin{Bmatrix} 1, & x > 0 \\  0, & x= 0\\  -1, & x < 0 \\ \end{Bmatrix}](https://render.githubusercontent.com/render/math?math=f(x)%3D%5Cbegin%7BBmatrix%7D%201%2C%20%26%20x%20%3E%200%20%5C%5C%20%200%2C%20%26%20x%3D%200%5C%5C%20%20-1%2C%20%26%20x%20%3C%200%20%5C%5C%20%5Cend%7BBmatrix%7D)

</div>

</div>


