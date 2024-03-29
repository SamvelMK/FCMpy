# Methods for simulating FCM's <a href="https://www.sciencedirect.com/science/article/pii/S2666521220300089?via%3Dihub">(Retrived from Mkhitaryan, Giabbanelli, Vries & Crutzen, 2020) </a>.
<div align = justify>
When creating an FCM, the emphasis is on fuzzy logic as shown in the <a href="\fcmpy\ExpertFcm\ExpertFcm.md">ExpertFcm</a> module. However, when using an FCM, the emphasis shifts to mathematics that are similar to neural networks. Indeed, an FCM can also be seen as a “recurrent artificial neural network that creates models as collections of concepts/neurons and the various causal relations that exist between them” (Tsadiras, 2008). Inferring how a determinant quantitatively impacts another is thus analogous to computing how the firing of a neuron propagates throughout a network (Dickerson & Kosko, 1994). Consequently, the equation governing the update of an FCM is often known as ‘activation function’, using the same vocabulary as in neural networks. <br>
<br>

## simulate()
The simulate method in the FcmSimulator module allows one to run simulations over the defined FCM structure. The method requires the initial state vector <em>(initial_state)</em> , the connection matrix <em>(weight_matrix)</em>, the preferred transfer and inference methods <em>(transfer, inference)</em>. The optional parameters include the the threshold for the error <em>(thresh)</em>, and the number of iterations for the simulation <em>(iterations)</em>. The <a href="https://www.springer.com/gp/book/9783642397387">threshold</a> will depend on the specific domain of application (in most cases it is 0.001).

```Python
simulate(initial_state, weight_matrix, transfer, inference, thresh=0.001, iterations=50, **params):
        
        Runs simulations over the passed FCM.
        
        Parameters
        ----------
        initial_state: dict
                        initial state vector of the concepts
                        keys ---> concepts, values ---> initial state of the associated concept

        weight_matrix: pd.DataFrame, np.ndarray
                        N*N weight matrix of the FCM.

        transfer: str
                    transfer function --> "sigmoid", "bivalent", "trivalent", "tanh"

        inference: str
                    inference method --> "kosko", "mKosko", "rescaled"

        thresh: float
                    threshold for the error

        iterations: int
                        number of iterations

        **kwargs: additional parameters for the methods (e.g., l parameter for the sigmoid transfer function)

        Return
        ----------
        y: pandas.DataFrame
                results of the simulation.       
```

In each iteration, the concept values of the defined FCM are updated according to the defined inference method (i.e., activation function). The module implements the following three types of inference methods:
</div>

Kosko <a href="https://link.springer.com/chapter/10.1007/978-3-642-03220-2_2">(Stach et al., 2010)</a>:
<div class=container, align=center>

$$ 
d_{i}^{s+1}=f(\sum_{j=1}^n d_{j}^s * C_{ji}) 
$$

</div>

Modified Kosko <a href="https://www.sciencedirect.com/science/article/abs/pii/S1568494609002646">(Papageorgiou, 2011)</a>:
<div class=container, align=center>

$$
d_{i}^{s+1}= f(d_{i}+\sum_{j=1}^n d_j^s * C_{ji})
$$

</div>

Rescaled <a href="https://www.sciencedirect.com/science/article/abs/pii/S1568494609002646">(Papageorgiou, 2011)</a>:
<div class=container, align=center>

$$
d_{i}^{s+1} = f((2d_{i} -1)  +\sum_{j=1}^n (2d_{j}^s -1) * C_{ji}) 
$$

</div>

<div align = justify>

Where $d_{j}$ is the value of concept $j$ at the simulation step $s$ and the $C_{j,i}$ is the causal impact of concept $j$ on concept $i$. Note that a (transfer) function <em>f</em> is applied to the result. As shown in equations above, this function is necessary to keep values within a certain range (e.g., [0,1] for sigmoid function or [-1,1] for hyperbolic tangent). In the current version, four such functions are implemented:

* Sigmoid:

<div class=container, align=center>

$$ f(x)=\frac{1}{1+e^{-\lambda x}}, x\in\mathbb{R}; [0,1] $$

</div>

* Hyperbolic Tangent:

<div class=container, align=center>

$$
f(x)=tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^{2x}-1}{e^{2x}+1}, x\in\mathbb{R}; [-1,1]
$$

</div>

* Bivalent:

<div class=container, align=center>

$$
f(x)=\begin{Bmatrix}  1, & x > 0\\   0, & x\leq 0 \end{Bmatrix}
$$

</div>

* Trivalent:

<div class=container, align=center>

$$
f(x)=\begin{Bmatrix} 1, & x > 0 \\  0, & x= 0\\  -1, & x < 0 \end{Bmatrix}
$$

</div>

One can alternatively add their own inference method and transfer function via [add_inference_methods](#add_inference_methods) and [add_transfer_func](#add_transfer_func) 


<div align = justify>

The simulation is run until either of the two conditions are met: 1) output (A) converges to a fixed point attractor (e.g., delta(T) $\le$ 0.001); or 2) maximum number of iterations passed to the function is reached. The latter indicates that either a cyclic or a chaotic behavior of the system (Napoles et al., 2020).

Fixed point <a href="https://dl.acm.org/doi/abs/10.1016/j.ins.2008.05.015">(Tsadiras 2008)</a>:
<div class=container, align=center>

$$
\exists t_{\alpha} \in \lbrace 1,2, ..., (T-1)\rbrace : A^{(t+1)}=A^{(t)}, \forall t \geq t_{\alpha}
$$

</div>

Limit Cycle <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC55187/">(Wang et al. 1990)</a>:
<div class=container, align=center>

$$
\exists t_{\alpha}, P \in \lbrace 1,2, ..., (T-1)\rbrace : A^{(t+P)}=A^{(t)}, \forall t \geq t_{\alpha}
$$

</div>

Example (replicated the example presented in the fcm inference package in R by Dikopoulou & Papageorgiou):

Let's first create a connection matrix for a sample FCM with eight concepts and define the initial state vector for these concepts.

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

Now let's create an instance of the FcmSimulator class and run a simulation over the defined FCM structure.

```Python
from fcmpy import FcmSimulator

sim = FcmSimulator()

res = sim.simulate(initial_state=init_state, weight_matrix=weight_matrix, transfer='sigmoid', inference='mKosko', thresh=0.001, iterations=50, l=1)
```

```Python
Output[1]:

The values converged in the 7 state (e <= 0.001)

        C1        C2        C3        C4        C5        C6        C7        C8
0  1.000000  1.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
1  0.750260  0.731059  0.645656  0.710950  0.500000  0.500000  0.549834  0.785835
2  0.738141  0.765490  0.749475  0.799982  0.746700  0.769999  0.838315  0.921361
3  0.730236  0.784168  0.767163  0.812191  0.805531  0.829309  0.898379  0.950172
4  0.727059  0.789378  0.769467  0.812967  0.816974  0.838759  0.908173  0.954927
5  0.726125  0.790510  0.769538  0.812650  0.818986  0.839860  0.909707  0.955666
6  0.725885  0.790706  0.769451  0.812473  0.819294  0.839901  0.909940  0.955774
```

![png](/figures/simulations.png)

<em>Figure 1:</em> The results of the FCM simulation. <br>
<br>

</div>