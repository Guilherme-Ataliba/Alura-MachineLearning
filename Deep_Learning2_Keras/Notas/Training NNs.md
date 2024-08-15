# 🔵 NN Structure
In general, neural networks are composed of different layers. The simplest case usually seen is as follows

![[Pasted image 20240814215428.png]]

1. We have an input layer that is responsible for treating and transforming the input data in a way that is comprehensible for the neural network.
2. Hidden layers are were the calculations and the "fitting" of the data happen. They make connections, update weights and biases in a way to best fit the data.
	1. The connections between multiple of these layers will increase the model's complexity and (maybe) its predictive capabilities (or may it'll just overfit).
3. Finally we have the output layer that represents the model's output. 

The number of nodes are related to the dimensions of the data treated. In this case, it receives two inputs, processes it through the **hidden layer** with four nodes and then outputs a single value.
- This could be for example a binary classifier. 


## 🔷 Parameters
Neural networks work based on a linear equation:
$$F(X) = A\left(\sum W_ix + b_i\right)$$
- **W**: Weights are matrices, they are the multiplicative parameter .
- **b**: Biases are vectors, these are the additive parameter.
- **A**: Activation Functions. It tries to insert non-linearity in the model.

Both parameters try to move the input closer to the desired output, by scaling and translating it trough the space. 
- ! Note that $W_1$ and $b_1$ are a matrix and a vector, respectively. They have the length of the input data. 

▶ The summation goes over the number of layers in the neural network.

### 🔹 Parameters in Layer
The number of weights and biases from a hidden layer follow a pattern:

**Weights:** 
$$nWeights = nNodes \cdot lengthInput$$
- This means that each entry in the input data has its own weight in every node.

**Biases**:
While the weights are a matrix, corresponding to each entry in the input data and the number of nodes, the bias is a vector with length equal to the number of nodes. Thus, there's a single bias per node (usually we think as a node connection)
$$nBiases = nNodes$$

So the number of parameters in a hidden layer is equal to
$$nParams = nNodes \cdot lengthInput + nNodes$$
- Its not a coincidence that this equation is very similar to the linear equation used in NN.


# 🔵 Training NNs

## 🔷 Initial Values
Weights and biases must be constructed once you construct a new layer in the neural network. Thus, they must be initialized with some value. 

A simple initialization with zeros would not be enough. The ideia is, for the NN to be able to train the different weights accordingly, and give different values for each neuron, they must be initialized in a non-symmetrical manner.
- That's we don't initialize everything with zeros, since that would make every node on even ground, and they could follow very similar paths in the training process. 

### 🔹 Vanishing Gradients
I "common problem" in neural networks happen when the weights and biases get so small or so big that the gradient (in the process of gradient descent in backward propagation) vanishes (goes to zero). 
- This process basically kills the neurons (and possibly the whole NN), since they wont change from that point on, with training. 

## 🔷 Training Methodologies

### 🔹 Forward Propagation
This is the simplest form of training, it is as we usually think of this flow. The data enters in the input data, that is then passed through each layer of the network. 
- Each neuron applies a linear transformation followed by a non-linear activation function.
Each result is passed to the next layer and this process continues until the final layer produces an output. 

In this process, *information* is only transmitted forward, one layer at a time, until the end. For that reason, it is called forward propagation. 


### 🔹 Back Propagation
The Backpropagation algorithm takes the parameters found in the forward pass and passes them again through the neural network. The idea is to update the parameters with each pass (epoch or iteration). 
- This process of updating the parameters to minimize the loss utilizes **gradient descent**. 

#### Stochastic Gradient Descent
Given that the ordinary gradient descent is very dependent on initial conditions and thus fall into local minima. The stochastic version of gradient descent is more used in training neural networks overall, since it can deal with these pitfalls better. 