# 🔵 Recurrent Neural Networks
The neural networks seen so far are called feed forward neural networks, these networks have a series of neurons were each of them makes calculations and each individual result is added and passed forward, until the output. 

> Recurrent neural networks use information about events that happened long ago and events that happen recently to make predictions abou tomorrow. 

On the other hand, recurrent neural networks uses feedback loops instead of multiple neurons to deal with the different entries from data. 
![[Pasted image 20240816092649.png]]

They are very useful in cases where the predict value depends of the previous observed values. Given the feedback loop, RNNs have a "memory", meaning that they use information from previous points to predict the next one. 

- Thus, instead of summing over each neuron, RNNs sum over each iteration in the feedback loop, to construct the linear equation.
![[Pasted image 20240816093038.png]]

Further more, RNNs can deal with a varying number of "features" with re-training the model. Think of a time series in which the features are expressed in a vector of n entries. The model's input would then be this whole vector, and at every iteration in the feedback loop the model would go over a element from the array at a time, each element would be added sequentially. 

- This process makes RNNs very powerful in time series problems, in which the window may vary depending on the amount of data available. 

![[Pasted image 20240816093430.png]]

## 🔷 Vanishing and Exploding Gradients
In RNNs the weights and bias are the same for each iteration in the feedback loop. Since the solution is multiplied by the weight in every iteration, if the model runs N times, the answer will have a factor of
$$W^{N}$$
This feature will cause different issues depending of the value of W:

1. **Vanishing Gradient**: If $W<1$, given enough iterations, $W^{N}\to 0$ and this factor makes part of the gradient descent algorithm. At the end, the gradient will also tend to zero and the minimization process will stop.
2. **Exploding Gradient**: Similarly, if $W>1$ then $W^{N}\to\infty$ and the gradient will explode, also making the optimization algorithm stop.

🛑 The reason to always use the same weights and biases is so that the model can accept data of any size.

## 🟢 In Python

Utilizing `keras`, the input shape expected to recurrent neural networks is of a 3-tensor. For that reason, usually reshape of the data is necessary. A common way of achieving this is:

```python
X_teste_novo = X_teste_novo.reshape((
	X_teste_novo.shape[0],
	X_teste_novo.shape[1],
	1
))
```


# 🔵 Long Short Term Memory (LSTM)
This an improvement from the original recurrent neural networks that tries to address the problem of vanishing and exploding gradients. 

Instead of using information about event that happened long ago and events that happened recently to make predictions (RNN), LSTMs use two different paths to make predictions, one for **long-term memory** and another for **short-term memory**.

## 🔷 Stages

Following is an example of a unit in a LSTM algorithm.
![[Pasted image 20240816103248.png]]

**Long-Term Memory**: Its the green line above. Notice that it has no weights and biases, but instead it stores a value in long-term memory, that can be update through the first and second stages

**Short-Therm Memory**: This is the pink line bellow. The value of the short-term memory is the output from the last iteration, thus it is only updated at the end of the unit. 

We can dived the unit in 3 different stages. Notice that in those stages two activation functions are frequently used:
1. **Sigmoid**: This is usually used in situations that define a percentage to be remembered from a value
2. **Tanh**: Is used to define a potential value of some kind. Then it is combined with the sigmoid to achieve the actual percentage that will be utilized.
	- Remember that $\tanh$ doesn't define percentages, since it ranges from -1 to 1.

### 🔹 Stage 1 - Forget Gate

The first stage determines how much of the long term memory will be remembered for this iteration
![[Pasted image 20240816103801.png|200]]
1. It sums the input and the short-term memory
2. Then i applies the sigmoid function 
3. This value is then multiplied by the long-term memory and stored for the later stages

### 🔹 Stage 2 - Input Gate
This stage determines how we update the long-term memory for this and the next generation.
![[Pasted image 20240816104653.png|300]]

1. It takes the input to two simultaneous paths and add it to the short-term memory.
2. Split into 2 paths:
	1. The right one defines the potential long-term memory to be remembered, by applying a tanh function on the value
	2. The left one multiplies the previous value by a percentage defining the potential memory to be remembered
3. After this, the resulting value is added to the long-term memory and updates it.

### 🔹 Stage 3 - Output Gate
This stage calculates the output value from a combination of long-term memory, short-term memory and the input. It also defines the short-term memory for the next iteration.

![[Pasted image 20240816105149.png]]

1. The process is divided into two paths:
	1. **Long-term memory path**: The upper path defines the potential short term memory by applying the $\tanh$ to the long-term memory
	2. **Short-term memory path**: The bottom path uses a combination of short-term memory and the input, applied to the sigmoid function, in order to calculate the potential memory to be remembered.
2. These values are then multiplied and together they define the new short-term memory.

## 🔷 Feedback Loop
Above we have described a unit process in a LSTM. But just like regular RNNs, there is a loop that repeat this process for each entry in the input data.

![[Pasted image 20240816105918.png]]

The whole process consists of a connection between the unit cells that transmits the long-term and short-term memory to next units (iterations). The value of the last iteration is the actual value predicted.  

## 🔶 Considerations
Notice that through all this process the values range from -1 to 1 and 0 to 1, this also includes to output. This is a important factor that makes normalizing / standardizing the input so important.


