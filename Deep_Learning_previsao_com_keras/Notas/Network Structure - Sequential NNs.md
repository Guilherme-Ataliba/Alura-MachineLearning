There are many types of neural networks structures. But for starters, lets consider `sequential` NNs, in which the information goes in a direct path, from one layer to the other (sequentially).

# 🔵 Layers
As is well known, neural networks are composed of a number of layers, each one with it's own type and purpose. Let's understand some important aspects of these layers.

### 🔹 Input Layer
This is the layer responsible to take in the input data. 
- Each neuron in this layer corresponds to a feature from the dataset. 

### 🔹 Output Layer
As the name suggests, this layer is responsible for the model's output. 
- The number of nodes in this layers corresponds to the dimension of the output data.

## 🔷 Hidden Layers
These are the layers that actually process the data and try to approximate it. 
- The number of hidden layers can range from 1 to an arbitrary number.
- This is where the weights and biases come into play, to build a linear equation. That is what happens in the hidden layers.

As expected, that are many different types of hidden layers, each one serves its own purpose in the context of the whole neural network model.

![[Pasted image 20240815151355.png]]
### 🔹 Dense Layer
Also called **fully connected**, they get his name from the fact that every node in this layer is connect to every single node of the previous layer. 
- This means that every node gets information about every other node 
	- If this happens right after the input layer, every node in this layer get information about every single feature in the dataset. 


## 🟢 In Python
Using Keras and Tensor Flow, this is the usual way of constructing a neural network model and defining its layers:

```python
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential([
    Dense(1, input_dim=1, kernel_initializer="Ones",
          use_bias = False, activation="linear")
])

regressor.compile(loss="mean_squared_error", optimizer="adam")

regressor.fit(X_train, y_train)
```

This code shows the creating of a sequential NN with a single dense layer. Its also required to define the loss function and the optimizer to use:
- **Loss Function**: How to calculate the error
- **Optimizer**: The technique used to optimize the weights and biases (train the NN). A common optimizer is the **gradient descent** family.

Node that we haven't define a proper input layer, this is a keras feature. If you set the `input_dim` in the first layer it automatically creates an implicit input layer with the number of nodes corresponding to the number of dimensions.