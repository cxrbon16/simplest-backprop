activations = {'relu', 'no-activation', 'sigmoid', 'tanh'}
import backprop
import numpy as np
import random
layer_key = 1
neuron_key = 1


class Neuron:
    def __init__(self, X, W_dim, b_dim, activation):
        self.X = X
        self.W_dim = W_dim
        self.b_dim = b_dim
        self.activation = activation
        self.W = [backprop.Item(value=random.randint(-1,1)) for _ in range(self.W_dim[0])]
        self.b = 0

        
    def forward_prop(self):
        def apply_activation(x):
            if self.activation == 'relu':
                if x.value > 0:
                    return x
                else:
                    return backprop.Item(value=0)
            if self.activation == 'no-activation':
                return x
            if self.activation == 'sigmoid':
                return 1/(1+np.exp(x))
        return apply_activation(np.dot(self.W, self.X) + self.b)
    
    def get_params(self):
        print(f"W: {[i for i in self.W]}, B: {self.b}")


class Layer:
    def __init__(self, num_units = 1, X = None, activation = '', ):
        if activation not in activations:
            raise ValueError(f"param1 must be one of: {', '.join(activations)}")
        self.activation = activation
        self.units = num_units
        self.activation = activation
        self.out_dim = num_units
        self.X = X
        self.b_dim = (num_units)
        self.w_dim = (len(X), num_units)
        self.units = []
        for _ in range(num_units):
            self.units.append(Neuron(X, W_dim=self.w_dim, b_dim=self.b_dim, activation=self.activation))
        print(self.units)
    def forward_prop(self):
        result = []
        for n in self.units:
            result.append(n.forward_prop())
        return result
    

x = Layer(activation='relu', num_units=12, X=[12,12,12,13,14])
x.forward_prop()[0].backprop()

x.units[0].get_params()
