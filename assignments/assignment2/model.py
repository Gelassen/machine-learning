import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # DONE Create necessary layers
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # DONE Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        
        # DONE Compute loss and fill param gradients
        # by running forward and backward passes through the model
        probs = self.first_layer.forward(X)
        activation = self.relu.forward(probs)
        second_probs = self.second_layer.forward(activation)
        
        loss, grad = softmax_with_cross_entropy(second_probs, y)
        
        d_out = self.second_layer.backward(grad)
        d_out = self.relu.backward(d_out)
        d_out = self.first_layer.backward(d_out)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params().values():
            loss_l2, grad_l2 = l2_regularization(param.value, self.reg)
            loss += loss_l2
            param.grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        step_1 = self.first_layer.forward(X)
        step_2 = self.relu.forward(step_1)
        step_3 = self.second_layer.forward(step_2)

        # shall we use softmax with cross entropy?
        probs = softmax(step_3)
        
        y_pred = np.array(list(map(lambda x: x.argsort()[-1], probs)))
        
        return y_pred

    def params(self):
#         result = {}

        # DONE Implement aggregating all of the params

        return {
            'first_layer.W' : self.first_layer.W, 'first_layer.B' : self.first_layer.B,
            'second_layer.W' : self.second_layer.W, 'second_layer.B' : self.second_layer.B
        }
