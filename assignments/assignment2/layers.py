import numpy as np

# from linear_classifier import softmax

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # DONE implement softmax
    # Your final implementation shouldn't have any loops
    predictions_copy = predictions.copy()
    lambda_exp = np.vectorize(lambda x: np.exp(x))
    if len(predictions_copy.shape) == 1: 
        predictions_copy -= np.max(predictions_copy)
        exp_probs = lambda_exp(predictions_copy)
        probs = exp_probs / np.sum(exp_probs)
    else:
        predictions_copy = list(map(lambda x: x - np.max(x), predictions_copy))
        exp_probs = lambda_exp(predictions_copy)
        # TODO check is there need in calc exp_probs by element as well
        probs = np.array(list(map(lambda x: x / np.sum(x), exp_probs)))
    
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # DONE implement cross-entropy
    # Your final implementation shouldn't have any loops

    lambda_log = np.vectorize(lambda x: -np.log(x))
     
    if len(probs.shape) == 1: 
        probs_target = probs[target_index]
        size = 1
    else:
        batch_size = np.arange(target_index.shape[0])
        probs_target = probs[batch_size, target_index.flatten()]
        size = target_index.shape[0]
    
    return np.sum(lambda_log(probs_target)) / size

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
#     loss = reg_strength * np.sum(np.dot(W.T, W))

#     batch_size = np.arange(W.shape[1])
#     grad = np.array((list(map(lambda x: np.sum(W,axis=1), batch_size))))
#     grad = 2 * reg_strength * np.transpose(grad)
    
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W

    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    predictions_copy = preds.copy()
    
    probs = softmax(predictions_copy)
    loss = cross_entropy_loss(probs, target_index)
    d_preds = probs

    if (len(preds.shape) == 1):
        d_preds[target_index] -= 1
    else:
        batch_size = np.arange(target_index.shape[0])
        d_preds[batch_size, target_index.flatten()] -= 1 
        d_preds = d_preds / target_index.shape[0]

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # DONE: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        zeros_dataset = np.zeros_like(X)
        self.diff = (X > 0).astype(float)
        return np.maximum(zeros_dataset, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # DONE: Implement backward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        return self.diff * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # DONE: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # DONE: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0, keepdims=True)
        return np.dot(d_out, self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
