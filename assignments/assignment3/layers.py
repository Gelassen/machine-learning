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

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO clarify why there is no Xavier or one another initialisation is used
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X

        out_height = width-self.filter_size+1+2*self.padding
        out_width = width-self.filter_size+1+2*self.padding
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        if self.padding>0:
            self.X_pad = np.zeros((batch_size, height+2*self.padding, width+2*self.padding,channels))
            self.X_pad[:,self.padding:height+self.padding,self.padding:width+self.padding,:] = self.X
        else: self.X_pad = self.X
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        W_step = self.W.value.reshape(self.filter_size*self.filter_size*channels, self.out_channels)
        B_step = np.zeros((batch_size, self.out_channels))
        B_step = np.array(list(map(lambda x: self.B.value, B_step)))
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_step = self.X_pad[:,x:x+self.filter_size,y:y+self.filter_size,:]
                X_step = X_step.reshape(batch_size,self.filter_size*self.filter_size*channels)
                result[:,x,y,:] = np.dot(X_step,W_step)+B_step
                
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_input_pad = np.zeros_like(self.X_pad)
        W_step = self.W.value.reshape(self.filter_size*self.filter_size*channels, out_channels)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                d_out_step = d_out[:,x,y,:].reshape(batch_size,out_channels)
                X_step = self.X_pad[:,x:x+self.filter_size,y:y+self.filter_size,:]
                res = np.dot(d_out_step,np.transpose(W_step)).reshape(X_step.shape)
                d_input_pad[:,x:x+self.filter_size,y:y+self.filter_size,:]+=res
                X_step = X_step.reshape(batch_size,self.filter_size*self.filter_size*channels)
                self.W.grad += np.dot(np.transpose(X_step),d_out_step).reshape(self.W.grad.shape)
                self.B.grad += np.sum(d_out_step, axis=0)
        if self.padding>0:
            d_input = d_input_pad[:,self.padding:height+self.padding,self.padding:width+self.padding,:]
        else: d_input = d_input_pad
        
        return d_input
    
#     def forward(self, X):    
#         batch_size, height, width, channels = X.shape       
#         height_filter, width_filter, _, n_filter = self.W.value.shape
        
#         padded_height = height + 2 * self.padding
#         padded_width = width + 2 * self.padding
#         step = 1   
#         out_height = padded_height - height_filter + step   
#         out_width = padded_width - width_filter + step
        
#         X_padded = np.zeros((batch_size, padded_height, padded_width, channels))
#         X_padded[:, self.padding:height+self.padding, self.padding:width+self.padding, :] = X
#         self.X_cache = (X, X_padded)
        
#         output = np.zeros((batch_size, out_height, out_width, self.out_channels), dtype=int)
#         # TODO: Implement forward pass
#         # Hint: setup variables that hold the result
#         # and one x/y location at a time in the loop below
        
#         # It's ok to use loops for going over width and height
#         # but try to avoid having any other loops
#         W_step = self.W.value.reshape(self.filter_size*self.filter_size*channels, self.out_channels)
#         B_step = np.zeros((batch_size, self.out_channels))
#         B_step = np.array(list(map(lambda x: self.B.value, B_step)))
#         for y in range(out_height):
#             for x in range(out_width):
#                 height_start, width_start = y,x
#                 height_end = height_start + height_filter
#                 width_end = width_start + width_filter

#                 X_step = X_padded[:, width_start:width_end, height_start:height_end, :]
#                 X_step = X.reshape(batch_size, self.filter_size*self.filter_size*channels)
#                 dot_product = np.dot(X_step, W_step)+ B_step

#                 output[:, x, y, :] = dot_product

#         return output

#     def backward(self, d_out):
#         # Hint: Forward pass was reduced to matrix multiply
#         # You already know how to backprop through that
#         # when you implemented FullyConnectedLayer
#         # Just do it the same number of times and accumulate gradients
#         X, X_padded = self.X_cache
        
#         batch_size, height, width, channels = X.shape
#         _, out_height, out_width, out_channels = d_out.shape

#         # TODO: Implement backward pass
#         # Same as forward, setup variables of the right shape that
#         # aggregate input gradient and fill them for every location
#         # of the output
        
#         d_input_pad = np.zeros_like(X_padded)
#         W_step = self.W.value.reshape(self.filter_size*self.filter_size*channels, out_channels)
#         # Try to avoid having any other loops here too
#         for y in range(out_height):
#             for x in range(out_width):
#                 # TODO: Implement backward pass for specific location
#                 # Aggregate gradients for both the input and
#                 # the parameters (W and B)
#                 d_out_step = d_out[:,x,y,:].reshape(batch_size,out_channels)
#                 X_step = X_padded[:,x:x+self.filter_size,y:y+self.filter_size,:]
#                 res = np.dot(d_out_step,np.transpose(W_step)).reshape(X_step.shape)
#                 d_input_pad[:,x:x+self.filter_size,y:y+self.filter_size,:]+=res
#                 X_step = X_step.reshape(batch_size,self.filter_size*self.filter_size*channels)
#                 self.W.grad += np.dot(np.transpose(X_step),d_out_step).reshape(self.W.grad.shape)
#                 self.B.grad += np.sum(d_out_step, axis=0)
#         if self.padding>0:
#             d_input = d_input_pad[:,self.padding:height+self.padding,self.padding:width+self.padding,:]
#         else: d_input = d_input_pad
        
#         return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        self.X = X.copy()

        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        
        out = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                out[:, y, x, :] = np.amax(X_slice, axis=(1, 2))

        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        
        extra_offset = 1
        out_height = int((height - self.pool_size) / self.stride) + extra_offset
        out_width = int((width - self.pool_size) / self.stride) + extra_offset
        
        out = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                mask = (X_slice == np.amax(X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                out[:, y:y + self.pool_size, x:x + self.pool_size, :] += grad * mask
                
        return out

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X = X.copy()
        return X.reshape(batch_size, height*width*channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X.shape)

    def params(self):
        # No params!
        return {}
