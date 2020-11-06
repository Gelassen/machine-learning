import numpy as np


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


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    predictions_copy = predictions.copy()
    
    probs = softmax(predictions_copy)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs

    if (len(predictions.shape) == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = np.arange(target_index.shape[0])
        dprediction[batch_size, target_index.flatten()] -= 1 
        dprediction = dprediction / target_index.shape[0]
        
#     print(dprediction)
#     dprediction = __calc_gradient_probs(probs, target_index)
    
    return loss, dprediction

def __calc_gradient_probs(softmax_probs, target_index):
    jacobian = np.diag(softmax_probs)
    for i in range(len(jacobian)):
        for j in range(len(jacobian)):
            if (i == j):
                jacobian[i][j] = softmax_probs[i] * (1 - softmax_probs[j]) # i or j here is do not matter as i == j
            else:
                jacobian[i][j] = -softmax_probs[i] * softmax_probs[j]
    
    return jacobian[target_index]

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # DONE: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops

    loss = reg_strength * np.sum(np.dot(W.T, W))

    batch_size = np.arange(W.shape[1])
    grad = np.array((list(map(lambda x: np.sum(W,axis=1), batch_size))))
    grad = 2 * reg_strength * np.transpose(grad)
    
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    
    predictions = np.dot(X, W)
    predictions_copy = predictions.copy()
    loss, dpred = softmax_with_cross_entropy(predictions, target_index) 
    
    probs = softmax(predictions_copy) 
    dW = np.dot(np.transpose(X), probs)
    
    p = np.zeros_like(probs)
    batch_size = np.arange(target_index.shape[0])
    p[batch_size,target_index.flatten()] = 1
    dW -= np.transpose(np.dot(np.transpose(p),X))
    dW = dW/target_index.shape[0]
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            size = sections.size
            print(size)
            for i in range(size):
                batch = X[batches_indices[i],:]
                target_index = y[batches_indices[i]]
                loss_W, dW = linear_softmax(batch, self.W, target_index)
                loss_l, dl = l2_regularization(self.W, reg)
                self.W = self.W - learning_rate*(dW+dl)
                loss = loss_W + loss_l
            
            loss_history.append(loss)

            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        prediction = np.dot(X, self.W)
        probs = softmax(prediction)
        y_pred = np.array(list(map(lambda x: x.argsort()[-1], probs)))

        return y_pred



                
                                                          

            

                
