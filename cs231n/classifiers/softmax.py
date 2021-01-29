from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. D=3073, C=10 (cifar)
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = len(X)
    for i in range(n):
        f = X[i] @ W
        f -= np.max(f)
        den = np.sum(np.exp(f))
        loss -= np.log(np.exp(f[y[i]]) / den)
        for j in range(len(f)):
            dW[:, j] += X[i] * np.exp(f[j]) / den
        dW[:, y[i]] -= X[i]
    dW = dW / n + reg * W
    loss = loss / n + reg * np.sum(np.square(W)) / 2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = len(X)
    f = X @ W
    f -= np.max(f, axis=1)[:, np.newaxis]
    exp_f = np.exp(f)
    correct_class_idx = y + W.shape[1] * np.arange(n)
    softmax = exp_f / np.sum(exp_f, axis=1)[:, np.newaxis]
    loss = np.sum(-np.log(np.take(softmax, correct_class_idx))) / n
    loss += 0.5 * reg * np.sum(W * W)
    softmax[np.arange(n), y] -= 1
    dW = X.T.dot(softmax) / n + reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
