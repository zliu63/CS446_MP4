"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    pass
    x = data['image']
    y = data['label']
    N = x.shape[0]
    num_steps_per_epoch = np.ceil(N/batch_size)
    num_epoch = np.ceil(num_steps/num_steps_per_epoch)
    count = 0
    for epoch in range(int(num_epoch)):
        #shuffle data if needed
        if shuffle:
            x,y = unison_shuffled(x,y)
        for i in range(int(num_steps_per_epoch)):
            if count >= num_steps:
                return model
            if i == (num_steps_per_epoch - 1):
                x_batch = x[i*batch_size:N]
                y_batch = y[i*batch_size:N]
            else:
                x_batch = x[i*batch_size:(i+1)*batch_size]
                y_batch = y[i*batch_size:(i+1)*batch_size]
            model = update_step(x_batch,y_batch,model,learning_rate)
            count += 1
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    pass
    f = model.forward(x_batch)
    g = model.backward(f,y_batch)
    model.w = model.w - learning_rate*g
    return model


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    pass
    # Set model.w
    model.w = None


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    P = None
    q = None
    G = None
    h = None
    # Implementation here.
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    x = data['image']
    y = data['label']
    y_hat = model.forward(x)
    f = model.predict(y_hat)
    loss = model.total_loss(f,y)
    N = y.shape[0]
    err = 0
    for i in range(N):
        if y[i][0] != f[i][0]:
            err += 1
    acc = 1 - float(err)/float(N)
    return loss, acc

def unison_shuffled(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
