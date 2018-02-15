"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1, 1).
        """
        reg_grad = None
        loss_grad = None
        # Implementation here.
        pass
        N = f.shape[0]
        K = self.w.shape[0]
        reg_grad = self.w_decay_factor*self.w
        loss_grad = np.zeros((K,1))
        for i in range(N):
            indicator = 1. if y[i][0]*f[i][0] < 1 else 0.
            x = self.x[i].reshape(K,1)
            loss_grad = loss_grad -y[i][0]*x*indicator
        total_grad = reg_grad + loss_grad
        return total_grad
        

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        # Implementation here.
        l2_loss = 0.5*self.w_decay_factor*(np.linalg.norm(self.w)**2)
        N = f.shape[0]
        z = np.matmul(y.transpose(),f)[0][0]
        hinge_loss = 0
        for i in range(N):
            z = max(0,1-y[i][0]*f[i][0])
            hinge_loss += z
        total_loss = hinge_loss + l2_loss
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        N = f.shape[0]
        y_predict = np.zeros((N,1))
        # Implementation here.
        for i in range(N):
            y_predict[i][0] = 1. if f[i][0] >= 0 else -1.
        return y_predict
