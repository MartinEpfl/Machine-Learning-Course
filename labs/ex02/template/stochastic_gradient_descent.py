# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""    
    solution =  -(1/y.shape[0])*(np.transpose(tx) @ (y - (tx @ w)))
    return solution


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                gradient = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
                loss = compute_loss(minibatch_y,minibatch_tx,w)
                w = w - gamma*gradient

        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    """Stochastic gradient descent algorithm."""
    return losses, ws