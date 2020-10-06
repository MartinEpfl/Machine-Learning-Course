def compute_loss(y, tx, w):
    N = y.shape[0]
    e = y - (tx @ w)
    result = 1/(2*N) * (np.transpose(e) @ e)
    return result