import numpy as np

VERY_LARGE_NUM = 1e20
VERY_SMALL_NUM = 1e-20

def compute_loss(y, tx, w):
    """
    Compute the loss using MSE

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    w : numpy.ndarray
        vector of weights of size D

    Returns
    -------
    float
        The loss computed using MSE

    """
    N = y.shape[0]
    e = y - tx.dot(w)
    loss = 1/(2*N)*e.T@e
    return loss

def compute_gradient(y, tx, w):
    """
    Compute the gradient of MSE

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    w : numpy.ndarray
        vector of weights of size D

    Returns
    -------
    float
        The gradient of MSE

    """

    N = y.shape[0]
    e = y - tx@w
    grad = -1/N*tx.T@e
    return grad

def least_squares_GD(y, tx, initial_w=None, max_iters=1000, gamma=1, early_stopping=False, tol=1e-5,max_n_iter_no_change=3,**kwargs):
    """
    Find optimal weights and loss using gradient descent

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    initial_w : numpy.ndarray
        vector of weights of size D
    max_iters : int
        the maximum number of iteration of the algorithm, default 1000
    gamma : float
        the initial step size, default 1
    early_stopping: bool
        whether to use early stopping, if not use constant step size equal to initial step size, default False
    tol: float
        the stopping criterion, default 1e-5
    max_n_iter_no_change: int
        number of iterations with no improvement to wait before stopping fitting, default 3
    Returns
    -------
    (numpy.ndarray, float)
        optimal weights, loss

    """
    if initial_w is None:
        initial_w = np.ones(tx.shape[1])

    w = initial_w
    best_w = w
    best_loss=VERY_LARGE_NUM
    n_iter_no_change=0

    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        if early_stopping:
            loss = compute_loss(y, tx, w)
            if loss > best_loss - tol:
                n_iter_no_change+=1
                if n_iter_no_change >= max_n_iter_no_change:
                    # print("Early stop after {} steps".format(n_iter))
                    # break
                    gamma = gamma/5
                    if gamma < 0.001:
                        # print("Early stop after {} steps".format(n_iter))
                        break
                    else:
                        # print("Lr decrease to {} after {} steps".format(gamma, n_iter))
                        pass
                    n_iter_no_change = 0
            else:
                best_loss = loss 
                best_w = w
                n_iter_no_change = 0
    if early_stopping:
        return best_w, best_loss
    else:
        return w, compute_loss(y, tx, w)

def least_squares_SGD(y, tx, initial_w=None, max_iters=1000, gamma=1, early_stopping=False, tol=1e-5,max_n_iter_no_change=3, **kwargs):
    """
    Find optimal weights and loss using stochastic gradient descent

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    initial_w : numpy.ndarray
        vector of weights of size D
    max_iters : int
        the maximum number of iteration of the algorithm, default 1000
    gamma : float
        the initial step size, default 1
    early_stopping: bool
        whether to use early stopping, if not use constant step size equal to initial step size, default False
    tol: float
        the stopping criterion, default 1e-5
    max_n_iter_no_change: int
        number of iterations with no improvement to wait before stopping fitting, default 3
    Returns
    -------
    (numpy.ndarray, float)
        optimal weights, loss

    """
    if initial_w is None:
        initial_w = np.ones(tx.shape[1])

    w = initial_w
    batch_size = 1
    best_w = w
    best_loss=VERY_LARGE_NUM
    n_iter_no_change=0

    for n_iter, (m_y, m_tx) in enumerate(batch_iter(y, tx, batch_size, num_batches=max_iters)):
        grad = compute_gradient(m_y, m_tx, w,**kwargs)
        w = w - gamma*grad
        if early_stopping:
            loss = compute_loss(y, tx, w)
            if loss > best_loss - tol:
                n_iter_no_change+=1
                if n_iter_no_change >= max_n_iter_no_change:
                    # print("Early stop after {} steps".format(n_iter))
                    # break
                    gamma = gamma/5
                    if gamma < 0.001:
                        # print("Early stop after {} steps".format(n_iter))
                        break
                    else:
                        # print("Lr decrease to {} after {} steps".format(gamma, n_iter))
                        pass
                    n_iter_no_change = 0
            else:
                best_loss = loss 
                best_w = w
                n_iter_no_change = 0
    if early_stopping:
        return best_w, best_loss
    else:
        return w, compute_loss(y, tx, w)

def least_squares(y, tx,**kwargs):
    """
    Find optimal weights and loss using least squares

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)

    Returns
    -------
    (numpy.ndarray, float)
        optimal weights, loss

    """

    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_=1,**kwargs):
    """
    Find optimal weights and loss using ridge regression

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    lambda : float
        the regularization term, default 1

    Returns
    -------
    (numpy.ndarray, float)
        optimal weights, loss

    """

    N, D = tx.shape
    I = np.identity(D)
    w = np.linalg.solve(tx.T@tx + lambda_*2*N*I, tx.T@y)
    loss = compute_loss(y, tx, w) + lambda_*(w**2).sum()
    return w, loss

def split_data(x, y, ratio, seed=1):
    """
    Split the dataset using the split ratio

    Parameters
    ----------
    x : numpy.ndarray
        matrix of features of size (NxD)
    y : numpy.ndarray
        vector of labels of size N
    ratio : float
        the percentage of data used in the training set (between 0 and 1)
    seed : int
        a seed to initialize the random number generator

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        features of the train set, labels of the train set,
        features of the test set, labels of the test set

    """
    # set seed
    np.random.seed(seed)
    N = x.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    mid = round(N*ratio)
    indices_train = indices[:mid]
    indices_test = indices[mid:]
    x_train = x[indices_train]
    y_train = y[indices_train]
    x_test = x[indices_test]
    y_test = y[indices_test]
    return x_train, y_train, x_test, y_test

def sigmoid(t):
    """
    Apply sigmoid function on t.

    Parameters
    ----------
    t : float, numpy.ndarray,...
        input value
    Returns
    -------
    same type as input
        value after applying sigmoid

    """
    return 1.0 / (1 + np.exp(np.clip(-t, -709.78, 709.78)))

def compute_log_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood.
    
    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    w : numpy.ndarray
        vector of weights of size D
    Returns
    -------
    float
        log loss corresponding to {-1,1} labels
    """
    pred = sigmoid(tx.dot(w))
    loss = (y + 1).T.dot(np.log(pred + VERY_SMALL_NUM)) + (1 - y).T.dot(np.log(1 - pred + VERY_SMALL_NUM))
    return np.squeeze(-loss/2)/len(y)

def compute_log_gradient(y, tx, w,**kwargs):
    """
    Compute the gradient of negative log likelihood.
    
    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    w : numpy.ndarray
        vector of weights of size D
    Returns
    -------
    numpy.ndarray
        gradient of weights of size D
    """
    pred = sigmoid(tx.dot(w))
    if 'weight_loss' in kwargs:
        weight = np.where(y > 0, kwargs['weight_loss'][1], kwargs['weight_loss'][0])
        grad = tx.T.dot((pred - y/2 - 1/2)*weight)/len(y)
    else:
        grad = tx.T.dot((pred - y/2 - 1/2))/len(y)
    return grad

def logistic_regression(y, tx, initial_w=None, max_iters=1000, gamma=1, early_stopping=False, tol=1e-5,max_n_iter_no_change=3,**kwargs):
    """
    Find optimal weights and loss using logistic regresssion

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    initial_w : numpy.ndarray
        vector of weights of size D
    max_iters : int
        the maximum number of iteration of the algorithm, default 1000
    gamma : float
        the initial step size, default 1
    early_stopping: bool
        whether to use early stopping, if not use constant step size equal to initial step size, default False
    tol: float
        the stopping criterion, default 1e-5
    max_n_iter_no_change: int
        number of iterations with no improvement to wait before stopping fitting, default 3
    Returns
    -------
    (numpy.ndarray, float)
        optimal weights, loss

    """
    if initial_w is None:
        initial_w = np.ones(tx.shape[1])

    w = initial_w
    best_w = w
    best_loss=VERY_LARGE_NUM
    n_iter_no_change=0
    for n_iter in range(max_iters):
        w -= gamma * compute_log_gradient(y, tx, w,**kwargs)
        if early_stopping:
            loss = compute_log_loss(y, tx, w)
            if loss > best_loss - tol:
                n_iter_no_change+=1
                if n_iter_no_change >= max_n_iter_no_change:
                    # print("Early stop after {} steps".format(n_iter))
                    # break
                    gamma = gamma/5
                    if gamma < 0.001:
                        # print("Early stop after {} steps".format(n_iter))
                        break
                    else:
                        # print("Lr decrease to {} after {} steps".format(gamma, n_iter))
                        pass
                    n_iter_no_change = 0
            else:
                best_loss = loss 
                best_w = w
                n_iter_no_change = 0
    if early_stopping:
        return best_w, best_loss
    else:
        return w, compute_log_loss(y, tx, w)

def reg_logistic_regression(y, tx, lambda_=1, initial_w=None, max_iters=1000, gamma=1, early_stopping=False, tol=1e-5,max_n_iter_no_change=3, penalty='l2',**kwargs):
    """
    Find optimal weights and loss using regularized logistic regression

    Parameters
    ----------
    y : numpy.ndarray
        vector of labels of size N
    tx : numpy.ndarray
        matrix of features of size (NxD)
    lambda_ : float
        the regularization term
    initial_w : numpy.ndarray
        vector of weights of size D
    max_iters : int
        the maximum number of iteration of the algorithm, default 1000
    gamma : float
        the initial step size, default 1
    early_stopping: bool
        whether to use early stopping, if not use constant step size equal to initial step size, default False
    tol: float
        the stopping criterion, default 1e-5
    max_n_iter_no_change: int
        number of iterations with no improvement to wait before stopping fitting, default 3
    penalty: string
        the penalty (aka regularization term) to be used, 'l1' or 'l2', defautl 'l2'
    Returns
    -------
    (numpy.ndarray, float)
        optimal weights, loss

    """
    assert penalty in ['l1','l2'], "penalty must be l1 or l2"
    if initial_w is None:
        initial_w = np.ones(tx.shape[1])

    w = initial_w
    best_w = w
    best_loss=VERY_LARGE_NUM
    n_iter_no_change=0

    for n_iter in range(max_iters):
        if penalty=='l2':
            grad = compute_log_gradient(y, tx, w) + lambda_ *w
        elif penalty=='l1':
            grad = compute_log_gradient(y, tx, w) + lambda_ *np.sign(w)
        
        w -= gamma * grad
        if early_stopping:
            if penalty=='l2':
                loss = compute_log_loss(y, tx, w,**kwargs) + lambda_ * (w**2).sum()
            elif penalty=='l1':
                loss = compute_log_loss(y, tx, w,**kwargs) + lambda_ * np.abs(w).sum()
            
            if loss > best_loss - tol:
                n_iter_no_change+=1
                if n_iter_no_change >= max_n_iter_no_change:
                    # print("Early stop after {} steps".format(n_iter))
                    # break
                    gamma = gamma/5
                    if gamma < 0.001:
                        # print("Early stop after {} steps".format(n_iter))
                        break
                    else:
                        # print("Lr decrease to {} after {} steps".format(gamma, n_iter))
                        pass
                    n_iter_no_change = 0
            else:
                best_loss = loss 
                best_w = w
                n_iter_no_change = 0
    if early_stopping:
        return best_w, best_loss
    else:
        if penalty=='l2':
            loss = compute_log_loss(y, tx, w,**kwargs) + lambda_ * (w**2).sum()
        elif penalty=='l1':
            loss = compute_log_loss(y, tx, w,**kwargs) + lambda_ * np.abs(w).sum()
        return w, loss 
        
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    Note: this function was taken from the labs.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
