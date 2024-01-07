import copy
import numpy as np
import matplotlib.pyplot as plt


def load_data_multi():
    data = np.loadtxt("C:\\Users\\enis\\Desktop\\doviz.txt")
    x = data[:, :7]
    y = data[:, 7]
    return x, y


def compute_cost(x, y, w, b, lambda_):
    """
    compute cost
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
      lambda_ : model parameter
    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]  # Number of data
    n = len(w)
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b  # Predicted Y value
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)

    reg_cost_L2 = (lambda_ / (2 * m)) * np.sum(w ** 2)  # L2 Regularization term
    reg_cost_L1 = (lambda_ / m) * np.sum(np.abs(w))  # L1 Regularization term

    total_cost = cost + reg_cost_L2 + reg_cost_L1  # cost with regularization
    return total_cost


def compute_gradient(x, y, w, b, lambda_):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
      lambda_ : model parameter
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = x.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))  # the derivative array of w parameters
    # to be more efficient while using
    # numpy, we set all values to 0 at first
    dj_db = 0.  # the derivative of bias term

    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, learning_rate, threshold: float, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      learning_rate (float)       : Learning rate
      threshold (int)     : trashold value to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    j_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    while True:

        # Calculate the gradient and if change ratio < threshold stop, else continue to iterate
        dj_db, dj_dw = gradient_function(x, y, w, b, lambda_)
        change = learning_rate * dj_dw
        if np.all(np.abs(change) < threshold):
            break
        else:
            # Update Parameters using w, b, alpha and gradient
            w = w - learning_rate * dj_dw  # Updating parameters
            b = b - learning_rate * dj_db  # Updating bias term
            # Save cost J at each change
            j_history.append(cost_function(x, y, w, b, lambda_))

    return w, b, j_history  # return final w,b and J history for graphing


X_train, y_train = load_data_multi()
print("Type of x_train:", type(X_train))
print("First five elements of x_train are:\n", X_train[:5])
print("First five elements of y_train are:\n", y_train[:5])

independent_variables = 7
initial_w = np.random.rand(independent_variables)
initial_w0 = np.random.rand()
threshold = 0.0001
learning_rate = 0.001
lambda_ = 0.1
print("Rastgele Katsayılar:", initial_w)

w_final, b_final, J_hist = gradient_descent(X_train, y_train,
                                            initial_w, initial_w0,
                                            compute_cost, compute_gradient,
                                            learning_rate, threshold, lambda_)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m, _ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
cost = compute_cost(X_train, y_train, w_final, b_final, lambda_)

# print(f"Cost Length: {len(J_hist)}")
# for i in range(0, len(J_hist), 10):
# print(*J_hist[i:i+10])

print(f'Cost at optimal w : {cost}')

x_new = [3.64, 3.84, 4.51, 0.4, 3.61, 2.73, 11.97]
numpy_array = np.array(x_new)
print(f"prediction for new x: {np.dot(x_new, w_final) + b_final:0.2f}")

# Gerçek ve tahmin edilen değerleri al
actual_values = y_train
predicted_values = np.dot(X_train, w_final) + b_final

# Scatter plot
plt.scatter(actual_values, actual_values, c='blue', label='Gerçek Değerler')
plt.scatter(actual_values, predicted_values, c='yellow', label='Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')

plt.legend()  # Efsaneleri (legends) göster
plt.show()
