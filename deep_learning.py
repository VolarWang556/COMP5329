import numpy as np

train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')


def initialize_parameters(layers_dim):
    """
    :param layers_dim:  a list [d0,d1,d2...] d0 is dim of examples, d1 is 1st hidden layer...
    :return parameter:  include all wight and bias(initialize one)
    """

    parameter = {}
    lay_number = len(layers_dim)

    # loop lay_times to initialize the parameters
    for i in range(1, lay_number):
        parameter["W" + str(i)] = np.random.randn(layers_dim[i], layers_dim[i - 1])
        parameter["b" + str(i)] = np.zeros((layers_dim[i], 1))

    return parameter


def forward_linear_part(A_before, W, b):
    """
    :param A_before: the activation values in the previous layer
    :param W: the weight matrix for this layer
    :param b: the bias vector for this layer
    :return: flash(A,W,b)
    """
    Z = np.dot(W, A_before) + b
    flash = (A_before, W, b)

    return Z, flash


# need to improve fo more activation function
# to this project "softmax" must be added
def forward_activation_part(A_before, W, b, activation="relu"):
    """
    :param A_before: the activation values in the previous layer
    :param W: the weight matrix for this layer
    :param b: the bias vector for this layer
    :param activation: the type of activation
    :return: flash_m(A,W,b,Z_next)
    """

    if activation == "relu":
        Z, lin_flash = forward_linear_part(A_before, W, b)
        A = np.maximum(0, Z)
        activation_flash = Z

    elif activation == "sigmoid":
        Z, lin_flash = forward_linear_part(A_before, W, b)
        A = 1 / (1 + np.exp(-Z))
        activation_flash = Z

    elif activation == "softmax":
        Z, lin_flash = forward_linear_part(A_before, W, b)
        exp_Z = np.exp(Z)
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        activation_flash = Z

    else:
        raise ValueError("Invalid input(activation function)")

    flash_m = (lin_flash, activation_flash)

    return A, flash_m


# can switch different types of activation function
def whole_forward(X, parameter):
    """
    :param X:  the sample matrix (number of features ,number of samples)
    :param parameter:  the parameter set contains W and b
    :return: the last A and flashes contains all flash_m(A,W,b,Z_next)
    """
    layer_number = len(parameter)//2  # every layer has W and b two parameters, so can use this to got layer numbers
    flashes = []

    A = X

    # loop to do the forward to every layers
    for i in range(1, layer_number):
        A, flash_m = forward_activation_part(A, parameter['W' + str(i)], parameter['b' + str(i)], activation="relu")
        flashes.append(flash_m)

    # the last layer
    A_last, flash_m = forward_activation_part(A, parameter['W' + str(layer_number)],
                                              parameter['b' + str(layer_number)], activation="sigmoid")
    flashes.append(flash_m)


    return A_last, flashes


# need to improve(best to choose 交叉熵损失)
def compute_cost(AL, Y):
    """
    cal cost
    """
    m = Y.shape[1]  # number of examples
    error = Y - AL
    cost = (1/m)*np.sum(error**2)
    cost = np.squeeze(cost)

    return cost


def backward_activation_part(dA, flash_m, activation="relu"):
    """
    the reverse operation of forward forward_activation_part
    :param dA:
    :param flash_m:(A,W,b,Z_next)
    :param activation:
    :return:
    """
    linear_cache, Z = flash_m

    if activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dA_prev, dW, db = backward_liner_part(dZ, linear_cache)

    elif activation == "sigmoid":
        s = 1/(1 + np.exp(-Z))
        dZ = dA * s * (1-s)
        dA_prev, dW, db = backward_liner_part(dZ, linear_cache)

    elif activation == "softmax":
        dZ = dA
        dA_prev, dW, db = backward_liner_part(dZ, linear_cache)

    else:
        raise ValueError("Invalid input(activation function)")

    return dA_prev, dW, db


def backward_liner_part(dZ, flash):
    """the reverse operation of forward propagation"""
    A_prev, W,b = flash

    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def L_model_backward(AL, Y, flashes):
    """
    the whole backward
    :param AL:
    :param Y:
    :param flashes:
    :return:
    """
    grads = {}
    L = len(flashes)  # the number of layers
    m = AL.shape[1]

    dAL = - (Y - AL)

    current_flash_m = flashes[-1]
    dA_prev_temp, dW_temp, db_temp = backward_activation_part(dAL, current_flash_m, activation="sigmoid")
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for i in reversed(range(L - 1)):
        # lth layer gradients.
        current_flash_m = flashes[i]
        dA_prev_temp, dW_temp, db_temp = backward_activation_part(grads["dA" + str(i + 1)], current_flash_m,
                                                                  activation="relu")
        grads["dA" + str(i)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return parameters





#试验区
train_data = train_data.T
train_label = train_label.T

# 设置网络层维度（例如：[输入特征数, 隐藏层1节点数, ..., 输出层节点数]）
layers_dims = [128, 5, 2, 1]  # 示例：输入层有train_data.shape[0]个特征，1个隐藏层有5个节点，输出层有1个节点

# 初始化参数
parameters = initialize_parameters(layers_dims)

# 设置学习率和迭代次数
learning_rate = 0.005
num_iterations = 3000

# 训练循环
for i in range(num_iterations):

    # 前向传播
    AL, caches = whole_forward(train_data, parameters)

    # 计算损失
    cost = compute_cost(AL, train_label)

    # 后向传播
    grads = L_model_backward(AL, train_label, caches)

    # 更新参数
    parameters = update_parameters(parameters, grads, learning_rate)

    # 每100次迭代打印损失
    if i % 100 == 0:
        print(f"Cost after iteration {i}: {cost}")
    

