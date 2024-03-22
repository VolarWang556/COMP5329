import numpy as np


class module(object):
    class Parameters:
        def __init__(self):
            self.parameters = []
            self.layer_type = []
            self.initial = False
            self.update_index = 0
            self.his_grad = []

        def update(self, value_lst, status=1):
            if self.initial is False:
                # It is the first time that this layer's parameters are added to class "Parameters". 
                if status == 1:  # This means that this layer contains the weight which needed to be upgraded
                    self.parameters.append(value_lst)
                    # If corresponding layer_type is 1, then value_lst should be [input, weight, output]
                    # If layer_type is 0, then value_lst should be [input, output]
                    # If layer_type is -1, then value_lst should be [output, target].
                    # This means the loss function, also the final layer of NN.
                    self.layer_type.append(1)
                    self.update_index += 1
                    self.his_grad.append(0)
                elif status == 2:  # This means the batch_normalization layer.
                    self.parameters.append(value_lst)
                    self.layer_type.append(2)
                    self.update_index += 1
                elif status == 0:
                    self.parameters.append(value_lst)
                    self.layer_type.append(0)
                    self.update_index += 1
                    self.his_grad.append(0)
                elif status == -1:
                    self.parameters.append(value_lst)
                    self.layer_type.append(-1)
                    self.initial = True  # This means that the all the layers have been initialized.
                    self.his_grad.append(0)
                else:
                    raise TypeError("The status of layer can only be 1 or 0 or -1.")
            else:
                self.parameters[self.update_index] = value_lst
                if self.layer_type[self.update_index] == -1:
                    pass
                else:
                    self.update_index += 1

    class Linear(Parameters):
        def __init__(self, input_size, output_size, bias=True, mode="kaiming_normal"):
            # Initialization function, defining the linear transformation parameters
            # Initialize weights based on the specified mode
            def kaiming_norm_init(fan_in, fan_out):
                std = np.sqrt(2 / fan_in)
                weights = np.random.randn(*(fan_in, fan_out)) * std
                return weights

            def kaiming_uniform_init(fan_in, fan_out):
                std = np.sqrt(2 / fan_in)
                a = np.sqrt(3) * std
                weights = np.random.uniform(-a, a, size=(fan_in, fan_out))
                return weights

            self.bias = bias
            self.mode = mode
            if self.bias:
                self.input = input_size + 1  # Include bias in input size
            else:
                self.input = input_size
            self.output = output_size
            # Initialize weights based on the specified mode
            if self.mode == "kaiming_normal":
                self.weight = kaiming_norm_init(self.input, self.output)
            elif self.mode == "zero":
                self.weight = np.zeros((self.input, self.output))
            elif self.mode == "kaiming_uniform":
                self.weight = kaiming_uniform_init(self.input, self.output)
            else:
                raise TypeError("Mode input is wrong. You can only input 'kaiming_normal', 'zero' or 'kaiming_uniform'")
            self.linear_initial = 0

        def __call__(self, x):
            # Callable object, calls the forward function
            if self.bias:
                x_ = np.hstack((x, np.ones((x.shape[0], 1))))  # Include bias in input
            else:
                x_ = x
            return self.forward(x_)

        def forward(self, matrix):
            # Forward propagation function, performs linear transformation
            if matrix.shape[1] != self.input:
                raise ValueError("The number of neurons does not corresponding the feature size you input.")
            else:
                if self.linear_initial == 1:
                    self.weight = super().parameters[self.update_index][1]
                output = matrix @ self.weight
                super().update([matrix, self.weight, output], 1)
                self.linear_initial = 1
                return output  # Matrix multiplication to compute the output

    class ReLU(Parameters):
        def __init__(self, mode="Relu"):
            self.mode = mode

        def __call__(self, x):
            return self.forward(x)

        def forward(self, x):
            if self.mode == "Relu":
                output = np.maximum(x, 0)
                super().update([x, output], 0)
                return output

    class CrossEntropyLoss(Parameters):

        def __init__(self, weight=None, reduction='mean', label_smoothing=0.0):
            # Initialization function, defining parameters of the loss function
            self.weight = weight
            self.reduction = reduction
            self.label_smoothing = label_smoothing
            # self.grad = 0  # "grad" is used to calculate the gradient. It is a gradient matrix.

        def __call__(self, output, target):
            # Callable object, calls the forward function
            return self.forward(output, target)

        def forward(self, output, target):
            # Forward propagation function, computes the cross-entropy loss
            if self.label_smoothing == 0:
                # If label smoothing parameter is 0, no label smoothing is performed
                matrix = np.zeros((len(target), target.max()))
                for i in range(0, len(target)):
                    matrix[i, int(target[i][0])] = 1

            elif 0 < self.label_smoothing <= 1:
                # If label smoothing parameter is between (0,1], perform label smoothing
                matrix = np.full((len(target), target.max() + 1), self.label_smoothing / (target.max() + 1))
                for i in range(0, len(target)):
                    matrix[i, int(target[i][0])] = (1 - self.label_smoothing) * 1 + self.label_smoothing / (
                            target.max() + 1)
            else:
                raise ValueError("Label_smoothing value can only dropped in [0,1]")

            softmax = module.Softmax()  # Create Softmax object
            output = softmax(output)  # Apply Softmax to the output
            if self.weight is None:
                target = matrix  # If weight is not set, use the label matrix
            else:
                target = self.weight * matrix  # Adjust the label matrix using weights
            # self.grad = target - target / output  # Compute gradient
            super().update([output, target], -1)
            output2 = -np.log2(output)  # Perform logarithmic transformation on the Softmax output
            loss_vec = np.sum(target * output2, axis=1)  # Compute the loss vector for each sample
            if self.reduction == "mean":
                loss = np.sum(loss_vec) / len(loss_vec)  # Compute mean loss
                return loss
            elif self.reduction == "sum":
                loss = np.sum(loss_vec)  # Compute total loss
                return loss
            elif self.reduction == "none":
                return loss_vec  # Return the loss vector for each sample
            else:
                raise TypeError("Reduction input is wrong. You can only input 'mean', 'sum' or 'none'.")

    class Softmax:
        def __init__(self, dim=None):
            self.dim = dim

        def __call__(self, x):
            return self.forward(x)

        def forward(self, x):
            if self.dim is None:
                exp = np.exp(x)
                return exp / np.sum(exp, axis=1, keepdims=True)

    class BatchNormalization(Parameters):
        def __init__(self, num_features, eps=1e-05, momentum=0.1, tracking_running_stats=True):
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.tracking_running_stats = tracking_running_stats
            # This means that we will use gamma and beta, y = gamma*X+beta
            self.gamma = 1
            self.beta = 0
            if tracking_running_stats:
                self.global_mean = 0
                self.global_variance = 0

        def __call__(self, x):
            x_ = x
            return self.forward(x_)

        def forward(self, x):
            x_mean = np.mean(x, axis=0).reshape(1, -1)
            x_var = np.var(x, axis=0).reshape(1, -1)
            value_lst = []
            if self.tracking_running_stats:
                self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * self.global_mean
                self.global_variance = (1 - self.momentum) * self.global_variance + self.momentum * self.global_variance
                value_lst.append(self.global_mean)
                value_lst.append(self.global_variance)
            x_ = x - x_mean/np.sqrt(x_var+self.eps)
            output = self.gamma * x_ + self.beta
            value_lst.append(self.gamma)
            value_lst.append(self.beta)
            super().update(value_lst, 2)
            return output


class Optim(object):
    class SGD(module.Parameters):
        def __init__(self, learning_rate=0.01, momentum=0.9):
            self.learning_rate = learning_rate
            self.momentum = momentum

        def step(self):
            grad = 1
            while self.update_index != -1:
                if self.layer_type[self.update_index] == -1:  # -1 代表损失函数
                    output = self.parameters[self.update_index][0]
                    target = self.parameters[self.update_index][1]
                    grad *= target - target / output
                    self.update_index -= 1
                elif self.layer_type[self.update_index] == 0:  # 0 意味着激活函数ReLU
                    output = np.zeros_like(self.parameters[self.update_index][1])
                    output[self.parameters[self.update_index][1] > 0] = 1
                    grad *= output
                    self.update_index -= 1
                elif self.layer_type[self.update_index] == 1:  # 1 代表全连接层
                    input_value = self.parameters[self.update_index][0]
                    # update the weight
                    vt = self.his_grad[self.update_index] * self.momentum + self.learning_rate * input_value.T @ grad
                    weight = self.parameters[self.update_index][1] - vt
                    super().update([self.parameters[self.update_index][0], weight,
                                    self.parameters[self.update_index][2]])
                    self.his_grad[self.update_index] = vt
                    self.update_index -= 1
                elif self.layer_type[self.update_index] == 2:  #代表batchnorm层
                    parameters = self.parameters[self.update_index]
                    if len(parameters) == 2:

            self.update_index = 0
