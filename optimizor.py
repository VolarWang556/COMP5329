import numpy as np
import deep_learning as dl

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, parameters, grads, learning_rate):
        pass

    @staticmethod
    def initialize_parameters(layers_dim):
        parameter = {}
        lay_number = len(layers_dim)

        # loop lay_times to initialize the parameters
        for i in range(1, lay_number):
            parameter["W" + str(i)] = np.zeros(layers_dim[i], layers_dim[i - 1])
            parameter["b" + str(i)] = np.zeros((layers_dim[i], 1))

        return parameter

class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, parameters, grads):
        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter
        for l in range(L):
            parameters["W" + str(l + 1)] -= self.learning_rate * grads[f"dW{l + 1}"]
            parameters["b" + str(l + 1)] -= self.learning_rate * grads[f"db{l + 1}"]

        return parameters

class SGDM(Optimizer):
    def __init__(self, layers_dim, learning_rate, gamma):
        super().__init__(learning_rate)
        self.v = self.initialize_parameters(layers_dim)
        self.gamma = gamma # momentum cofficient
        
    def step(self, parameters, grads):
        L = len(parameters) // 2  # number of layers in the neural network
        # Update rule for each parameter
        for l in range(L):
            self.v[f"dW{l + 1}"] = self.v[f"dW{l+1}"] + grads[f"dW{l + 1}"] # weight momentum update
            self.v[f"db{l + 1}"] = self.v[f"db{l+1}"] + grads[f"db{l + 1}"] # bias momentum update
            parameters[f"W{l + 1}"] = self.gamma * self.v[f"dW{l+1}"] - self.learning_rate * grads[f"dW{l + 1}"]
            parameters[f"b{l + 1}"] = self.gamma * self.v[f"db{l+1}"] - self.learning_rate * grads[f"db{l + 1}"]
        return parameters
        
class Adam(Optimizer):
    def __init__(self, layers_dim, learning_rate, beta1, beta2, epsilon):
        super().__init__(learning_rate)
        self.v = self.initialize_parameters(layers_dim)
        self.m = self.initialize_parameters(layers_dim)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    def step(self, parameters, grads):
        L = len(parameters) // 2  # number of layers in the neural network
        # Update rule for each parameter
        for l in range(L):
            # weight part
            self.v[f"dW{l + 1}"] = self.beta2 * self.v[f"dW{l+1}"] + (1-self.beta2)*grads[f"dW{l + 1}"]**2 # v update
            self.m[f"dW{l + 1}"] = self.beta1 * self.m[f"dW{l+1}"] + (1-self.beta1)*grads[f"dW{l + 1}"] # m update
            v_hat = self.v[f"dW{l+1}"]/(1-self.beta2**self.t) # v bias correction
            m_hat = self.m[f"dW{l+1}"]/(1-self.beta1**self.t) # m bias correction
            parameters[f"W{l + 1}"] -=  (self.learning_rate * m_hat)/np.sqrt((v_hat - self.epsilon))
            # bias part
            self.v[f"db{l + 1}"] = self.beta2 * self.v[f"db{l+1}"] + (1-self.beta2)*grads[f"db{l + 1}"]**2# bias v update
            self.m[f"db{l + 1}"] = self.beta1 * self.m[f"db{l+1}"] + (1-self.beta1)*grads[f"db{l + 1}"]# bias m update
            v_hat = self.v[f"db{l+1}"]/(1-self.beta2**self.t)
            m_hat = self.m[f"db{l+1}"]/(1-self.beta1**self.t)
            parameters[f"b{l + 1}"] -=  (self.learning_rate * m_hat)/np.sqrt((v_hat - self.epsilon))
        self.t += 1#time step forward
        return parameters
    
    
    
if __name__ == "__main__":
    #试验区
    train_data = np.load('train_data.npy')
    train_label = np.load('train_label.npy')
    train_data = train_data.T
    train_label = train_label.T

    # 设置网络层维度（例如：[输入特征数, 隐藏层1节点数, ..., 输出层节点数]）
    layers_dims = [128, 5, 2, 1]  # 示例：输入层有train_data.shape[0]个特征，1个隐藏层有5个节点，输出层有1个节点

    # 初始化参数
    parameters = dl.initialize_parameters(layers_dims)
    #opti = Adam(layers_dims, 0.005, 0.9, 0.999, 1e-8)
    #opti = SGD(0.005)
    opti = SGDM(layers_dims, 0.005, 0.9)
    # 设置学习率和迭代次数
    learning_rate = 0.005
    num_iterations = 3000

    # 训练循环
    for i in range(num_iterations):

        # 前向传播
        AL, caches = dl.whole_forward(train_data, parameters)

        # 计算损失
        cost = dl.compute_cost(AL, train_label)

        # 后向传播
        grads = dl.L_model_backward(AL, train_label, caches)

        # 更新参数
        parameters = opti.step(parameters, grads)

        # 每100次迭代打印损失
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")