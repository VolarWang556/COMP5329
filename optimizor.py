import numpy as np

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

    def step(self, parameters, grads, learning_rate):
        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter
        for l in range(L):
            parameters["W" + str(l + 1)] -= learning_rate * grads[f"dW{l + 1}"]
            parameters["b" + str(l + 1)] -= learning_rate * grads[f"db{l + 1}"]

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
    
