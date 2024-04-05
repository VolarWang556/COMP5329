from model2 import module as nn
from model2 import Optim
from model2 import CrossEntropyLoss
import os
import numpy as np

# Load the Data
os.chdir("D:/dataset/COMP5329/Assignment1-Dataset")
X_train = np.load("train_data.npy")
y_train = np.load("train_label.npy")
X_test = np.load("test_data.npy")
y_test = np.load("test_label.npy")


class SimpleNet(nn):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 30)
        self.fc2 = nn.Linear(30, 10)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


model = SimpleNet()
output = model.forward(X_train)


# Optim.SGD(model)
