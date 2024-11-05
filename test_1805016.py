from network import train
import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch
def read_data():
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    independent_test_dataset = ds.EMNIST(root='./data',
                                split='letters',
                                train=False,
                                transform=transforms.ToTensor())
    return train_validation_dataset, independent_test_dataset

train_validation_dataset, independent_test_dataset = read_data()
# print(type(train_validation_dataset))
# train_size = int(0.8 * len(train_validation_dataset))
# val_size = len(train_validation_dataset) - train_size

# # from sklearn.model_selection import train_test_split
# train_dataset, val_dataset = random_split(train_validation_dataset, [train_size, val_size])
# train_dataset = DataLoader(train_dataset)
# val_dataset = DataLoader(val_dataset)


# print(train_dataset.data.size())
# print(type(val_dataset))
import numpy as np
x_test = independent_test_dataset.data.view(independent_test_dataset.data.shape[0], -1).numpy()
y_test = independent_test_dataset.targets
y_test = torch.nn.functional.one_hot(y_test, num_classes= 27).numpy().reshape(y_test.shape[0], 27)
test_data = np.hstack((x_test, y_test))

import sys
# sys.exit(0)

from dense import Dense
from dropout import Dropout
from activations import ReLU, Softmax
from losses import binary_cross_entropy, binary_cross_entropy_prime, mse, mse_prime
from network import train, predict
import pickle
file_path = 'model4.pkl'

# neural network
# Load data from the pickle file
with open(file_path, 'rb') as file:
    network = pickle.load(file)




# test
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix    

x_test = test_data[:, :-27].astype("float64") / 255
y_test = test_data[:, -27:].astype("float64")

print("x_test" ,x_test.shape, "y_test", y_test.shape)

correct = 0
wrong = 0
test_pred = []

def one_hot_encode(x):
    return np.array([1 if i == np.argmax(x) else 0 for i in range(27)])

for x, y in zip(x_test, y_test):
    x = x.reshape(1, x.shape[0])
    y = y.reshape(1, y.shape[0])
    output = predict(network, x)
    # print(output.shape)
    correct += np.argmax(output) == np.argmax(y)
    wrong += np.argmax(output) != np.argmax(y)
    test_pred.append(one_hot_encode(output))
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

test_accuracy = accuracy_score(y_test, test_pred)
test_f1_macro = f1_score(y_test, test_pred, average='macro')


table = [["model3", round(100 * test_accuracy, 2), round(100 * test_f1_macro, 2)],
			 ]
from tabulate import tabulate
print(tabulate(table, headers=["Model", "test_accuracy", "test_f1_macro"], tablefmt="grid"))


print("correct", correct, "wrong", wrong)
print(test_data.shape)
