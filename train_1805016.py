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
x_trainval = train_validation_dataset.data.view(train_validation_dataset.data.shape[0], -1).numpy()
y_trainval = train_validation_dataset.targets
y_trainval = torch.nn.functional.one_hot(y_trainval, num_classes= 27).numpy().reshape(y_trainval.shape[0], 27)
trainval_data = np.hstack((x_trainval, y_trainval))

from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(trainval_data, test_size=.2, random_state=420)

print(train_data.shape, val_data.shape)
import sys
# sys.exit(0)

from dense import Dense
from dropout import Dropout
from activations import ReLU, Softmax
from losses import binary_cross_entropy, binary_cross_entropy_prime, mse, mse_prime
from network import train, predict

batch_size = 128
epochs = 100
learning_rate = 0.005
limit = 128 * 1400

train_data = train_data[:limit]

# neural network
network = [
    Dense(28 * 28, 1000),
    ReLU(),
    Dense(1000, 27),
    Softmax()
]

import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
# cur = 0
# idx = -1
# for _ in range(epochs):
#     print("epoch", _)
#     np.random.shuffle(train_data)

#     x_train = train_data[:, :-27].astype("float64") / 255
#     y_train = train_data[:, -27:].astype("float64")
#     for i in range(0, len(x_train), batch_size):
#         x_batch = x_train[i:i+batch_size]
#         y_batch = y_train[i:i+batch_size]

#         # train
#         train(network, binary_cross_entropy, binary_cross_entropy_prime, x_batch, y_batch, epochs=epochs, learning_rate=learning_rate)

#     val_pred = []

#     def one_hot_encode(x):
#         return np.array([1 if i == np.argmax(x) else 0 for i in range(27)])

#     x_val = val_data[:, :-27].astype("float64") / 255
#     y_val = val_data[:, -27:].astype("float64")
#     for x, y in zip(x_val, y_val):
#         x = x.reshape(1, x.shape[0])
#         y = y.reshape(1, y.shape[0])
#         output = predict(network, x)
#         # print(output.shape)
#         val_pred.append(one_hot_encode(output))
#         # print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

#     val_accuracy = accuracy_score(y_val, val_pred)
#     val_f1_macro = f1_score(y_val, val_pred, average='macro')

#     print(f"Validation Accuracy: {val_accuracy:.4f}")
#     print(f"Validation Macro-F1: {val_f1_macro:.4f}")

#     if val_accuracy > cur:
#         cur = val_accuracy
#         idx = _
#         file_path = 'model4.pkl'

#         # Save data to a pickle file
#         with open(file_path, 'wb') as file:
#             pickle.dump(network, file)

#         print(f'Data saved to {file_path}')
# # train(network, mse, mse_prime, x_train, y_train, epochs=200, learning_rate=0.1)
# print(idx)

# Load data from the pickle file
file_path = 'model4.pkl'
with open(file_path, 'rb') as file:
    network = pickle.load(file)

# test
       

x_val = val_data[:, :-27].astype("float64") / 255
y_val = val_data[:, -27:].astype("float64")

print("x_val" ,x_val.shape, "y_val", y_val.shape)

correct = 0
wrong = 0
train_pred = []
val_pred = []

def one_hot_encode(x):
    return np.array([1 if i == np.argmax(x) else 0 for i in range(27)])

for x, y in zip(x_val, y_val):
    x = x.reshape(1, x.shape[0])
    y = y.reshape(1, y.shape[0])
    output = predict(network, x)
    # print(output.shape)
    correct += np.argmax(output) == np.argmax(y)
    wrong += np.argmax(output) != np.argmax(y)
    val_pred.append(one_hot_encode(output))
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

val_accuracy = accuracy_score(y_val, val_pred)
val_f1_macro = f1_score(y_val, val_pred, average='macro')

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Macro-F1: {val_f1_macro:.4f}")

x_train = train_data[:, :-27].astype("float64") / 255
y_train = train_data[:, -27:].astype("float64")
for x, y in zip(x_train, y_train):
    x = x.reshape(1, x.shape[0])
    y = y.reshape(1, y.shape[0])
    output = predict(network, x)
    # print(output.shape)
    correct += np.argmax(output) == np.argmax(y)
    wrong += np.argmax(output) != np.argmax(y)
    train_pred.append(one_hot_encode(output))
    # print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

train_accuracy = accuracy_score(y_train, train_pred)
train_f1_macro = f1_score(y_train, train_pred, average='macro')

table = [["model3", round(100 * train_accuracy, 2), round(100 * train_f1_macro, 2), round(100 * val_accuracy, 2), round(100 * val_f1_macro, 2)],
			 ]
from tabulate import tabulate
print(tabulate(table, headers=["Model", "train_accuracy", "train_f1_macro", "val_accuracy", "val_f1_macro"], tablefmt="grid"))

print("correct", correct, "wrong", wrong)
print(train_data.shape)

