def predict(network, input):
    output = input.T
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    # forward
    # print("x_train", x_train.shape)
    output = predict(network, x_train)
    # print("output", output.shape)

    # backward
    grad = loss_prime(y_train.T, output)
    for layer in reversed(network):
        grad = layer.backward(grad, learning_rate)
    
    