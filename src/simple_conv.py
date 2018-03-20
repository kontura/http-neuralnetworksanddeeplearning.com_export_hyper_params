import ny_net3
from ny_net3 import Network
from ny_net3 import ReLU
from ny_net3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ThroughConvPoolLayer
training_data, validation_data, test_data = ny_net3.load_data_shared()
mini_batch_size = 10

#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(6, 1, 5, 5), poolsize=(2, 2)),
#    ThroughConvPoolLayer(mini_batch_size),
#    SoftmaxLayer(n_in=4*4*16, n_out=10)
#    ], mini_batch_size)
#

#radoby lenet5, ale nema vahy pro pooling a nedari se mi propojit konvoluci naskrz.
#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(6, 1, 5, 5), poolsize=(2, 2)),
#    ConvPoolLayer(image_shape=(mini_batch_size, 6, 12, 12), filter_shape=(16, 6, 3, 3), poolsize=(2, 2)),
#    ConvPoolLayer(image_shape=(mini_batch_size, 16, 5, 5), filter_shape=(120, 16, 5, 5), poolsize=(1, 1)),
#    FullyConnectedLayer(n_in=120, n_out=84),
#    SoftmaxLayer(n_in=84, n_out=10)
#    ], mini_batch_size)

#good net from book
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20 , 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU, p_dropout=0.5),
    SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)
    ], mini_batch_size)
#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
#    ConvPoolLayer(image_shape=(mini_batch_size, 20 , 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
#    SoftmaxLayer(n_in=40*4*4, n_out=10, p_dropout=0.5)
#    ], mini_batch_size)
#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2), activation_fn=ReLU),
#    SoftmaxLayer(n_in=20*12*12, n_out=10, p_dropout=0.5)
#    ], mini_batch_size)
net.SGD(training_data, 5, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

