import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#import network
#net = network.Network([784, 30, 10])
#net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)


f = open('test_letters', 'w+')

for i in range(0,9):
    f.write("const float32_t test_num" + str(i) + "[LETTER_SIZE] = {\n")
    f.write(" " + str(test_data[i][0]).replace("]", ",").replace("[", "")[:-2])
    f.write("\n};")
    f.write("\n\n");

f.close()

