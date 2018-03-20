
import numpy as np

def calc_size(shape, x):
    if len(shape) == 4: #convo layer
        return(shape[0] * shape[1] * shape[2] * (shape[3]+x))
    if len(shape) == 2: #fully connected layer
        return(shape[0] * shape[1])
    return(-1)


def extend_by_zeros(weights, image_shape, filter_shape):
    out = [[] for i in range(len(weights))]
    index = 0
    for five_by_five in weights:
        w = []
        #for line in five_by_five[0]:
        l = len(five_by_five[0]) - 1
        for i in range(0,l):
            w = np.concatenate((w, np.array(np.append(five_by_five[0][i], np.zeros((image_shape[3]-filter_shape[3])))) ), axis=0)
        w = np.concatenate((w, np.array(five_by_five[0][l]) ), axis=0)
        out[index].append(w[::-1]) # = np.concatenate((out, [w[::-1]] ), axis=1)
        index += 1
    return(out)



np.set_printoptions(threshold=999999999999999999999, linewidth=99999999999999)
str_to_export = ""

index = 0
for x in net.layers:
    str_to_export += "//" + str(x) + ": " + "weights: " + str(x.params[0].shape.eval()) + ", biases: " + str(x.params[1].shape.eval()) + "\n"
    str_to_export += "//weights:\n"
    if (len(x.params[0].get_value().shape) == 4): #layer is convolutional
        total_size = calc_size(x.params[0].shape.eval(), x.image_shape[3]-x.filter_shape[3])
        weights = np.rot90(np.rot90(x.params[0].get_value(),axes=(-2,-1)), axes=(-2,-1))  
        weights_z = extend_by_zeros(weights, x.image_shape, x.filter_shape)
        str_to_export += "const float32_t l"+str(index)+"_w_o["+str(total_size)+"] = {\n" + str(weights_z).replace("])],", "])],\n").replace("array(","").replace("[","").replace("]","").replace(")","") + "};\n"
        total_size = calc_size(x.params[0].shape.eval(), 0)
        str_to_export += "const float32_t l"+str(index)+"_w["+str(total_size)+"] = {\n" + str(weights.tolist()).replace("[","").replace("],",",\n").replace("]],",",\n").replace("],", ",\n").replace("]","") + "\n};\n"
    else:
        total_size = calc_size(x.params[0].shape.eval(), 0)
        str_to_export += "const float32_t l"+str(index)+"_w["+str(total_size)+"] = {\n" + str(x.params[0].get_value().tolist()).replace("[","").replace("],",",\n").replace("]]","\n") + "};\n"
    str_to_export += "\n"
    str_to_export += "//biases:\n"
    str_to_export += "float32_t l"+str(index)+"_b["+str(x.params[1].shape.eval()[0])+"] = {\n" + str((x.params[1].get_value()).tolist()).replace("[","").replace("]","") + "\n};\n"
    str_to_export += "\n"
    str_to_export += "\n"
    index += 1

f = open('exported_for_test12.h', 'w+')
f.write(str_to_export)
f.close()
