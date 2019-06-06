import numpy as np
import os
import pickle
import json
import tensorflow as tf
from Training import NNmodel

from keras.models import Sequential
from keras.layers import Dense


def main(args):
    # Load weights and save them as .h5
    x = tf.placeholder(tf.float32)
    logits, f = NNmodel(x, reuse=False)
    modelpath = "BestModel/"
    checkpoint_path = tf.train.latest_checkpoint(modelpath)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    weights = {}
    for i in range(1, 6):
        weights["w%u"%i] = tf.get_default_graph().get_tensor_by_name("model/w%u:0"%i).eval(session=sess)
        weights["b%u"%i] = tf.get_default_graph().get_tensor_by_name("model/b%u:0"%i).eval(session=sess)
    #model.load_weights("./BestModel/checkpoint") 
    #model.save_weights("NNrecoil_weights.h5")
    tf.reset_default_graph()

    # Define model structure as in Training.py
    input_dim = 19
    output_dim = 2
    n_nodes = 128
    bias_initializer = "Zeros"
    kernel_initializer = "glorot_uniform"
    model = Sequential()
    model.add(Dense(n_nodes, activation='relu', input_shape=(input_dim,)))
    for i in range(3):
        model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.summary()

    # Load tensorflow weights into keras model
    for i in range(5):
        model.layers[i].set_weights([weights["w%u"%(i+1)], weights["b%u"%(i+1)]])

    # Write model to .json
    with open("NNrecoil_model.json", "w") as f:
        f.write(model.to_json())
    f.close()

    # Use preprocessing & input names to write a .json for variables
    variables_list = [
        'metpx','metpy','metsumet',
        'trackmetpx','trackmetpy','trackmetsumet',
        'nopumetpx','nopumetpy','nopumetsumet',
        'pucormetpx','pucormetpy','pucormetsumet',
        'pumetpx','pumetpy','pumetsumet',
        'puppimetpx','puppimetpy','puppimetsumet',
        'npv']
    variables_dict = {
        "class_labels" : ["NN_px", "NN_py"],
        "inputs" : [{"name" : v, "offset" : 0.0, "scale" : 1.0} for v in variables_list]
    } 
    scaler = pickle.load(open("./BestModel/preprocessing_input.pickle", "rb"))
    for variable, offset, scale in zip(variables_dict["inputs"], scaler.mean_, scaler.scale_):
        variable["offset"] = -offset
        variable["scale"] = 1.0/scale
        with open("NNrecoil_variables.json", "w") as f:
            f.write(json.dumps(variables_dict))
        f.close()

if __name__ == "__main__":
    args = None
    main(args)
