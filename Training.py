import numpy as np
np.random.seed(1234)
import tensorflow as tf
from tensorflow_derivative.inputs import Inputs as InputsDer
from tensorflow_derivative.outputs import Outputs as OutputsDer
from tensorflow_derivative.derivatives import Derivatives
import datetime
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import pandas as pd
from pandas import Series, MultiIndex, DataFrame
import seaborn as sns
from matplotlib.lines import Line2D
import pickle
from matplotlib import ticker

formatter = ticker.ScalarFormatter(useMathText=True) 
formatter.set_scientific(True) 
formatter.set_powerlimits((-2,2)) 


reweighting = True


def loadAxesSettingsLoss(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')    
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)
    ax.yaxis.offsetText.set_fontsize(18) 
    plt.grid(color='grey', linestyle='-', linewidth=0.4)


def loadInputsTargetsWeights(outputD, ll):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (outputD,ll), "r")
    norm = InputsTargets['Boson_Pt']
    Target =  InputsTargets['Target']
    weight =  InputsTargets['weights']
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi'],
                InputsTargets['NVertex']
                ))
    return (np.transpose(Input), np.transpose(Target), np.transpose(weight))

def loadInputsTargetsPVWeights(outputD, ll):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (outputD,ll), "r")
    norm = InputsTargets['Boson_Pt']
    Target =  InputsTargets['Target']
    weight =  InputsTargets['weights']
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi'],
                InputsTargets['NVertex']
                ))
    return (np.transpose(Input), np.transpose(Target), np.transpose(weight), np.transpose(InputsTargets['NVertex']))

def moving_average(data_set, periods):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def costExpectedRelAsypTRange(y_true,y_pred, weight, Ranges):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_long = tf.cos(alpha_diff)*a_

    cost = 0
    for i in range(0,len(Ranges)-1):
        mask = [tf.logical_and((pZ>Ranges[i]) , (pZ<=Ranges[i+1]))]
        Response1 = tf.divide(tf.boolean_mask(u_long,tf.reshape(mask, [-1])), tf.boolean_mask(pZ,tf.reshape(mask,[-1])))
        #print("tf shape Response1", tf.shape(Response1))
        Response_Diff1 = tf.square(tf.reduce_sum(tf.nn.relu(Response1-1))-tf.reduce_sum(tf.nn.relu(1-Response1)))
        cost1 = Response_Diff1*0.03
        cost = cost + cost1

    Response = tf.divide(u_long, pZ)
    return cost+tf.sqrt(tf.reduce_sum(tf.square(Response-1)))


def costExpectedRelAsypTPVRange(y_true,y_pred, weight, pTRanges, PVRanges):
    PV = weight
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_long = tf.cos(alpha_diff)*a_

    cost = 0
    for i in range(0,len(pTRanges)-1):
        for j in range(0,len(PVRanges)-1):
            maskpT = tf.logical_and((pZ>pTRanges[i]) , (pZ<=pTRanges[i+1]))
            maskPV = tf.logical_and((PV>PVRanges[j]) , (PV<=PVRanges[j+1]))
            mask = [tf.logical_and(tf.reshape(maskpT, [-1]),tf.reshape(maskPV, [-1]))]

            Response1 = tf.divide(tf.boolean_mask(u_long,tf.reshape(mask, [-1])), tf.boolean_mask(pZ,tf.reshape(mask,[-1])))
            Response_Diff1 = tf.square(tf.reduce_sum(tf.nn.relu(Response1-1))-tf.reduce_sum(tf.nn.relu(1-Response1)))
            cost1 = Response_Diff1*0.03
            cost = cost + cost1

    Response = tf.divide(u_long, pZ)
    return cost+tf.sqrt(tf.reduce_sum(tf.square(Response-1)))


def getpTRanges(pT):
    pTRanges = np.linspace(np.floor(np.min(pT)),np.max(pT),10)
    return pTRanges

def getPVRanges(PV):
    PVbins = 5
    pPV = 100/PVbins
    PVRanges = [0]
    PVRanges = np.append(PVRanges, [np.percentile(PV, i*pPV) for i in range(1,PVbins+1)])
    return PVRanges

def NNmodel(x, reuse):
    ndim = 128
    with tf.variable_scope("model") as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=(19,ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b1 = tf.get_variable('b1', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable('w2', shape=(ndim, ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b2 = tf.get_variable('b2', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

        w3 = tf.get_variable('w3', shape=(ndim, ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b3 = tf.get_variable('b3', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        w4 = tf.get_variable('w4', shape=(ndim, ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b4 = tf.get_variable('b4', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

        w5 = tf.get_variable('w5', shape=(ndim, 2), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b5 = tf.get_variable('b5', shape=(2), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))


    l1 = tf.nn.relu(tf.add(b1, tf.matmul(x, w1)))
    l2 = tf.nn.relu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.relu(tf.add(b3, tf.matmul(l2, w3)))
    l4 = tf.nn.relu(tf.add(b4, tf.matmul(l3, w4)))
    logits = tf.add(b5, tf.matmul(l4, w5), name='output')
    return logits, logits


def getNNModel(outputDir, loss_fct, ll, plotsD):
    if loss_fct == 'relResponseAsypTPVRange':
        Inputs, Targets, Weights, PV = loadInputsTargetsPVWeights(outputDir, ll)
    else:
        Inputs, Targets, Weights = loadInputsTargetsWeights(outputDir, ll)
    Boson_Pt = np.sqrt(np.square(Targets[:,0])+np.square(Targets[:,1]))

    num_events = Inputs.shape[0]
    train_test_splitter = 0.5
    training_idx = np.random.choice(np.arange(Inputs.shape[0]), int(Inputs.shape[0]*train_test_splitter), replace=False)

    #Write Test Idxs
    test_idx = np.setdiff1d(  np.arange(Inputs.shape[0]), training_idx)
    dset = Test_Idx.create_dataset("Test_Idx",  dtype='f', data=test_idx)

    #Get Datasets for training and validation
    Inputs_train, Inputs_test = Inputs[training_idx,:], Inputs[test_idx,:]
    Targets_train, Targets_test = Targets[training_idx,:], Targets[test_idx,:]

    train_val_splitter = 0.9
    train_train_idx_idx = np.random.choice(np.arange(training_idx.shape[0]), int(training_idx.shape[0]*train_val_splitter), replace=False)
    train_train_idx = training_idx[train_train_idx_idx]
    train_val_idx = training_idx[ np.setdiff1d(  np.arange(training_idx.shape[0]), train_train_idx_idx)]
    Inputs_train_train, Inputs_train_val = Inputs[train_train_idx,:], Inputs[train_val_idx,:]
    Targets_train, Targets_test = Targets[train_train_idx,:], Targets[train_val_idx,:]
    if reweighting and not (loss_fct == 'relResponseAsypTPVRange'):
        weights_train_, weights_val_ = Weights[train_train_idx,:], Weights[train_val_idx,:]
    elif reweighting and loss_fct == 'relResponseAsypTPVRange':
        prob_train_, prob_val_ = Weights[train_train_idx,:], Weights[train_val_idx,:]
        weights_train_, weights_val_ = PV[train_train_idx,:], PV[train_val_idx,:]
    else:
        print("No reweighting")
        weights_train_, weights_val_ = np.repeat(1., len(train_train_idx)) , np.repeat(1., len(train_val_idx))
        weights_train_.shape = (len(train_train_idx),1)
        weights_val_.shape = (len(train_val_idx),1)
        
    #Assignment of datasets used in the training
    data_train = Inputs_train_train
    labels_train = Targets_train
    data_val = Inputs_train_val
    labels_val = Targets_test
    weights_train = weights_train_
    weights_val = weights_val_
    
    #Batchsizes
    batchsize = 4500
    batchsize_val = 10000

    #Get stings of inputs
    MET_definitions = ['PF', 'Track', 'NoPU', 'PUCorrected', 'PU', 'Puppi']
    Variables = ['x','y','SumEt']
    Variables = [Variables[:] for _ in range(len(MET_definitions))]
    MET_definitions = np.repeat(MET_definitions,3)
    Variables = [item for sublist in Variables for item in sublist]
    Inputstring = [MET+'_'+Variable for MET,Variable in zip(MET_definitions,Variables)]
    Inputstring = np.append(Inputstring, 'NVertex')

    #Placeholders
    xDer = InputsDer(Inputstring)
    x = xDer.placeholders
    y = tf.placeholder(tf.float32, shape=[batchsize, labels_train.shape[1]])
    w = tf.placeholder(tf.float32, shape=[batchsize, weights_train.shape[1]])
    x_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    w_ = tf.placeholder(tf.float32)

    #GPU configs
    print("tf.test.gpu_device_name()", tf.test.gpu_device_name())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #Define the neural network architecture
    batch_train = [data_train, labels_train, weights_train]
    batch_val = [data_val, labels_val, weights_val]
    logits_train, f_train= NNmodel(x, reuse=False)
    yDer = OutputsDer(logits_train, ['x','y'])
    logits_val, f_val= NNmodel(x_, reuse=True)
    
    #Initialize derivates
    derivatives = Derivatives(xDer, yDer)
    d={}
    for i in range(0,len(Inputstring)):
            Variable = sorted(Inputstring)[i]
            d["1dxd"+Variable]=derivatives.get('x', [Variable])
            d["1dyd"+Variable]=derivatives.get('y', [Variable])
    list_derivatestensor = d.values()


    print("loss fct", loss_fct)
    if (loss_fct=="mean_squared_error"):
        loss_train = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=y, predictions=logits_train))
        loss_val = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=y_, predictions=logits_val))
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="relResponseAsypTRange"):
        pTRanges = []
        pTRanges = getpTRanges(Boson_Pt)
        loss_train = costExpectedRelAsypTRange(y, logits_train, w, pTRanges)
        loss_val = costExpectedRelAsypTRange(y_, logits_val, w_, pTRanges)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="relResponseAsypTPVRange"):
        pTRanges, PVRanges = [], []
        pTRanges = getpTRanges(Boson_Pt)
        PVRanges = getPVRanges(PV)
        loss_train = costExpectedRelAsypTPVRange(y, logits_train, w, pTRanges, PVRanges)
        loss_val = costExpectedRelAsypTPVRange(y_, logits_val, w_, pTRanges, PVRanges)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    else:
        sys.exit("Error, no suitable loss declared")



    # ## Run the training
    sess.run(tf.global_variables_initializer())

    losses_train = []
    losses_val = []
    
    #Initialize summary
    summary_train = tf.summary.scalar("loss_train", loss_train)
    summary_val = tf.summary.scalar("loss_val", loss_val)
    writer = tf.summary.FileWriter("./logs/{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), sess.graph)
    saver = tf.train.Saver()
    
    #Initialize Training steps
    min_valloss = [1000000000000]
    max_steps = 300000
    saveStep = 100
    early_stopping = 0
    best_model = 0
    
    if loss_fct == 'relResponseAsypTPVRange':
        batch_prob = prob_train_.flatten() * 1 / (np.sum(prob_train_.flatten()))
        batch_prob_val = prob_val_.flatten() * 1 / (np.sum( prob_val_.flatten()))
    else:
        batch_prob = weights_train.flatten() * 1 / (np.sum(weights_train.flatten()))
        batch_prob_val = weights_val.flatten() * 1 / (np.sum( weights_val.flatten()))
    pT = np.sqrt(np.square(labels_train[:,0]) + np.square(labels_train[:,1]))

    #Preprocessing
    preprocessing_input = StandardScaler()
    preprocessing_output = StandardScaler()
    preprocessing_i = preprocessing_input.fit(Inputs)
    preprocessing_o = preprocessing_output.fit(Targets)

    #Save preprocessing in pickle
    pickle.dump(preprocessing_i, open("preprocessing_input.pickle", "wb"))
    pickle.dump(preprocessing_o, open("preprocessing_output.pickle", "wb"))    


    for i_step in range(max_steps):

        batch_train_idx = np.random.choice(np.arange(data_train.shape[0]), batchsize, p=batch_prob, replace=False)
        batch_val_idx = np.random.choice(np.arange(data_val.shape[0]), batchsize, p=batch_prob_val, replace=False)
        summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss], feed_dict={x: preprocessing_input.transform(data_train[batch_train_idx,:]), y: labels_train[batch_train_idx,:], w: weights_train[batch_train_idx,:]})
        losses_train.append(loss_)
        writer.add_summary(summary_, i_step)
        
        summary_, loss_ = sess.run([summary_val, loss_val], feed_dict={x_: preprocessing_input.transform(data_val[batch_val_idx,:]), y_: labels_val[batch_val_idx,:], w_: weights_val[batch_val_idx,:]})
        losses_val.append(loss_)
        writer.add_summary(summary_, i_step)
   
        if i_step % saveStep == 0:
            batch_val_idx_100 =  np.random.choice(np.arange(data_val.shape[0]), batchsize_val, p=batch_prob_val, replace=False)
            loss_ = sess.run(loss_val, feed_dict={x_: preprocessing_input.transform(data_val[batch_val_idx_100,:]), y_: labels_val[batch_val_idx_100,:], w_: weights_val[batch_val_idx_100,:]})
            if loss_<min(min_valloss):
                best_model = i_step
                saver.save(sess, "%sNNmodel"%outputDir, global_step=i_step)
                outputs = ["output"]
                constant_graph = tf.graph_util.convert_variables_to_constants(
                    sess, sess.graph.as_graph_def(), outputs)
                tf.train.write_graph(constant_graph, outputDir, "constantgraph.pb", as_text=False)
                early_stopping = 0
                print("better val loss found at ", i_step)
                
                list_derivatestensor = d.values()
            else:
                early_stopping += 1
                print("increased early stopping to ", early_stopping)
            if early_stopping == 120:
                break
            min_valloss.append(loss_)
            print('gradient step No ', i_step)
            print("validation loss", loss_)


    #Plot loss functions
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    plt.plot(range(1, len(moving_average(np.asarray(losses_train[0:(best_model+800)]), 800))+1), moving_average(np.asarray(losses_train[0:(best_model+800)]), 800), lw=3, label="Training batches", color='#ED2024')
    plt.plot(range(1, len(moving_average(np.asarray(losses_val[0:(best_model+800)]), 800))+1), moving_average(np.asarray(losses_val[0:(best_model+800)]), 800), lw=3, label="Validation batches", color='#2C6AA8')
    plt.xlabel("Training step", fontsize=22), plt.ylabel("$\\langle \mathscr{L} \\rangle_{800}$", fontsize=22)
    plt.yscale('log')
    plt.xscale('log')
    legend = plt.legend()
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    ax.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    loadAxesSettingsLoss(ax)
    plt.savefig("%sLoss_ValLoss.png"%(plotsD), bbox_inches="tight")
    plt.close()

    dset = NN_Output.create_dataset("loss", dtype='f', data=losses_train)
    dset2 = NN_Output.create_dataset("val_loss", dtype='f', data=losses_val)
    NN_Output.close()


if __name__ == "__main__":
    outputDir = sys.argv[1]
    loss_fct = str(sys.argv[2])
    ll = sys.argv[3]
    plotsD = sys.argv[4]
    print(outputDir)
    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputDir,ll), "w")
    Test_Idx = h5py.File("%sTest_Idx_%s.h5" % (outputDir, ll), "w")
    getNNModel(outputDir, loss_fct, ll, plotsD)
