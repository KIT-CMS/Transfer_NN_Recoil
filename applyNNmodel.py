#!/usr/bin/env python
import h5py
import matplotlib as mpl
mpl.use('Agg')
import sys
import numpy as np
from os import environ
import tensorflow as tf
from numpy import s_
from Training import NNmodel
from sklearn.preprocessing import StandardScaler
import pickle


def loadInputsTargets(outputD, ll):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5"%(outputDir,ll), "r")
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi'],
                InputsTargets['NVertex']
                ))

    Target =  InputsTargets['Target']
    return (np.transpose(Input), np.transpose(Target))


def loadBosonPt(inputD):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (inputD, ll), "r")
    Target =  np.squeeze(InputsTargets['Boson_Pt'][:])
    return (np.transpose(Target))



def applyModel(outputD, ll, training, lossFct):
    Inputs, Targets = loadInputsTargets(outputD, ll)
    
    if training == "false":
        if lossFct == "mean_squared_error":
            modelpath = "BestModel_mse/"
            print("Use model storen in folder BestModel_mse")
        else:    
            modelpath = "BestModel/"
    elif training == "true":    
        modelpath =  outputD    
    else:
        print("There was no training switch provided. Please provide training switch (True if you want to train, false if you want to apply) in shell script")    
        sys.exit()
    
    #get prediction
    x = tf.placeholder(tf.float32)
    logits, f = NNmodel(x, reuse=False)
    checkpoint_path = tf.train.latest_checkpoint(modelpath)
    
    #Restore
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
        
    if training == "true":
        #Get Cross-Validation
        x_CV = tf.placeholder(tf.float32)
        logits_CV, f_CV = NNmodel(x_CV, reuse=True)
        checkpoint_path_CV = tf.train.latest_checkpoint(outputD+"CV/")
        sess_CV = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.Saver()
        saver.restore(sess_CV, checkpoint_path_CV)
        #Preprocessing
        preprocessing_input = pickle.load(open("preprocessing_input.pickle", "rb"))
        predictions_CV = []   
    else:
        #Preprocessing
        preprocessing_input = pickle.load(open("./BestModel/preprocessing_input.pickle", "rb"))
        predictions_CV = []   


    applicationsplit = int(len(Inputs[:,0])/10)
    predictions = []
    

    for i in range(0,10):
        start, end = i*applicationsplit, (i+1)*applicationsplit+1
        print("shape Inputs", Inputs.shape)
        predictions_i = sess.run(f, {x: preprocessing_input.transform(np.array_split(Inputs, 10, axis=0)[i])})
        if training == "true":
            predictions_CV_i = sess_CV.run(f_CV, {x_CV: preprocessing_input.transform(np.array_split(Inputs, 10, axis=0)[i])})
        print("shape np.array_split(Inputs, 10, axis=0)[i]", np.array_split(Inputs, 10, axis=0)[i].shape)
        if i==0:
            predictions = predictions_i
            if training == "true":
                predictions_CV = predictions_CV_i
        else:
            predictions = np.append(predictions, predictions_i, axis=0)
            if training == "true":
                predictions_CV = np.append(predictions_CV, predictions_CV_i, axis=0)

    
    
    if training == "true":      
        #merge both predictions
        Test_Idx_CV2 = h5py.File("%sTest_Idx_CV_%s.h5" % (outputDir, ll), "r")
        Test_Idx_CV = Test_Idx_CV2["Test_Idx"].value.astype(int)
        predictions[Test_Idx_CV,:] = predictions_CV[Test_Idx_CV,:]
    
    #Write output
    dset = NN_Output_applied.create_dataset("MET_Predictions", dtype='f', data=predictions)
    dset2 = NN_Output_applied.create_dataset("MET_GroundTruth", dtype='f', data=Targets)
    dset3 = NN_Output_applied.create_dataset('Boson_Pt', dtype='f', data=loadBosonPt(outputD)[:])
    NN_Output_applied.close()

if __name__ == "__main__":
    ll = sys.argv[1]
    outputDir = sys.argv[2]
    Training = sys.argv[3]
    lossfct = sys.argv[4]
    NN_Output_applied = h5py.File("%sNN_Output_applied_%s.h5"%(outputDir,ll), "w")
    applyModel(outputDir, ll, Training, lossfct)
