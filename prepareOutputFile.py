#!/usr/bin/env python
import h5py
import sys
import root_numpy as rnp
import pandas as pd
import numpy as np
from os import environ
from getNNinputs import kar2pol, pol2kar, angularrange
from Training import loadInputsTargetsWeights
import matplotlib as mpl
import matplotlib.pyplot as plt



def prepareOutput(outputD, ll, plotsD, rootFile):
    NN_Output = h5py.File("%sNN_Output_applied_%s.h5"%(outputD,ll), "r+")
    mZ_x, mZ_y = (NN_Output['MET_GroundTruth'][:,0]), (NN_Output['MET_GroundTruth'][:,1])
    a_x, a_y = (NN_Output['MET_Predictions'][:,0]), (NN_Output['MET_Predictions'][:,1])
    mZ_r, mZ_phi =  kar2pol(mZ_x, mZ_y)
    mZ_r = NN_Output['Boson_Pt'][:]
    a_r, a_phi = kar2pol(a_x, a_y)


    NN_LongZ, NN_PerpZ = -np.cos(angularrange(np.add(a_phi,-mZ_phi)))*a_r, np.sin(angularrange(a_phi-mZ_phi))*a_r

    #HDF5
    dset = NN_MVA.create_dataset("NN_LongZ", dtype='d', data=NN_LongZ)
    dset1 = NN_MVA.create_dataset("NN_PerpZ", dtype='d', data=NN_PerpZ)
    dset2 = NN_MVA.create_dataset("NN_Phi", dtype='d', data=a_phi)
    dset3 = NN_MVA.create_dataset("NN_Pt", dtype='d', data=a_r)
    dset4 = NN_MVA.create_dataset("Boson_Pt", dtype='d', data=mZ_r)
    dset5 = NN_MVA.create_dataset("NN_x", dtype='d', data=a_x)
    dset6 = NN_MVA.create_dataset("NN_y", dtype='d', data=a_y)
    dset7 = NN_MVA.create_dataset("Boson_x", dtype='d', data=mZ_x)
    dset8 = NN_MVA.create_dataset("Boson_y", dtype='d', data=mZ_y)
    dset9 = NN_MVA.create_dataset("Boson_Phi", dtype='d', data=mZ_phi)
    NN_MVA.close()

    #Root
    treename = ll+"_nominal/ntuple"
    Root_array = rnp.root2array(rootFile, treename=treename)
    print("shape Root_array", Root_array.shape)
    NN_array = np.array([NN_LongZ, NN_PerpZ, a_phi, a_r, a_x, a_y],
              dtype=[('NN_LongZ', np.float32),
                     ('NN_PerpZ', np.float32),
                     ('NN_Phi', np.float32),
                     ('NN_Pt', np.float32),
                     ('NN_x', np.float32),
                     ('NN_y', np.float32)]) 
    print("shape NN_array", NN_array.shape)   
    #conc_array = np.concatenate((Root_array, NN_array))
    #rnp.array2root(conc_array, "%s/NN_MVA_%s.root"%(outputDir,ll), mode='recreate')


if __name__ == "__main__":
    outputDir = sys.argv[1]
    ll = sys.argv[2]
    plotsD = sys.argv[3]
    rootFile = sys.argv[4]
    NN_MVA = h5py.File("%s/NN_MVA_%s.h5"%(outputDir,ll), "w")

    prepareOutput(outputDir, ll, plotsD, rootFile)
