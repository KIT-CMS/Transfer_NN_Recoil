#!/usr/bin/env python
import h5py
import sys
import pandas as pd
import numpy as np
from os import environ
from getNNinputs import kar2pol, pol2kar, angularrange
from Training import loadInputsTargetsWeights
import matplotlib as mpl
import matplotlib.pyplot as plt



def prepareOutput(outputD, ll, plotsD):
    NN_Output = h5py.File("%sNN_Output_applied_%s.h5"%(outputD,ll), "r+")
    mZ_x, mZ_y = (NN_Output['MET_GroundTruth'][:,0]), (NN_Output['MET_GroundTruth'][:,1])
    a_x, a_y = (NN_Output['MET_Predictions'][:,0]), (NN_Output['MET_Predictions'][:,1])
    mZ_r, mZ_phi =  kar2pol(mZ_x, mZ_y)
    mZ_r = NN_Output['Boson_Pt'][:]
    a_r, a_phi = kar2pol(a_x, a_y)


    NN_LongZ, NN_PerpZ = -np.cos(angularrange(np.add(a_phi,-mZ_phi)))*a_, np.sin(angularrange(a_phi-mZ_phi))*a_


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


if __name__ == "__main__":
    outputDir = sys.argv[1]
    ll = sys.argv[2]
    plotsD = sys.argv[3]
    NN_MVA = h5py.File("%s/NN_MVA_%s.h5"%(outputDir,ll), "w")

    prepareOutput(outputDir, ll, plotsD)
