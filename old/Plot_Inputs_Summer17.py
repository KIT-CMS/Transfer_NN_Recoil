import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.mlab as mlab
import ROOT
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, MaxNLocator
import h5py
import sys
from matplotlib.lines import Line2D
from getNNinputs import kar2pol, angularrange

colors_InOut = cm.brg(np.linspace(0, 1, 8))
colors_In = ['#ed2024', '#4800B7', '#2C6AA8', '#00A878', '#DE3E1E', '#FF8360', '#1BE7FF' , '#312F2F', '#000072']
#colors_In = ['#1BE7FF', '#1BE7FF', '#5C6D69', '#00A878', '#DE3E1E', '#FF8360', '#1BE7FF' , '#312F2F', '#000072']
#6A8D73 #blau
#320A28 #aubergine
#4800B7 PF 
#DE3E1E
nbinsHistBin = 100
pTMin, pTMax = 20, 200
VertexMin, VertexMax = 23, 24

def pol2kar(norm, phi):
    x = np.cos(phi)*norm
    y = np.sin(phi)*norm
    return(x, y)

def kar2pol(x, y):
    rho = np.sqrt(np.multiply(x,x) + np.multiply(y,y))
    phi = np.arctan2(y, x)
    return(rho, phi)

def loadInputsTargets(file):
    InputsTargets = h5py.File("%s"%(file), "r")
    print("InputsTargets keys", InputsTargets.keys())
    '''
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi'],
                InputsTargets['NVertex']
                ))
    '''
    Target =  InputsTargets['Target']
    return (InputsTargets, Target)

def getHistogram(data, String, String2, pltmin, pltmax):
    Mean = np.mean(data)
    Std = np.std(data)
    if String2 == " new":
        bin = 2
    else:
        bin = 6
    #Reso = np.divide(-(DFName[branchString]),DFName[Target_Pt].values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName[Target_Pt].values, weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax]nbinsHistBin)
    plt.hist(data, weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label=String+String2+': %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', linewidth=3.0, ec=colors_InOut[bin], normed=True)



def pltHistograms(plotDir, NewData,OldData, String):
    
    nbinsHistBin= 50
    
    
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    pltmin = np.min([np.percentile(NewData, 2), np.percentile(OldData, 2)])
    pltmax = np.max([np.percentile(NewData, 98), np.percentile(OldData, 98)])
    getHistogram(NewData, String, " new", pltmin, pltmax)
    getHistogram(OldData, String, " old", pltmin, pltmax)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel(String, fontsize=22)
    plt.xlim(pltmin,pltmax)
    plt.ylabel('Density', fontsize=22)
    


    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.55), loc=4, borderaxespad=0., fontsize='large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%s%s.png"%(plotDir,String), bbox_inches="tight")
    plt.close()


def compareInputs(plotDir, filesDir):
    #FileOld = "%sNN_Input_apply_xy_old.h5"%(filesDir)
    FileNew = "%sNN_Input_apply_mm.h5"%(filesDir)
    #InputsOld, TargetsOld = loadInputsTargets(FileOld)
    InputsOld, TargetsOld = loadInputsTargets(FileNew)
    nbinsHistBin = 50
    weights = np.ones_like(InputsOld["Track"][0,:])/float(len(InputsOld["Track"][0,:]))
    
    
    print("Plots beginnen hier")
    #p_x
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    pltmin = -50
    pltmax = 50
    plt.hist(InputsOld["Track"][0,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='Track', histtype='step', linewidth=3.0, ec=colors_In[2], normed=True)
    plt.hist(InputsOld["NoPU"][0,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='No PU', histtype='step', linewidth=3.0, ec=colors_In[3], normed=True)
    plt.hist(InputsOld["PUCorrected"][0,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU corrected', histtype='step', linewidth=3.0, ec=colors_In[5], normed=True)
    plt.hist(InputsOld["PU"][0,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU', histtype='step', linewidth=3.0, ec=colors_In[6], normed=True)
    plt.hist(InputsOld["PF"][0,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PF', histtype='step', linewidth=3.0, ec=colors_InOut[1], normed=True)
    plt.hist(InputsOld["Puppi"][0,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PUPPI', histtype='step', linewidth=3.0, ec=colors_InOut[4], normed=True)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))


    plt.xlim(pltmin,pltmax)
    plt.ylabel('Density', fontsize=22)
    plt.xlabel('$p_x$ in GeV', fontsize=22)


    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.55), loc=4, borderaxespad=0., fontsize='large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%s%s.png"%(plotDir,'Px'), bbox_inches="tight")
    plt.close()    
    
    print("Px done")
    #p_y
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    pltmin = -50
    pltmax = 50
    plt.hist(InputsOld["Track"][1,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='Track', histtype='step', linewidth=3.0, ec=colors_In[2], normed=True)
    plt.hist(InputsOld["NoPU"][1,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='No PU', histtype='step', linewidth=3.0, ec=colors_In[3], normed=True)
    plt.hist(InputsOld["PUCorrected"][1,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU corrected', histtype='step', linewidth=3.0, ec=colors_In[5], normed=True)
    plt.hist(InputsOld["PU"][1,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU', histtype='step', linewidth=3.0, ec=colors_In[6], normed=True)
    plt.hist(InputsOld["PF"][1,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PF', histtype='step', linewidth=3.0, ec=colors_InOut[1], normed=True)
    plt.hist(InputsOld["Puppi"][1,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PUPPI', histtype='step', linewidth=3.0, ec=colors_InOut[4], normed=True)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlim(pltmin,pltmax)
    plt.ylabel('Density', fontsize=22)
    plt.xlabel('$p_y$ in GeV', fontsize=22)


    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.55), loc=4, borderaxespad=0., fontsize='large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%s%s.png"%(plotDir,'Py'), bbox_inches="tight")
    plt.close()
      
    print("Py done")

    #SumEt
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    pltmin = 0
    pltmax = 1500
    plt.hist(InputsOld["Track"][2,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='Track', histtype='step', linewidth=3.0, ec=colors_In[2], normed=True)
    plt.hist(InputsOld["NoPU"][2,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='No PU', histtype='step', linewidth=3.0, ec=colors_In[3], normed=True)
    plt.hist(InputsOld["PUCorrected"][2,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU corrected', histtype='step', linewidth=3.0, ec=colors_In[5], normed=True)
    plt.hist(InputsOld["PU"][2,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU', histtype='step', linewidth=3.0, ec=colors_In[6], normed=True)
    plt.hist(InputsOld["PF"][2,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PF', histtype='step', linewidth=3.0, ec=colors_InOut[1], normed=True)
    plt.hist(InputsOld["Puppi"][2,:], weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PUPPI', histtype='step', linewidth=3.0, ec=colors_InOut[4], normed=True)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlim(pltmin,pltmax)
    plt.ylabel('Density', fontsize=22)
    plt.xlabel('$\sum_i |p_{T,i}|$ in GeV', fontsize=22)


    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.55), loc=4, borderaxespad=0., fontsize='large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%s%s.png"%(plotDir,'SumEt'), bbox_inches="tight")
    plt.close()  

    print("Sum Et done")

    #nPV
    pltmin = 0
    pltmax = 75    
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')


    plt.hist(InputsOld["NVertex"], bins=np.arange(pltmin, pltmax+(pltmax-pltmin)/nbinsHistBin, (pltmax-pltmin)/50), range=[pltmin, pltmax], label='Number of vertices', histtype='step', linewidth=3.0, ec='black', normed=True)
 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlim(pltmin,pltmax)
    plt.ylabel('Density', fontsize=22)
    plt.xlabel('$N_{\mathrm{Vtx}}$', fontsize=22)


    #legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.90), loc=4, borderaxespad=0., fontsize='large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], c=h.get_edgecolor()) for h in handles],  labels=labels)
    #plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%s%s.png"%(plotDir,'NVertex'), bbox_inches="tight")
    plt.close()  
    
    print("NVertex done")


    #Delta Alpha
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    pltmin = -np.pi
    pltmax = np.pi
    Target_r, Target_phi = kar2pol(InputsOld["Target"][0,:],InputsOld["Target"][1,:])
    Track_r, Track_phi = kar2pol(InputsOld["Track"][0,:],InputsOld["Track"][1,:])
    NoPU_r, NoPU_phi = kar2pol(InputsOld["NoPU"][0,:],InputsOld["NoPU"][1,:])
    PUCorrected_r, PUCorrected_phi = kar2pol(InputsOld["PUCorrected"][0,:],InputsOld["PUCorrected"][1,:])
    PU_r, PU_phi = kar2pol(InputsOld["PU"][0,:],InputsOld["PU"][1,:])
    PF_r, PF_phi = kar2pol(InputsOld["PF"][0,:],InputsOld["PF"][1,:])
    Puppi_r, Puppi_phi = kar2pol(InputsOld["Puppi"][0,:],InputsOld["Puppi"][1,:])
    nbinsHistBin = 10
    colors = [colors_In[1], colors_In[3], colors_In[2], colors_In[6], colors_In[5], colors_InOut[4]]
    deltaPhi = [angularrange(PF_phi-Target_phi), angularrange(NoPU_phi-Target_phi), angularrange(Track_phi-Target_phi), angularrange(PU_phi-Target_phi), angularrange(PUCorrected_phi-Target_phi), angularrange(Puppi_phi-Target_phi)]
    vardeltaPhi = [np.var(angularrange(PF_phi-Target_phi)), np.var(angularrange(NoPU_phi-Target_phi)), np.var(angularrange(Track_phi-Target_phi)), np.var(angularrange(PU_phi-Target_phi)), np.var(angularrange(PUCorrected_phi-Target_phi)), np.var(angularrange(Puppi_phi-Target_phi))]
    print("Variance angles ", vardeltaPhi)
    labels= ['PF', 'PV', 'Charged PV', 'PU', 'PV+NU', 'PUPPI']
    print("lables ", labels)
    def getCos2MSin2(Phi):
        deltaphi= angularrange(Phi-Target_phi)
        return np.mean(np.transpose((np.square(np.cos(deltaphi)))))-np.mean(np.transpose((np.square(np.sin(deltaphi)))))
    print("cos^2-sin^2", getCos2MSin2(PF_phi), getCos2MSin2(NoPU_phi), getCos2MSin2(Track_phi), getCos2MSin2(PU_phi), getCos2MSin2(PUCorrected_phi), getCos2MSin2(Puppi_phi))
    weights2=[weights for i in range(0,6)]
    plt.hist(deltaPhi, weights=weights2, bins=nbinsHistBin, range=[pltmin,pltmax], label=labels, linewidth=3.0, color=colors, normed=True, histtype='bar')  
    
    #plt.hist(angularrange(Puppi_phi-Target_phi), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PUPPI', histtype='bar', linewidth=3.0, ec=colors_InOut[4], normed=True)
    #plt.hist(angularrange(NoPU_phi-Target_phi), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='No PU', histtype='bar', linewidth=3.0, ec=colors_In[3], normed=True)
    #plt.hist(angularrange(Track_phi-Target_phi), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='Track', histtype='bar', linewidth=3.0, ec=colors_In[2], normed=True)
    #plt.hist(angularrange(PU_phi-Target_phi), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU', histtype='bar', linewidth=3.0, ec=colors_In[6], normed=True)
    #plt.hist(angularrange(PUCorrected_phi-Target_phi), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU corrected', histtype='bar', linewidth=3.0, ec=colors_In[5], normed=True)
    #plt.hist(angularrange(PF_phi-Target_phi), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PF', histtype='bar', linewidth=3.0, ec=colors_InOut[1], normed=True)
    

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlim(pltmin,pltmax)
    #plt.yscale('log')
    plt.ylabel('Density', fontsize=22)
    plt.xlabel('$\\alpha^{\mathrm{recoil}}-(\\alpha^{\chi}+\pi)$ in rad', fontsize=22)


    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.55), loc=4, borderaxespad=0., fontsize='large', numpoints=1, framealpha=1.0	,  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%s%s.png"%(plotDir,'DeltaPhi'), bbox_inches="tight")
    plt.close()  
    
    print("DeltaPhi done")

    #Delta Alpha
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    nbinsHistBin = 70
    pltmin = 0
    pltmax = 3

    colors = [colors_InOut[4], colors_In[3], colors_In[2], colors_In[6], colors_In[5], colors_In[1]]
    relpT = [np.divide(Puppi_r, Target_r), np.divide(NoPU_r, Target_r), np.divide(Track_r, Target_r), np.divide(PU_r, Target_r), np.divide(PUCorrected_r, Target_r), np.divide(PF_r, Target_r)]
    labels= ['PF', 'PV', 'Charged PV', 'PU', 'PV+NU', 'PUPPI']
    #plt.hist(relpT, weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label=labels, linewidth=3.0, color=colors, normed=True, histtype='bar', fill=True)
    weights = np.ones_like(np.divide(PF_r, Target_r))/float(len(np.divide(PF_r, Target_r)))
    #plt.hist(myarray, weights=weights)
    
    plt.hist(np.divide(PF_r, Target_r), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PF', linewidth=3.0, ec=colors_InOut[1], normed=False, histtype='step', fill=False)
    plt.hist(np.divide(NoPU_r, Target_r), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PV', linewidth=3.0, ec=colors_In[3], normed=False, histtype='step', fill=False)
    plt.hist(np.divide(Track_r, Target_r), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='Charged PV', linewidth=3.0, ec=colors_In[2], normed=False, histtype='step', fill=False)
    plt.hist(np.divide(PU_r, Target_r), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PU', linewidth=3.0, ec=colors_In[6], normed=False, histtype='step', fill=False)
    plt.hist(np.divide(PUCorrected_r, Target_r), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PV+NU', linewidth=3.0, ec=colors_In[5], normed=False, histtype='step', fill=False)
    plt.hist(np.divide(Puppi_r, Target_r), weights=weights, bins=nbinsHistBin, range=[pltmin,pltmax], label='PUPPI', linewidth=3.0, ec=colors_InOut[4], normed=False, histtype='step', fill=False)
    
    print("var u", np.var(PF_r), np.var(NoPU_r), np.var(Track_r), np.var(PU_r), np.var(PUCorrected_r), np.var(Puppi_r))
    print("mean square u", np.mean(np.square(PF_r)), np.mean(np.square(NoPU_r)), np.mean(np.square(Track_r)), np.mean(np.square(PU_r)), np.mean(np.square(PUCorrected_r)), np.mean(np.square(Puppi_r)))
    print(" ptz", np.var(Target_r))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlim(pltmin,pltmax)
    plt.ylabel('Density', fontsize=22)
    plt.xlabel('$p_T^{\:\\mathrm{recoil}}/p_T^{\\chi}}$', fontsize=22)


    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.55), loc=4, borderaxespad=0., fontsize='large', numpoints=1, framealpha=1.0	,  handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles], labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%s%s.png"%(plotDir,'rel_pT'), bbox_inches="tight")
    plt.close()  
    
    print("DeltaPhi done")


    print("mean track rel pT", np.mean(np.divide(Track_r, Target_r)))

if __name__ == "__main__":
    plotsDir = sys.argv[1]
    filesDir = sys.argv[2]
    compareInputs(plotsDir,filesDir) 
