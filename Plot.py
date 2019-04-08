
import numpy as np
from getNNinputs import kar2pol, pol2kar, angularrange
#import root_numpy as rnp
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import optimize
import matplotlib.mlab as mlab
import ROOT
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats
import matplotlib.ticker as mtick
from scipy import optimize
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, MaxNLocator
from getNNinputs import loadRoot
import h5py
import sys
from matplotlib.lines import Line2D
from scipy import stats
#import seaborn as sns
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True) 
formatter.set_scientific(True) 
formatter.set_powerlimits((-2,2)) 

pTMin, pTMax = 20, 200
VertexMin, VertexMax = 2, 50

mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.color'] = 'black'
mpl.rcParams["axes.grid"] = True
mpl.rcParams['axes.linewidth'] = 0.5



nbins = int((pTMax-pTMin)/20)
binsAngle = 7
nbinsVertex = 5
nbinsHist = 40
nbinsHistBin = 40
nbins_relR = 10
colors = cm.brg(np.linspace(0, 1, 8))

colors_InOut = cm.brg(np.linspace(0, 1, 8))
colors2 = colors
HistLimMin, HistLimMax = -50, 50
ResponseMin, ResponseMax = -1,3
MassMin, MassMax = 0, 160
DeltaMassMin, DeltaMassMax = -100, 100
MassW = 80.385
ResolutionParaMin, ResolutionParaMax = -60, 60
ResolutionPerpMin, ResolutionPerpMax = -60, 60
ResponseMinErr, ResponseMaxErr = 0, 1.05
ylimResMVAMin, ylimResMVAMax = 0, 45
errbars_shift2 = 10

pTTresh = 10

#Data settings

def fct(x, a, b):
    return a*np.sqrt(x)+b
    
def getab(x_data, y_data):
    params, params_covariance = optimize.curve_fit(fct, x_data, y_data)
    return params




def getCI(data):
    meanp = stats.percentileofscore(data, np.mean(data))
    start = np.percentile(data,meanp-34)
    mean = np.mean(data)
    end = np.percentile(data,meanp+34)
    return [mean-start, end-mean]

def getCIStartEnd(data):
    meanp = stats.percentileofscore(data, np.mean(data))
    start = np.percentile(data,meanp-34)
    mean = np.mean(data)
    end = np.percentile(data,meanp+34)
    return [start, end]


def getCImed(data):
    start = np.percentile(data,50-34)
    mean = np.percentile(data,50)
    end = np.percentile(data,50+34)
    return [mean-start, end-mean]

def getLong(a_x, a_y, true_x, true_y):
    a_, a_phi = kar2pol(a_x, a_y)
    mZ_r, mZ_phi = kar2pol(true_x, true_y)
    NN_LongZ, NN_PerpZ = -np.cos(angularrange(np.add(a_phi,-mZ_phi)))*a_, np.sin(angularrange(a_phi-mZ_phi))*a_
    return NN_LongZ
def getPerp(a_x, a_y, true_x, true_y):
    a_, a_phi = kar2pol(a_x, a_y)
    mZ_r, mZ_phi = kar2pol(true_x, true_y)
    NN_LongZ, NN_PerpZ = -np.cos(angularrange(np.add(a_phi,-mZ_phi)))*a_, np.sin(angularrange(a_phi-mZ_phi))*a_
    return NN_PerpZ



def getNeutrino(branchString):
    x, y = pol2kar(DFName[branchString+"_Pt"], DFName[branchString+"_Phi"])
    pXmiss = -(x+DFName["Px_1"])
    pYmiss = -(y+DFName["Py_1"])
    pTmiss, phimiss = kar2pol(pXmiss, pYmiss)
    pTl, phil = kar2pol(DFName["Px_1"], DFName["Py_1"]) 
    Boson_x, Boson_y = pol2kar(DFName["Boson_Pt"],DFName["Boson_Phi"])
    Neutrino_x, Neutrino_y = Boson_x-DFName["Px_1"], Boson_y-DFName["Py_1"]
    deltPhi = phil - phimiss
    mT = np.sqrt(2*pTl*pTmiss*(1-np.cos(deltPhi)))
    return np.sqrt(np.square(pXmiss+Neutrino_x)+np.square(pYmiss+Neutrino_y))

def getTransverseMass(branchString):
    x, y = pol2kar(DFName[branchString+"_Pt"], DFName[branchString+"_Phi"])
    pXmiss = -(x+DFName["Px_1"])
    pYmiss = -(y+DFName["Py_1"])
    pTmiss, phimiss = kar2pol(pXmiss, pYmiss)
    pTl, phil = kar2pol(DFName["Px_1"], DFName["Py_1"])
    deltPhi = phil - phimiss
    mT = np.sqrt(2*pTl*pTmiss*(1-np.cos(deltPhi)))
    return mT

def getTransverseMassZ(branchString):
    x, y = pol2kar(DFNameZ[branchString+"_Pt"], DFNameZ[branchString+"_Phi"])
    l1_Pt, l1_phi = kar2pol(DFNameZ["Px_1"], DFNameZ["Py_1"])
    l2_Pt, l2_phi = kar2pol(DFNameZ["Px_2"], DFNameZ["Py_2"])
    l1 = np.greater(l1_Pt,l2_Pt)
    print("l1",  l1)
    l2 = np.invert(l1)
    print("l2",  l2)
    pXmiss = -(x+DFNameZ["Px_1"][l1]+DFNameZ["Px_2"][l2])
    pYmiss = -(y+DFNameZ["Py_1"][l1]+DFNameZ["Py_2"][l2])
    pTmiss, phimiss = kar2pol(pXmiss, pYmiss)
    #pTl, phil = kar2pol(DFName["Px_1"], DFName["Py_1"])
    pTl = np.add(np.multiply(l1_Pt, l1), np.multiply(l2_Pt, l2))
    phil = np.add(np.multiply(l1_phi, l1), np.multiply(l2_phi, l2))
    deltPhi = phil - phimiss
    mT = np.sqrt(2*pTl*pTmiss*(1-np.cos(deltPhi)))
    return mT    

def loadAxesSettings(ax):
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
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18) 
    plt.grid(color='grey', linestyle='-', linewidth=0.4)

def getpTMissZ(branchString):
    pTMin, pTMax = 50, 200
    l1_Pt, l1_phi = kar2pol(DFNameZ["Px_1"], DFNameZ["Py_1"])
    l2_Pt, l2_phi = kar2pol(DFNameZ["Px_2"], DFNameZ["Py_2"])
    maxl1l2 = np.maximum(l1_Pt,l2_Pt)
    IndpTCut= (DFNameZ["Boson_Pt"]>pTMin) & (DFNameZ["Boson_Pt"]<=pTMax)
    #IndpTCut= (maxl1l2>pTMin) 
    x, y = pol2kar(DFNameZ[branchString+"_Pt"], DFNameZ[branchString+"_Phi"])
    pXmiss = -(x[IndpTCut]+DFNameZ["Px_1"][IndpTCut]+DFNameZ["Px_2"][IndpTCut])
    pYmiss = -(y[IndpTCut]+DFNameZ["Py_1"][IndpTCut]+DFNameZ["Px_2"][IndpTCut])
    pTmiss, phimiss = kar2pol(pXmiss, pYmiss)
    return pTmiss

def getpTMiss(branchString):
    pTMin, pTMax = 50, 200
    l1_Pt, l1_phi = kar2pol(DFName["Px_1"], DFName["Py_1"])
    #IndpTCut= (l1_Pt>pTMin) 
    IndpTCut= (DFName["Boson_Pt"]>pTMin) & (DFName["Boson_Pt"]<=pTMax)
    #prob = DFName["weights"]
    #IndpTCut = np.random.choice(np.arange(DFName["Boson_Pt"].shape[0]), 10000, p=prob, replace=False)
    x, y = pol2kar(DFName[branchString+"_Pt"], DFName[branchString+"_Phi"])
    pXmiss = -(x[IndpTCut]+DFName["Px_1"][IndpTCut])
    pYmiss = -(y[IndpTCut]+DFName["Py_1"][IndpTCut])
    pTmiss, phimiss = kar2pol(pXmiss, pYmiss)
    return pTmiss

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

def loadData(filesDir, rootInput,ll, Wswitch):
    NN_MVA = h5py.File("%sNN_MVA_%s.h5"%(filesDir,ll), "r+")
    DFNameInput_h5 = h5py.File("%sNN_Input_apply_%s.h5"%(filesDir,ll), "r+")
    DFNameInput = loadRoot(rootInput,ll,filesDir, Wswitch)
    DFName = pd.DataFrame(index=np.arange(len(NN_MVA['NN_LongZ'])))
    DFnames = ['NN_LongZ', 'NN_PerpZ', 'NN_x', 'NN_y', 'NN_Pt', 'NN_Phi', 'Boson_Pt', 'Boson_Phi', 'Boson_x', 'Boson_y']
    for col in DFnames:
        DFName[col] = NN_MVA[col]
    if Wswitch=='true':
        DFName['genBosonTransMass'] = DFNameInput_h5['genBosonTransMass'][0,:]   
    else: 
        DFName['Px_2'] = DFNameInput_h5['Lepton2'][0,:]    
        DFName['Py_2'] = DFNameInput_h5['Lepton2'][1,:] 
    
    DFName['weights'] = DFNameInput_h5['weights'][0,:]          
    DFName['Px_1'] = DFNameInput_h5['Lepton1'][0,:]    
    DFName['Py_1'] = DFNameInput_h5['Lepton1'][1,:]    
    DFName['NVertex'] = DFNameInput_h5['NVertex'][0,:]
    DFName['PF_LongZ'] = getLong(DFNameInput_h5['PF'][0,:], DFNameInput_h5['PF'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['PF_PerpZ'] = getPerp(DFNameInput_h5['PF'][0,:], DFNameInput_h5['PF'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['PF_Pt'], DFName['PF_Phi'] = kar2pol(DFNameInput_h5['PF'][0,:], DFNameInput_h5['PF'][1,:])
    DFName['Puppi_LongZ'] = getLong(DFNameInput_h5['Puppi'][0,:], DFNameInput_h5['Puppi'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['Puppi_PerpZ'] = getPerp(DFNameInput_h5['Puppi'][0,:], DFNameInput_h5['Puppi'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['Puppi_Pt'], DFName['Puppi_Phi'] = kar2pol(DFNameInput_h5['Puppi'][0,:], DFNameInput_h5['Puppi'][1,:])
    print('len DFName', len(DFName['Boson_Pt']))

    return(DFName)

def loadDataZ(filesDir, rootInput,ll, Wswitch):
    Wswitch = 'false'
    NN_MVA = h5py.File("%sNN_MVA_Zmm.h5"%(filesDir), "r+")
    DFNameInput_h5 = h5py.File("%sNN_Input_apply_Zmm.h5"%(filesDir), "r+")
    #DFNameInput = loadRoot(rootInput,ll,filesDir, Wswitch)
    DFName = pd.DataFrame(index=np.arange(len(NN_MVA['NN_LongZ'])))
    DFnames = ['NN_LongZ', 'NN_PerpZ', 'NN_x', 'NN_y', 'NN_Pt', 'NN_Phi', 'Boson_Pt', 'Boson_Phi', 'Boson_x', 'Boson_y']
    for col in DFnames:
        DFName[col] = NN_MVA[col]
    if Wswitch=='true':
        DFName['genBosonTransMass'] = DFNameInput_h5['genBosonTransMass'][0,:]   
    else: 
        DFName['Px_2'] = DFNameInput_h5['Lepton2'][0,:]    
        DFName['Py_2'] = DFNameInput_h5['Lepton2'][1,:]

    DFName['weights'] = DFNameInput_h5['weights'][0,:]         
    DFName['Px_1'] = DFNameInput_h5['Lepton1'][0,:]    
    DFName['Py_1'] = DFNameInput_h5['Lepton1'][1,:]    
    DFName['NVertex'] = DFNameInput_h5['NVertex'][0,:]
    DFName['PF_LongZ'] = getLong(DFNameInput_h5['PF'][0,:], DFNameInput_h5['PF'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['PF_PerpZ'] = getPerp(DFNameInput_h5['PF'][0,:], DFNameInput_h5['PF'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['PF_Pt'], DFName['PF_Phi'] = kar2pol(DFNameInput_h5['PF'][0,:], DFNameInput_h5['PF'][1,:])
    DFName['Puppi_LongZ'] = getLong(DFNameInput_h5['Puppi'][0,:], DFNameInput_h5['Puppi'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['Puppi_PerpZ'] = getPerp(DFNameInput_h5['Puppi'][0,:], DFNameInput_h5['Puppi'][1,:], DFNameInput_h5['Target'][0,:],DFNameInput_h5['Target'][1,:])
    DFName['Puppi_Pt'], DFName['Puppi_Phi'] = kar2pol(DFNameInput_h5['Puppi'][0,:], DFNameInput_h5['Puppi'][1,:])
    print('len DFName', len(DFName['Boson_Pt']))

    return(DFName)


def getResponse(METlong):
    Response = -DFName[METlong]/DFName['Boson_Pt']
    Response = Response[~np.isnan(Response)]
    return Response
def getResponse_pTRange( METlong, rangemin, rangemax):
    Response = -DFName[METlong][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<=rangemax) ]/DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<=rangemax) ]
    PhiStr = METlong.replace('LongZ','Phi')
    array=getAngle(PhiStr)
    return array[(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<=rangemax) ], Response


def getResponseIdx(METlong):
    Response = -DFName[METlong]/DFName['Boson_Pt']
    Response = Response[~np.isnan(Response)]
    return ~np.isnan(Response)

def getAngle(METPhi):
    if METPhi=='NN_Phi':
        NN_r, deltaPhi = kar2pol(-DFName['NN_LongZ'], DFName['NN_PerpZ'])
    else:
        deltaPhi = angularrange(DFName[METPhi]-DFName['Boson_Phi'])
    return deltaPhi


ScaleErr_Response_PV=1

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c



def plotMVAResponseOverpTZ_woutError(branchString, labelName, errbars_shift, ScaleErr):

    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getResponse(branchString)))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]/DFName['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc = np.mean((getResponse(branchString)))
    stdc = np.std((getResponse(branchString)))
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAmedianResponseOverpTZ_woutError(branchString, labelName, errbars_shift, ScaleErr):

    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getResponse(branchString)))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]/DFName['Boson_Pt'])**2)
    end, start = _[1:], _[:-1]
    median = [np.median(getResponse(branchString)[(DFName['Boson_Pt']>=start[i]) & (DFName['Boson_Pt']<end[i])]) for i in range(0,len(start))]
    if branchString=="NN_LongZ":
        print("median NN", median)
        print("median NN, len(getResponse(branchString)))", len(getResponse(branchString)))
        print("median NN, [(DFName['Boson_Pt']>=start[0]) & (DFName['Boson_Pt']<end[0])]", [(DFName['Boson_Pt']>=start[0]) & (DFName['Boson_Pt']<end[0])])
        print("any nans", np.any(np.isnan(getResponse(branchString)[(DFName['Boson_Pt']>=start[0]) & (DFName['Boson_Pt']<end[0])])))
    #mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    meanc = np.median((getResponse(branchString)))
    stdc = np.std((getResponse(branchString)))
    plt.errorbar((_[1:] + _[:-1])/2, median, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVANormOverpTZ(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    if branchString=='NN_LongZ':
        NN_Pt = np.sqrt(np.multiply(DFName['NN_LongZ'],DFName['NN_LongZ'])+np.multiply(DFName['NN_PerpZ'],DFName['NN_PerpZ']))
        sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(NN_Pt/DFName['Boson_Pt']))
        sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(NN_Pt/DFName['Boson_Pt'])**2)
        meanc = np.mean(NN_Pt/DFName['Boson_Pt'])
        stdc = np.std(NN_Pt/DFName['Boson_Pt'])
    else:
        sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=((DFName[branchString])/DFName['Boson_Pt']))
        sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]/DFName['Boson_Pt'])**2)
        meanc = np.mean(DFName[branchString]/DFName['Boson_Pt'])
        stdc = np.std(DFName[branchString]/DFName['Boson_Pt'])
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ(branchStringPhi, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=getAngle(branchStringPhi))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getAngle(branchStringPhi))**2)
    meanc = np.mean(getAngle(branchStringPhi))
    stdc = np.std(getAngle(branchStringPhi))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVANormOverpTZ_wErr(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'][getResponseIdx(branchString)], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'][getResponseIdx(branchString)], bins=nbins, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName['Boson_Pt'][getResponseIdx(branchString)], bins=nbins, weights=getResponse(branchString)**2)
    meanc = np.mean(getResponse(branchString))
    stdc = np.std(getResponse(branchString))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAAngularOverpTZ_wErr(branchStringPhi, labelName, errbars_shift, ScaleErr):
    nbins=1000
    ScaleErr=1
    #binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getAngle(branchStringPhi)))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getAngle(branchStringPhi))**2)
    meanc = np.mean(getAngle(branchStringPhi))
    stdc = np.std(getAngle(branchStringPhi))
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #print('std', std)
    Bool45Degrees = std<45*np.pi/180

    indSmaller45Degrees = [i for i,x in enumerate(Bool45Degrees) if x==True]
    #print(std<45*np.pi/180)
    #print(np.where(std<45*np.pi/180))
    #print(indSmaller45Degrees)
    print('MET definition std under 45 degrees', labelName)
    print('pT bin Start with std under 45 degrees', _[indSmaller45Degrees[0:10]])
    #print('pT bin End with std under 45 degrees', _[indSmaller45Degrees[0:10]]+binwidth)
    print('crosscheck std with std under 45 degrees', std[indSmaller45Degrees[0:10]])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVAResponseOverpTZ_woutError_Tresh(branchString, labelName, errbars_shift, ScaleErr):
    DFName_Tresh = DFName[DFName['Boson_Pt']>pTTresh]
    binwidth = (DFName_Tresh.Boson_Pt.values.max() - DFName_Tresh.Boson_Pt.values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins)
    sy, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins, weights=(-(DFName_Tresh[branchString])/DFName_Tresh.Boson_Pt))
    sy2, _ = np.histogram(DFName_Tresh.Boson_Pt, bins=nbins, weights=(DFName_Tresh[branchString]/DFName_Tresh.Boson_Pt)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResponseOverNVertex_woutError(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    IndexRange = (DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)
    DF_Response_PV = DFName[(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    binwidth = (DF_Response_PV.NVertex.values.max() - DF_Response_PV.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Response_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Response_PV.NVertex, bins=nbinsVertex, weights=getResponse(branchString)[IndexRange])
    sy2, _ = np.histogram(DF_Response_PV.NVertex, bins=nbinsVertex, weights=(getResponse(branchString)[IndexRange])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc=np.mean(getResponse(branchString))
    stdc=np.std(getResponse(branchString))
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def plotMVAResponseOverNVertex_woutError_Tresh(branchString, labelName, errbars_shift, ScaleErr):
    DFName_Tresh = DFName[DFName['Boson_Pt']>pTTresh]
    binwidth = (DFName_Tresh.NVertex.values.max() - DFName_Tresh.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName_Tresh.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName_Tresh.NVertex, bins=nbinsVertex, weights=-(DFName_Tresh[branchString])/DFName_Tresh['Boson_Pt'])
    sy2, _ = np.histogram(DFName_Tresh.NVertex, bins=nbinsVertex, weights=(DFName_Tresh[branchString]/DFName_Tresh['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def pT_PVbins(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #IdxPVbins = (DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemax)
    DV_PVbins = DFName[(DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemax)]
    binwidth = (DV_PVbins.NVertex.values.max() - DV_PVbins.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex, weights=DV_PVbins['Boson_Pt'])
    sy2, _ = np.histogram(DV_PVbins.NVertex, bins=nbinsVertex, weights=(DV_PVbins['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc= np.mean(DV_PVbins['Boson_Pt'])
    stdc=np.std(DV_PVbins['Boson_Pt'])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])



def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName.NVertex.values.max() - DFName.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])


    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def plotMVAResolutionOverpTZ_woutError_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=-(DFName[branchString]+DFName['Boson_Pt']))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]+DFName['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #rangemin, rangemax = 20, 200
    stdc = np.std((-(DFName[branchString])-DFName['Boson_Pt']))
    #stdc = np.std(-(DFName[branchString]))
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_para_woutOutliers(branchString, labelName, errbars_shift, ScaleErr):
    woutOutliers = (-(DFName[branchString])-DFName['Boson_Pt'])
    problembinmin, problembinmax = 60, 80
    woutOutliers_5_95 = (woutOutliers >= np.percentile(woutOutliers,0.1)) & (woutOutliers <= np.percentile(woutOutliers,99.9)) 
    #& (DFName['Boson_Pt'] >= problembinmin) & (DFName['Boson_Pt'] <= problembinmax)
    Outliers_5_95 = [not i for i in woutOutliers_5_95]
    perpStr = branchString.replace("_LongZ", "_PerpZ")
    branchStringPhi = branchString.replace("_LongZ", "_Phi")
        
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    
    n, _ = np.histogram(DFName['Boson_Pt'][woutOutliers_5_95], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'][woutOutliers_5_95], bins=nbins, weights=-(DFName[branchString][woutOutliers_5_95]+DFName['Boson_Pt'][woutOutliers_5_95]))
    sy2, _ = np.histogram(DFName['Boson_Pt'][woutOutliers_5_95], bins=nbins, weights=(DFName[branchString][woutOutliers_5_95]+DFName['Boson_Pt'][woutOutliers_5_95])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    #rangemin, rangemax = 20, 200
    stdc = np.std((-(DFName[branchString][woutOutliers_5_95])-DFName['Boson_Pt'][woutOutliers_5_95]))
    #stdc = np.std(-(DFName[branchString]))
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def removeMaxOutlier(DFName):
    branchString = "PF_LongZ"
    resPara = -(DFName[branchString]+DFName['Boson_Pt'])
    mask = np.ones(DFName[branchString].shape,dtype=bool) 
    mask[resPara.idxmin()] = False 
    mask[resPara.idxmax()] = False 
    return DFName[mask]

def removeMaxOutlier_xy(DFName):
    TargetPt = DFName['Boson_Pt'][:]
    Pf_r, Pf_phi = kar2pol(DFName['PF'][0,:], DFName['PF'][1,:]) 
    Targ_r, Targ_phi = kar2pol(DFName['Target'][0,:], DFName['Target'][1,:])
    upara = Pf_r * np.cos(Pf_phi-Targ_phi)
    resPara = upara-Targ_r
    print(" index to remove ", np.argmax(resPara), np.argmin(resPara))
    return np.argmax(resPara), np.argmin(resPara)

def plotOutliers(branchString, labelName, errbars_shift, ScaleErr, ax1, ax2, ax3, ax4, ax5, ax6):
    woutOutliers = (-(DFName[branchString])-DFName['Boson_Pt'])
    problembinmin, problembinmax = 60, 80
    woutOutliers_5_95 = (woutOutliers >= np.percentile(woutOutliers,1)) & (woutOutliers <= np.percentile(woutOutliers,99)) 
    #& (DFName['Boson_Pt'] >= problembinmin) & (DFName['Boson_Pt'] <= problembinmax)
    Outliers_5_95 = [not i for i in woutOutliers_5_95]
    perpStr = branchString.replace("_LongZ", "_PerpZ")
    branchStringPhi = branchString.replace("_LongZ", "_Phi")
    Angle = getAngle(branchStringPhi)
    resPara = -(DFName[branchString][Outliers_5_95]+DFName['Boson_Pt'][Outliers_5_95])
    Outliers_5_95_woutmin = Outliers_5_95
    Outliers_5_95_woutmin[np.argmin(resPara)] = False
    resParawoutMin = -(DFName[branchString][Outliers_5_95_woutmin]+DFName['Boson_Pt'][Outliers_5_95_woutmin])
    print("max Resolution para", np.min(resPara))
    print("NVertex of min reso para", DFName['NVertex'][np.argmin(resPara)])
    print("Boson pT of min reso para", DFName['Boson_Pt'][np.argmin(resPara)])
    print("para of min reso para", DFName[branchString][np.argmin(resPara)])
    print("####### Outliers investigation for ", branchString, " ####### ")
    #print("print boolean 1 to 99 percent ", Outliers_5_95)
    print("print sum boolean 1 to 99 percent ", sum(Outliers_5_95))
    print("Outliers  NVertex<1 percent and >99 percent resolution deviation: ", np.mean(DFName['NVertex'][Outliers_5_95]), np.std(DFName['NVertex'][Outliers_5_95]))    
    print("Outliers  Boson pT <1 percent and >99 percent resolution deviation: ", np.mean(DFName['Boson_Pt'][Outliers_5_95]), np.std(DFName['Boson_Pt'][Outliers_5_95]))    
    print("Outliers  Long Z  <1 percent and >99 percent resolution deviation: ", np.mean(DFName[branchString][Outliers_5_95]), np.std(DFName[branchString][Outliers_5_95]))  
    print("Outliers  Perp Z  <1 percent and >99 percent resolution deviation: ", np.mean(DFName[perpStr][Outliers_5_95]), np.std(DFName[perpStr][Outliers_5_95]))      
    print("Outliers  Long Z-boson pT  <1 percent and >99 percent resolution deviation: ", np.mean(resPara), np.std(resPara))    
    print("Outliers  delta alpha abs  <1 percent and >99 percent resolution deviation: ", np.mean(np.abs(getAngle(branchStringPhi))), np.std(np.abs(getAngle(branchStringPhi))))
    ax1.hist(DFName['NVertex'][Outliers_5_95], bins=48, range=[2, 51], label=labelName+'%8.2f $\pm$ %8.2f'%(np.mean(DFName['NVertex'][Outliers_5_95]),np.std(DFName['NVertex'][Outliers_5_95])), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    ax2.hist(DFName['Boson_Pt'][Outliers_5_95], bins=nbinsHist, range=[20, 180], label=labelName+'%8.2f $\pm$ %8.2f'%(np.mean(DFName['Boson_Pt'][Outliers_5_95]),np.std(DFName['Boson_Pt'][Outliers_5_95])), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    
    ax4.hist(resPara, bins=nbinsHist, range=[np.min(resPara), np.max(resPara)], label=labelName+'%8.2f $\pm$ %8.2f'%(np.mean(resPara),np.std(resPara)), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    
    ax3.hist(Angle[Outliers_5_95], bins=nbinsHist, range=[-np.pi, np.pi], label=labelName+'%8.2f $\pm$ %8.2f'%(np.mean(Angle[Outliers_5_95]),np.std(Angle[Outliers_5_95])), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    ax5.hist(resParawoutMin, bins=nbinsHist, range=[np.min(resParawoutMin), np.max(resParawoutMin)], label=labelName+'%8.2f $\pm$ %8.2f'%(np.mean(resParawoutMin),np.std(resParawoutMin)), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    ax6.hist(np.min(resPara), bins=nbinsHist, range=[np.min(resPara)-1, np.min(resPara)+1], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
   

def mean_Response(branchString_Long, branchString_Phi, labelName, errbars_shift, ScaleErr):
    binwidth = (-np.pi - np.pi)/binsAngle
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    NN_phi = getAngle(branchString_Phi)[getResponseIdx(branchString_Long)]

    n, _ = np.histogram(NN_phi, bins=20)
    sy, _ = np.histogram(NN_phi, bins=20, weights=getResponse(branchString_Long))
    sy2, _ = np.histogram(NN_phi, bins=20, weights=(getResponse(branchString_Long))**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)

def mean_Response_wErr(branchString_Long, branchString_Phi, labelName, errbars_shift, ScaleErr, rangemin, rangemax, nbins):
    binwidth = (-np.pi - np.pi)/(binsAngle)
    IndexRangeResponse = (getResponseIdx(branchString_Long)) & (DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)
    IndexRange = (DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)
    NN_phi = getAngle(branchString_Phi)[IndexRangeResponse]

    n, _ = np.histogram(NN_phi, bins=nbins)
    sy, _ = np.histogram(NN_phi, bins=nbins, weights=getResponse(branchString_Long)[IndexRange])
    sy2, _ = np.histogram(NN_phi, bins=nbins, weights=(getResponse(branchString_Long)[IndexRange])**2)
    mean = sy / n
    #std = np.sqrt(sy2/n - mean*mean)
    #print('pT von grossen Errors von ', branchString_Long, ' ist ', DFName['Boson_Pt'][getResponseIdx(branchString_Long) and np.abs(getResponse(branchString_Long))>10])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    std = 1. / n
    print('std', std)
    print('n', n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift2), mean, yerr=std*0.1, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)



def mean_Response_CR(branchString_Long, branchString_Phi, labelName, errbars_shift, ScaleErr):
    binwidth = (-np.pi/180*10 - np.pi/180*10)/binsAngle
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    NN_phi = getAngle(branchString_Phi)[getResponseIdx(branchString_Long)]

    n, _ = np.histogram(NN_phi, bins=20, range=[-np.pi/180*10, np.pi/180*10])
    sy, _ = np.histogram(NN_phi, bins=20, weights=getResponse(branchString_Long), range=[-np.pi/180*10, np.pi/180*10])
    sy2, _ = np.histogram(NN_phi, bins=20, weights=getResponse(branchString_Long)**2, range=[-np.pi/180*10, np.pi/180*10])
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/2*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift], linewidth=1.0)


def plotMVAResolutionOverNVertex_woutError_para(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #IdxRange = (DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemin)
    Rall = getResponse(branchString)
    
    DF_Resolution_PV = DFName[(DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemax)]
    R = Rall[(DF_Resolution_PV.NVertex>40)]
    binwidth = (50)/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=-(DF_Resolution_PV[branchString]+DF_Resolution_PV['Boson_Pt']))
    sy2, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=(DF_Resolution_PV[branchString]+DF_Resolution_PV['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc, stdc = np.mean(mean), np.std((-(DFName[branchString])-DFName['Boson_Pt']))
    print("Resolution PV para of ", branchString, " is ", std)
    print("Resolution PV last bin RC para of ", branchString, " is ", std[-1]/np.mean(R))
    print("Resolution PV para of ", branchString, " parameter a, b ", getab((_[1:] + _[:-1])/2, std))
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def MeanDeviation_Pt(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=-(DFName[branchString]+DFName['Boson_Pt']))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]+DFName['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])


def MeanDeviation_PV(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName.NVertex.values.max() - DFName.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=-(DFName[branchString]+DFName['Boson_Pt']))
    sy2, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(DFName[branchString]+DFName['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=-(DFName[branchString]))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc=np.mean(-(DFName[branchString]))
    stdc=np.std(-(DFName[branchString]))
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])




def Histogram_Deviation_para_pT(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean(-(DFName[branchString])-DFName['Boson_Pt'].values)
    Std = np.std(-(DFName[branchString])-DFName['Boson_Pt'].values)
    if branchString in ['NN_LongZ', 'PF_LongZ']:
        plt.hist((-(DFName[branchString])-DFName['Boson_Pt'].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+'%8.2f'%(Std), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist((-(DFName[branchString])-DFName['Boson_Pt'].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+'%8.2f'%(Std), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])


def Histogram_Deviation_perp_pT(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean((DFName[branchString]))
    Std = np.std((DFName[branchString]))
    if branchString in ['NN_PerpZ', 'PF_PerpZ']:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Hist_Response(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #DFName.loc[DFName['Boson_Pt']]
    Response = -(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])/DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    Response = Response[~np.isnan(Response)]
    Mean = np.mean(Response)
    Std = np.std(Response)

    plt.hist(Response, bins=nbinsHist, range=[ResponseMin, ResponseMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Hist_InvMET(branchString, labelName, errbars_shift, ScaleErr):
    Response2 = np.divide(-(DFName[branchString]), DFName['Boson_Pt'])
    Response = np.divide(1, DFName['Boson_Pt'])
    Response = Response[~np.isnan(Response2)]
    Mean = np.mean(Response)
    Std = np.std(Response)

    plt.hist(Response, bins=nbinsHist, range=[ResponseMin, ResponseMax], label='$\\frac{1}{p_T^{\ \\chi}}$, %8.2f $\pm$ %8.2f'%(Mean, Std), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])


def Hist_Resolution_para(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    Mean = np.mean((-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])-DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
    Std = np.std((-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])-DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
    plt.hist((-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])-DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Resolution_para_woutOutliers(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    woutOutliers = (-(DFName[branchString])-DFName['Boson_Pt'])
    woutOutliers_5_95 = woutOutliers[(woutOutliers >= np.percentile(woutOutliers,1)) & (woutOutliers <= np.percentile(woutOutliers,99))]
    Mean = np.mean(woutOutliers_5_95)
    Std = np.std(woutOutliers_5_95)
    plt.hist(woutOutliers_5_95, bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])


def getstdcpara(branchString, rangemin, rangemax):
    return np.std((-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])-DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))

def Histogram_Response_filled(branchString, labelName, bin, ScaleErr):
    Mean = np.mean(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Std = np.std(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Reso = np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values)
    n, _ = np.histogram(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax])
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin], stacked=False)
    plt.hist(Reso[~np.isnan(Reso)], bins=int(0.5*nbinsHistBin), range=[ResponseMin, 1] ,  histtype='stepfilled', normed=False, color='b', alpha=0.2, label=None)
    plt.hist(Reso[~np.isnan(Reso)], bins=int(0.5*nbinsHistBin), range=[1, ResponseMax] ,  histtype='stepfilled', normed=False, color='r', alpha=0.2, label=None)

def Histogram_Response_NN(branchString, labelName, bin, ScaleErr):
    Mean = np.mean(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Std = np.std(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Reso = np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values)
    n, _ = np.histogram(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax])
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin], stacked=False, linewidth=3.0)


def Hist_Resolution_para_RC(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    u_para = -(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])
    pZ = DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    RC = np.divide((u_para-pZ),np.divide(u_para,pZ))
    Mean = np.mean(RC)
    Std = np.std(RC)
    plt.hist(RC, bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Hist_Resolution_perp_RC(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    u_para = -(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])
    u_perp = DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    pZ = DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    RC = np.divide((u_perp),np.divide(u_para,pZ))
    Mean = np.mean(RC)
    Std = np.std(RC)
    plt.hist(RC, bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])


def Hist_Resolution_para_0_100(branchString, labelName, errbars_shift, ScaleErr):
    DFName_DC = DFName
    DFName_DC = DFName_DC[DFName_DC['Boson_Pt']<=100]
    Mean = np.mean((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']))
    Std = np.std((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']))
    plt.hist((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Hist_Resolution_para_100_150(branchString, labelName, errbars_shift, ScaleErr):
    DFName_DC = DFName
    DFName_DC = DFName_DC[DFName_DC['Boson_Pt']>100]
    DFName_DC = DFName_DC[DFName_DC['Boson_Pt']<=150]
    Mean = np.mean((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']))
    Std = np.std((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']))
    plt.hist((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Hist_Resolution_para_150_200(branchString, labelName, errbars_shift, ScaleErr):
    DFName_DC = DFName
    DFName_DC = DFName_DC[DFName_DC['Boson_Pt']>150]
    DFName_DC = DFName_DC[DFName_DC['Boson_Pt']<=200]
    Mean = np.mean((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']))
    Std = np.std((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']))
    plt.hist((-(DFName_DC[branchString])-DFName_DC['Boson_Pt']), bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])


def Hist_Resolution_perp(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    Mean = np.mean(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])
    Std = np.std(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])
    plt.hist(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)], bins=nbinsHist, range=[ResolutionPerpMin, ResolutionPerpMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])


def Histogram_Deviation_para_Bin(branchString, labelName, bin):
    Mean = np.mean(-(DFName[branchString])-DFName['Boson_Pt'].values)
    Std = np.std(-(DFName[branchString])-DFName['Boson_Pt'].values)
    n, _ = np.histogram(-(DFName[branchString])-DFName['Boson_Pt'].values, bins=nbinsHistBin)
    plt.hist((-(DFName[branchString])-DFName['Boson_Pt'].values), bins=nbinsHistBin, range=(HistLimMin, HistLimMax), label=labelName, histtype='step', lw=3.0,  ec=colors2[bin])


def Histogram_Response(branchString, labelName, bin, ScaleErr):
    Mean = np.mean(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Std = np.std(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Reso = np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName['Boson_Pt'].values, bins=nbinsHistBin)
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin])

def Histogram_Response_ax(branchString, labelName, bin, ScaleErr, ax):
    Mean = np.mean(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Std = np.std(np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values))
    Reso = np.divide(-(DFName[branchString]),DFName['Boson_Pt'].values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName['Boson_Pt'].values, bins=nbinsHistBin)
    ax.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin])

def Histogram_Mass_ax(Masse, labelName, bin, ScaleErr, ax, rangemin, rangemax):
    Mass = Masse[(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    Mean = np.mean(Mass)
    Std = np.std(Mass)
    Reso = Mass
    ax.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[MassMin, MassMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin])

def Histogram_Mass_ax_median(Masse, labelName, bin, ScaleErr, ax, rangemin, rangemax):
    Mass = Masse[(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    Mean = np.median(Mass)
    CI1, CI2 = np.percentile(Mass, 50)-np.percentile(Mass, 50-34), np.percentile(Mass, 50+34)-np.percentile(Mass, 50)
    Reso = Mass
    ax.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[MassMin, MassMax], label=labelName+': %8.2f $+$ %8.2f $-$ %8.2f'%(Mean, CI2, CI1), histtype='step', lw=3.0,  ec=colors_InOut[bin])


def Histogram_Resolution_para_ax(branchString, labelName, bin, ScaleErr, ax):
    Mean = np.mean((-(DFName[branchString])-DFName['Boson_Pt']))
    Std = np.std((-(DFName[branchString])-DFName['Boson_Pt']))
    ax.hist(-(DFName[branchString])-DFName['Boson_Pt'], bins=nbinsHist, range=[ResolutionParaMin, ResolutionParaMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin])

    #ax.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin])

def Histogram_Resolution_perp_ax(branchString, labelName, bin, ScaleErr, ax):
    Mean = np.mean(DFName[branchString])
    Std = np.std(DFName[branchString])
    ax.hist(DFName[branchString], bins=nbinsHist, range=[ResolutionPerpMin, ResolutionPerpMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin])


def Histogram_Response_bin(branchString, labelName, bin, ScaleErr, rangemin, rangemax):
    Mean = np.mean(np.divide(-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]),DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)].values))
    Std = np.std(np.divide(-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]),DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)].values))
    Reso = np.divide(-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]),DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)].values)
    #n, _ = np.histogram(-(DFName[branchString])-DFName['Boson_Pt'].values, bins=nbinsHistBin)
    plt.hist(Reso[~np.isnan(Reso)], bins=nbinsHistBin, range=[ResponseMin, ResponseMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[bin])




def Histogram_Angle_Dev(branchStringPhi, labelName, errbars_shift, ScaleErr):
    print('getAngle(branchStringPhi).shape', getAngle(branchStringPhi).shape)
    if branchStringPhi in ['NN_Phi', 'PF_Phi']:
        plt.hist(getAngle(branchStringPhi), bins=nbinsHist, range=[-np.pi, np.pi], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist(getAngle(branchStringPhi), bins=nbinsHist, range=[-np.pi, np.pi], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Histogram_pT_Dev(pT, labelName, errbars_shift, pTTarget):
    print("pT difference for ", labelName, " mean, std, CI ", np.mean(np.subtract(pT, pTTarget)), np.std(np.subtract(pT, pTTarget)), getCI(np.subtract(pT, pTTarget)))
    plt.hist(np.subtract(pT, pTTarget), bins=nbinsHist, range=[-60, 60], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])



def Histogram_Norm(branchStringNorm,  labelName, errbars_shift, ScaleErr):
    Norm_ = DFName[branchStringNorm]
    Mean = np.mean(Norm_)
    Std = np.std(Norm_)
    if branchStringNorm in ['NN_Pt', 'PF_Pt']:
        plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Histogram_Norm_Pt(branchStringLong, labelName, errbars_shift, ScaleErr):
    Norm_ = DFName['Boson_Pt'].values
    Mean = np.mean(DFName['Boson_Pt'].values)
    Std = np.std(Norm_)
    plt.hist(Norm_, bins=nbinsHist, range=[0, 75], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)


def Histogram_Deviation_para_PV(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean(-(DFName[branchString])-DFName['Boson_Pt'].values)
    Std = np.std(-(DFName[branchString])-DFName['Boson_Pt'].values)
    if branchString in ['NN_LongZ', 'PF_LongZ']:
        plt.hist((-(DFName[branchString])-DFName['Boson_Pt'].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+'%8.2f'%(Std), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist((-(DFName[branchString])-DFName['Boson_Pt'].values), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+'%8.2f'%( Std), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Histogram_Response_asy2(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #DFName.loc[DFName['Boson_Pt']]
    Response = -(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])/DFName['Boson_Pt'][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]
    Response = Response[~np.isnan(Response)]

    return Response

def Histogram_Resolution_para_asy2(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #DFName.loc[DFName['Boson_Pt']]
    Res = (-(DFName[branchString])-DFName['Boson_Pt'])

    return Res

def Histogram_Resolution_perp_asy2(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #DFName.loc[DFName['Boson_Pt']]
    Res = DFName[branchString]

    return Res

def Histogram_Deviation_perp_PV(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean((DFName[branchString]))
    Std = np.std((DFName[branchString]))
    if branchString in ['NN_PerpZ', 'PF_PerpZ']:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+'%8.2f'%(Std), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName+'%8.2f'%(Std), histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Hist_LongZ(branchString, labelName, errbars_shift, ScaleErr):
    if branchString == 'Boson_Pt':
        Mean = np.mean((DFName[branchString]))
        Std = np.std((DFName[branchString]))
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[0, 75], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)

    else:
        if branchString in ['NN_LongZ', 'PF_LongZ']:
            Mean = np.mean(-(DFName[branchString]))
            Std = np.std(-(DFName[branchString]))
            plt.hist((-(DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
        else:
            Mean = np.mean(-(DFName[branchString]))
            Std = np.std(-(DFName[branchString]))
            plt.hist((-(DFName[branchString])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])



def Hist_LongZ_bin(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    if branchString == 'Boson_Pt':
        Mean = np.mean((DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
        Std = np.std((DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
        plt.hist(((DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])), bins=nbinsHist, range=[0, 75], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)

    else:
        if branchString in ['NN_LongZ', 'PF_LongZ']:
            Mean = np.mean(-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
            Std = np.std(-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
            plt.hist((-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
        else:
            Mean = np.mean(-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
            Std = np.std(-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
            plt.hist((-(DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])



def Hist_PerpZ(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean((DFName[branchString]))
    Std = np.std((DFName[branchString]))
    if branchString in ['NN_PerpZ', 'PF_PerpZ']:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[-60, 60], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist(((DFName[branchString])), bins=nbinsHist, range=[-60, 60], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])

def Hist_PerpZ_bin(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    Mean = np.mean((DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
    Std = np.std((DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)]))
    if branchString in ['NN_PerpZ', 'PF_PerpZ']:
        plt.hist(((DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift], linewidth=3.0)
    else:
        plt.hist(((DFName[branchString][(DFName['Boson_Pt']>rangemin) & (DFName['Boson_Pt']<rangemax)])), bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors_InOut[errbars_shift])




def NN_Response_pT( labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=getResponse('NN_LongZ'))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=getResponse('NN_LongZ')**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def NN_Response_PV(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName.NVertex.values.max() - DFName.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverpTZ_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getResponse(branchString)))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

def plotMVAResponseOverNVertex_wError(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName.NVertex.values.max() - DFName.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    mean = sy / n
    std = np.divide(np.sqrt(sy2/n - mean*mean), n)
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.fill_between((_[1:] + _[:-1])/2, mean-std*ScaleErr, mean+std*ScaleErr, alpha=0.2, edgecolor=colors[errbars_shift], facecolor=colors[errbars_shift])

    #plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])
    #plt.plot((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , label=labelName, linestyle="None", color=MVAcolors[errbars_shift])
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])




def plotMVAResolutionOverpTZ_woutError_para_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    Rall = getResponse(branchString)
    
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=-(DFName[branchString]+DFName['Boson_Pt']))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]+DFName['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    
    nR, _R = np.histogram(DFName['Boson_Pt'], bins=nbins)
    syR, _R = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=Rall)
    sy2R, _R = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(np.add(Rall,-1))**2)
    meanR = syR / nR


    plt.errorbar((_[1:] + _[:-1])/2, np.divide(std,meanR), marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])

    




def plotMVAResolutionOverNVertex_woutError_perp(branchString, labelName, errbars_shift, ScaleErr, rangemin, rangemax):
    #IdxRange = (DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemin)
    branchstr_2 = branchString.replace('Perp','Long')
    Rall = getResponse(branchstr_2)
    
    DF_Resolution_pe_PV = DFName[(DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemax)]

    R = Rall[(DF_Resolution_pe_PV.NVertex>40)]
    binwidth = (50)/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=-(DF_Resolution_pe_PV[branchString]))
    sy2, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=(DF_Resolution_pe_PV[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    meanc=np.mean((DF_Resolution_pe_PV[branchString]))
    stdc=np.std((DF_Resolution_pe_PV[branchString]))
    print("Resolution PV perp of ", branchString, " is ", std)
    print("Resolution PV last bin RC perp of ", branchString, " is ", std[-1]/np.mean(R))
    print("Resolution PV perp of ", branchString, " parameter a, b ", getab((_[1:] + _[:-1])/2, std))
    plt.errorbar((_[1:] + _[:-1])/2, std, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])


def plotMVAResolutionOverNVertex_woutError_para_RC(branchString, labelName, errbars_shift, ScaleErr):
    rangemin, rangemax = 20, 200
    #IdxRange = (DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemin)
    Rall = getResponse(branchString)
    
    DF_Resolution_PV = DFName[(DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemax)]

    n, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=-(DF_Resolution_PV[branchString]+DF_Resolution_PV['Boson_Pt']))
    sy2, _ = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=(DF_Resolution_PV[branchString]+DF_Resolution_PV['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    nR, _R = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex)
    syR, _R= np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=getResponse(branchString))
    sy2R, _R = np.histogram(DF_Resolution_PV.NVertex, bins=nbinsVertex, weights=(getResponse(branchString))**2)
    meanR = syR / nR

    
    meanc=np.mean((DF_Resolution_PV[branchString]))
    stdc=np.std((DF_Resolution_PV[branchString]))
    print("Resolution PV para of ", branchString, " is ", std)
    print("mean PV response ", branchString, " is ", meanR)
    print("Resolution PV last bin RC para of ", branchString, " is ", std[-1]/np.mean(getResponse(branchString)))
    print("###########    Resolution PV para RC of ", branchString, " parameter a, b ", getab((_[1:] + _[:-1])/2, np.divide(std,meanR)))
    RC_res = np.divide(std,meanR)
    x = (_[1:] + _[:-1])/2
    print("Resolution PV para RC are  ", branchString, " is ", RC_res)
    print("linear function parameters of ", branchString, " are ", getFctParameters(x, RC_res))
    
    
    plt.errorbar((_[1:] + _[:-1])/2, np.divide(std,meanR), marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def getFctParameters(x_data, y_data):
    x_data = x_data.astype(np.float64)
    y_data = y_data.astype(np.float64)

    p0=[ 0.1, 15.0]
    params, params_covariance = optimize.curve_fit(Function, x_data, y_data, p0=p0)
    print('Fitted Parametrization with parameters ', params)
    return(params)
def Function(x, a, c):
    return a*x+c

def plotMVAResolutionOverNVertex_woutError_perp_RC(branchString, labelName, errbars_shift, ScaleErr):
    rangemin, rangemax = 20, 200
    #IdxRange = (DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemin)
    branchstr_2 = branchString.replace('Perp','Long')
    Rall = getResponse(branchstr_2)
    
    DF_Resolution_pe_PV = DFName[(DFName['Boson_Pt']>=rangemin) & (DFName['Boson_Pt']<rangemax)]

    R = Rall[(DF_Resolution_pe_PV.NVertex>40)]
    binwidth = (50)/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=-(DFName[branchString]+DFName['Boson_Pt']))
    sy2, _ = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=(DFName[branchString]+DFName['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    
    nR, _R = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex)
    syR, _R = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=Rall)
    sy2R, _R = np.histogram(DF_Resolution_pe_PV.NVertex, bins=nbinsVertex, weights=(np.add(Rall,-1))**2)
    meanR = syR / nR
    
    meanc=np.mean((DF_Resolution_pe_PV[branchString]))
    stdc=np.std((DF_Resolution_pe_PV[branchString]))
    print("Resolution PV perp of ", branchString, " is ", std)
    print("Resolution PV last bin RC perp of ", branchString, " is ", std[-1]/np.mean(R))
    print("Resolution PV perp of ", branchString, " parameter a, b ", getab((_[1:] + _[:-1])/2, std))
    
    plt.errorbar((_[1:] + _[:-1])/2, np.divide(std,meanR), marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def plotMVAResolutionOverpTZ_woutError_perp_RC(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName.NVertex.values.max() - DFName.NVertex.values.min())/(nbinsVertex) 
    branchstr_2 = branchString.replace('Perp','Long')
    Rall = getResponse(branchstr_2)
    
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=-(DFName[branchString]+DFName['Boson_Pt']))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]+DFName['Boson_Pt'])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    
    nR, _R = np.histogram(DFName['Boson_Pt'], bins=nbins)
    syR, _R = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=Rall)
    sy2R, _R = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(np.add(Rall,-1))**2)
    meanR = syR / nR

    plt.errorbar((_[1:] + _[:-1])/2, np.divide(std,meanR), marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors_InOut[errbars_shift])

def Histogram_Deviation_perp(branchString, labelName, errbars_shift, ScaleErr):
    Mean = np.mean(DFName[branchString])
    Std = np.std(DFName[branchString])
    plt.hist(DFName[branchString], bins=nbinsHist, range=[HistLimMin, HistLimMax], label=labelName, histtype='step', lw=3.0,  ec=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(-(DFName[branchString])-DFName['Boson_Pt'].values))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(-(DFName[branchString])-DFName['Boson_Pt'].values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_pTZ_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName['Boson_Pt'].values.max() - DFName['Boson_Pt'].values.min())/(nbins) #5MET-Definitionen
    n, _ = np.histogram(DFName['Boson_Pt'], bins=nbins)
    sy, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName['Boson_Pt'], bins=nbins, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_para(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName.NVertex.values.max() - DFName.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(-(DFName[branchString])-DFName.NVertex.values))
    sy2, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(-(DFName[branchString])-DFName.NVertex.values)**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])

def Mean_Std_Deviation_PV_perp(branchString, labelName, errbars_shift, ScaleErr):
    binwidth = (DFName.NVertex.values.max() - DFName.NVertex.values.min())/(nbinsVertex) #5MET-Definitionen
    n, _ = np.histogram(DFName.NVertex, bins=nbinsVertex)
    sy, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(DFName[branchString]))
    sy2, _ = np.histogram(DFName.NVertex, bins=nbinsVertex, weights=(DFName[branchString])**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, marker='o', elinewidth=3.0 , xerr=(_[1:]-_[:-1])/2, label=labelName, linestyle="None", capsize=0,  color=colors[errbars_shift])
    if errbars_shift==5: errbars_shift2 = 0
    elif errbars_shift==6: errbars_shift2 = 2
    else: errbars_shift2 = errbars_shift
    plt.errorbar((_[:-1]+(_[1:]-_[:-1])/3*errbars_shift2), mean, yerr=std*ScaleErr, marker='', linestyle="None", capsize=0,  color=colors[errbars_shift])





def getPlotsOutput(inputD, filesD, plotsD, DFName):
    chi = []
    if Wswitch=='true':
        chi='W'
    else:
        chi='Z'


    #Plot settings
    ScaleErr = 1
    NPlotsLines = 6
    MVA_NPlotsLines = 3
    pTRangeString_Err = '$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$ \\ Error scaled: '%(pTMin,pTMax)+str(ScaleErr)
    pTRangeString= '$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(pTMin,pTMax)
    pTRangeString_low= pTRangeString_mid= pTRangeString_high= pTRangeString



    pTRangeString_Tresh = '$1\ \mathrm{GeV} < p_T^{\ \chi} \leq 200\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'
    pTRangeStringNVertex = pTRangeString
    if 'Boson_Pt'=='Boson_Pt':
        LegendTitle = 'MET definition: mean $\pm$ standard deviation'
        LegendTitleMedian = 'MET definition: median $\pm$ 34 \% CI'
        LegendTitleSTD = 'MET definition: standard deviation'
    else:
        LegendTitle = '$\mathrm{Summer\ 17\ campaign}$' '\n'  '$\mathrm{Z \  \\rightarrow \ \\tau \\tau   \\rightarrow \ \mu \mu}$'
    NPlotsLines=7
    colors = cm.brg(np.linspace(0, 1, NPlotsLines))
    colors_InOut = cm.brg(np.linspace(0, 1, 8))
    colors = colors_InOut
    MVAcolors =  colors
    ylimResMin, ylimResMax = 7.5 , 50
    ylimResMVAMin_RC, ylimResMax_RC = 0 , 50

    PF_Delta_pT, PF_Delta_Phi = kar2pol(DFName['PF_LongZ'],DFName['PF_PerpZ'])





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plotMVAResponseOverpTZ_woutError('Puppi_LongZ', 'PUPPI', 4, ScaleErr)
    plotMVAResponseOverpTZ_woutError('PF_LongZ', 'PF', 1, ScaleErr)
    plotMVAResponseOverpTZ_woutError('NN_LongZ', 'NN', 6, ScaleErr)
    plt.plot([pTMin, pTMax], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_T^{\ %s} $ in GeV'%(chi), fontsize=22)
    plt.ylabel('$\\langle \\frac{u_{\parallel}}{p_T^{\ %s}} \\rangle$ '%(chi), fontsize=22)
    #plt.title('Response $u_{\parallel}$', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)

    loadAxesSettings(ax)
    plt.ylim(0.7, 1.3)
    plt.xlim(pTMin, pTMax)
    loadAxesSettings(ax)
    plt.savefig("%sResponse_pT_2.png"%(plotsD), bbox_inches="tight")
    plt.close()






    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    loadAxesSettings(ax)

    plotMVAResolutionOverpTZ_woutError_para('Puppi_LongZ', 'PUPPI', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para('PF_LongZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_T^{\ %s} $ in GeV'%(chi), fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\parallel} - p_T^{\ %s} \\right) $ in GeV'%(chi), fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\perp}$', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')    
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(20, 200)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    loadAxesSettings(ax)
    #plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.savefig("%sResolution_para_pT_2.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    loadAxesSettings(ax)

    plotMVAResolutionOverpTZ_woutError_para_RC('Puppi_LongZ', 'PUPPI', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para_RC('PF_LongZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para_RC('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_T^{\ %s} $ in GeV'%(chi), fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\parallel} - p_T^{\ %s} \\right)/ \\langle \\frac{u_{\parallel}}{p_T^{\: %s}}  \\rangle $ in GeV'%(chi,chi), fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\perp}$', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')    
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(20, 200)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    loadAxesSettings(ax)
    #plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.savefig("%sResolution_para_pT_RC.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    loadAxesSettings(ax)

    plotMVAResolutionOverpTZ_woutError_para_woutOutliers('Puppi_LongZ', 'PUPPI', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para_woutOutliers('PF_LongZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para_woutOutliers('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_T^{\ %s} $ in GeV'%(chi), fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\parallel} - p_T^{\ %s} \\right) $ in GeV'%(chi), fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\perp}$', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    #legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(20, 200)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    loadAxesSettings(ax)
    #plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.savefig("%sResolution_para_pT_woutOutliers_01_999.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    loadAxesSettings(ax)

    plotMVAResolutionOverpTZ_woutError_para_woutOutliers('Puppi_LongZ', 'PUPPI', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para_woutOutliers('PF_LongZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_para_woutOutliers('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))
    
    plt.xlabel('$p_T^{\ %s} $ in GeV'%(chi), fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\parallel} - p_T^{\ %s} \\right) $ in GeV'%(chi), fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\perp}$', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    #legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    plt.xlim(20, 200)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    loadAxesSettings(ax)
    #plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.savefig("%sResolution_para_pT_woutOutliers_01_999.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')


    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)    

    plotOutliers('Puppi_LongZ', 'PUPPI', 4, ScaleErr, ax1, ax2, ax3, ax4, ax5, ax6)
    plotOutliers('PF_LongZ', 'PF', 1, ScaleErr, ax1, ax2, ax3, ax4, ax5, ax6)
    plotOutliers('NN_LongZ', 'NN', 6, ScaleErr, ax1, ax2, ax3, ax4, ax5, ax6)
    ax1.title.set_text("nPV")
    ax2.title.set_text("pT")
    ax4.title.set_text("Res para")
    ax3.title.set_text("Winkel")
    ax5.title.set_text("Res para wout min")
    ax6.title.set_text("Res para min")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    plt.savefig("%sOutliers_01_999.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)

    Hist_Resolution_para('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 40, 60)
    Hist_Resolution_para('PF_LongZ', 'PF', 1, ScaleErr, 40, 60)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, 40, 60)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label='$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(20,40)))

    plt.xlabel('$u_{\parallel} - p_T^{\ %s}$ in GeV'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    ax.yaxis.offsetText.set_fontsize(18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_40_60.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)

    Hist_Resolution_para('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 60, 80)
    Hist_Resolution_para('PF_LongZ', 'PF', 1, ScaleErr, 60, 80)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, 60, 80)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label='$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(20,40)))

    plt.xlabel('$u_{\parallel} - p_T^{\ %s}$ in GeV'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.offsetText.set_fontsize(18)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)   
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_60_80.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)

    Hist_Resolution_para('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 80, 100)
    Hist_Resolution_para('PF_LongZ', 'PF', 1, ScaleErr, 80, 100)
    Hist_Resolution_para('NN_LongZ', 'NN', 6, ScaleErr, 80, 100)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label='$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(20,40)))

    plt.xlabel('$u_{\parallel} - p_T^{\ %s}$ in GeV'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)

    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18) 
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_80_100.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)

    Resolution_para_woutOutliers('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 40, 60)
    Resolution_para_woutOutliers('PF_LongZ', 'PF', 1, ScaleErr, 40, 60)
    Resolution_para_woutOutliers('NN_LongZ', 'NN', 6, ScaleErr, 40, 60)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label='$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(20,40)))

    plt.xlabel('$u_{\parallel} - p_T^{\ %s}$ in GeV'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    ax.yaxis.offsetText.set_fontsize(18)  
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_40_60_woutOutliers_5_95.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)

    Resolution_para_woutOutliers('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 60, 80)
    Resolution_para_woutOutliers('PF_LongZ', 'PF', 1, ScaleErr, 60, 80)
    Resolution_para_woutOutliers('NN_LongZ', 'NN', 6, ScaleErr, 60, 80)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label='$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(20,40)))

    plt.xlabel('$u_{\parallel} - p_T^{\ %s}$ in GeV'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    ax.yaxis.offsetText.set_fontsize(18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_60_80_woutOutliers_1_99.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    #Hist_Resolution_para('LongZCorrectedRecoil_LongZ', 'GBRT', 5, ScaleErr, 0, 200)

    Resolution_para_woutOutliers('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 80, 100)
    Resolution_para_woutOutliers('PF_LongZ', 'PF', 1, ScaleErr, 80, 100)
    Resolution_para_woutOutliers('NN_LongZ', 'NN', 6, ScaleErr, 80, 100)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label='$%8.2f\ \mathrm{GeV} < p_T^{\ \chi} \leq %8.2f\ \mathrm{GeV}$ \n $\mathrm{\# Vertex} \leq 50$'%(20,40)))

    plt.xlabel('$u_{\parallel} - p_T^{\ %s}$ in GeV'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    ax.yaxis.offsetText.set_fontsize(18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax )
    plt.savefig("%sHist_Resolution_para_80_100_woutOutliers_5_95.png"%(plotsD), bbox_inches="tight")
    plt.close()



    #######u perp ######
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plotMVAResolutionOverpTZ_woutError_perp('Puppi_PerpZ', 'PUPPI', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp('PF_PerpZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_T^{\ %s} $ in GeV'%(chi), fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\perp} \\right) $ in GeV', fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\perp}$', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    #legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', title=LegendTitleStd, numpoints=1, framealpha=1.0	)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')  
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.xlim(20, 200)
    plt.savefig("%sResolution_perp_pT_2.png"%(plotsD), bbox_inches="tight")
    plt.close()


    #######u perp ######
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plotMVAResolutionOverpTZ_woutError_perp_RC('Puppi_PerpZ', 'PUPPI', 4, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp_RC('PF_PerpZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverpTZ_woutError_perp_RC('NN_PerpZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.xlabel('$p_T^{\ %s} $ in GeV'%(chi), fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\perp}  \\right) / \\langle \\frac{u_{\parallel}}{p_T^{\:%s}} \\rangle$ in GeV'%(chi), fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\perp}$', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    #legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', title=LegendTitleStd, numpoints=1, framealpha=1.0	)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')  
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.xlim(20, 200)
    plt.savefig("%sResolution_perp_pT_RC.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)


    plotMVAResponseOverNVertex_woutError('Puppi_LongZ', 'PUPPI', 4, ScaleErr_Response_PV, 0, 200)
    plotMVAResponseOverNVertex_woutError('PF_LongZ', 'PF', 1, ScaleErr_Response_PV, 0, 200)
    plotMVAResponseOverNVertex_woutError('NN_LongZ', 'NN', 6, ScaleErr_Response_PV, 0, 200)
    plt.plot([0, 50], [1, 1], color='k', linestyle='--', linewidth=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()

    plt.xlabel('$N_{\mathrm{Vtx}}$ ', fontsize=22)
    plt.ylabel('$\\langle \\frac{u_{\parallel}}{p_T^{\ %s}} \\rangle$ '%(chi), fontsize=22)



    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')  
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.ylim(0.7, 1.3)
    plt.xlim(0, 50)
    plt.savefig("%sResponse_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plotMVAResolutionOverNVertex_woutError_para('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_para('PF_LongZ', 'PF', 1, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_para('NN_LongZ', 'NN', 6, ScaleErr, 0, 200)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()

    plt.xlabel('$N_{\mathrm{Vtx}}$ ', fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\parallel} - p_T^{\ %s} \\right) $ in GeV'%(chi), fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$')

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.xlim(0, 50)
    plt.savefig("%sResolution_para_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plotMVAResolutionOverNVertex_woutError_para_RC('Puppi_LongZ', 'PUPPI', 4, ScaleErr)
    plotMVAResolutionOverNVertex_woutError_para_RC('PF_LongZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverNVertex_woutError_para_RC('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()

    plt.xlabel('$N_{\mathrm{Vtx}}$ ', fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\parallel} + \\vec{p}_T^{\ %s} \\right) / \\langle  \\frac{u_{\parallel}}{p_T^{\:%s}} \\rangle $ in GeV'%(chi,chi), fontsize=22)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\parallel}}{p_T^{\ \chi}} \\right) $ in GeV')
    #plt.title('Resolution $u_{\parallel}$')

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.setp(legend.get_texts(), fontsize='18')
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.xlim(0, 50)
    plt.savefig("%sResolution_para_PV_RC.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plotMVAResolutionOverNVertex_woutError_perp('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, 0, 200)

    plotMVAResolutionOverNVertex_woutError_perp('PF_PerpZ', 'PF', 1, ScaleErr, 0, 200)
    plotMVAResolutionOverNVertex_woutError_perp('NN_PerpZ', 'NN', 6, ScaleErr, 0, 200)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()

    plt.xlabel('$N_{\mathrm{Vtx}}$ ', fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\perp} \\right) $ in GeV', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)

    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.xlim(0, 50)
    plt.savefig("%sResolution_perp_PV.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    plotMVAResolutionOverNVertex_woutError_perp_RC('Puppi_PerpZ', 'PUPPI', 4, ScaleErr)

    plotMVAResolutionOverNVertex_woutError_perp_RC('PF_PerpZ', 'PF', 1, ScaleErr)
    plotMVAResolutionOverNVertex_woutError_perp_RC('NN_PerpZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()

    plt.xlabel('$N_{\mathrm{Vtx}}$ ', fontsize=22)
    plt.ylabel('$\sigma \\left( u_{\perp} \\right) / \\langle \\frac{u_{\parallel}}{p_T^{\:%s}}  \\rangle$ in GeV'%(chi), fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)

    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.ylim(ylimResMVAMin, ylimResMVAMax)
    plt.xlim(0, 50)
    plt.savefig("%sResolution_perp_PV_RC.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_Angle_Dev('Puppi_Phi',  'Puppi', 4, ScaleErr)
    Histogram_Angle_Dev('PF_Phi',  'PF', 1, ScaleErr)
    Histogram_Angle_Dev('NN_Phi', 'NN', 6, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$ \\Delta \\alpha $ in rad', fontsize=22)
    plt.xlim(-np.pi,np.pi)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    ax.yaxis.offsetText.set_fontsize(18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%sHist_Delta_Alpha.png"%(plotsD), bbox_inches="tight")
    plt.close()





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)


    Hist_LongZ_bin('Puppi_LongZ', 'PUPPI',4, ScaleErr, 20, 40)
    Hist_LongZ_bin('PF_LongZ', 'PF', 1, ScaleErr, 20, 40)
    Hist_LongZ_bin('NN_LongZ', 'NN', 6, ScaleErr, 20, 40)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$ u_{\parallel}   $ in GeV', fontsize=22)
    plt.xlim(HistLimMin,HistLimMax)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18) 
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%sHist_para_20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    Hist_PerpZ('Puppi_PerpZ', 'PUPPI', 4, ScaleErr)
    Hist_PerpZ('PF_PerpZ', 'PF', 1, ScaleErr)
    Hist_PerpZ('NN_PerpZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$ u_{\perp}  $ in GeV', fontsize=22)
    plt.xlim(-60,60)


    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18) 
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%sHist_perp.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    Hist_PerpZ_bin('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, 20, 40)
    Hist_PerpZ_bin('PF_PerpZ', 'PF', 1, ScaleErr, 20, 40)
    Hist_PerpZ_bin('NN_PerpZ', 'NN', 6, ScaleErr, 20, 40)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString))

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$ u_{\perp}  $ in GeV', fontsize=22)
    plt.xlim(HistLimMin,HistLimMax)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    ax.yaxis.offsetText.set_fontsize(18)  
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%sHist_perp_20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    nbinsHist = 40

    InputsTargets_hdf5 = h5py.File("%sNN_Input_apply_%s.h5" % (filesD,ll), "r")
    print("Keys: %s" % InputsTargets_hdf5.keys())
    keys = InputsTargets_hdf5.keys()
    values = [InputsTargets_hdf5[k].value for k in keys]
    #InputsTargets2 = pd.DataFrame(index=np.arange(NN_Output_applied["MET_GroundTruth"].shape))
    d, _ = np.histogram(InputsTargets_hdf5['NVertex'][:], range=[0,75], normed=True, bins=75)

    print("length index", len(np.arange(len(InputsTargets_hdf5['Target'][0,:]))))
    print("length values", len(InputsTargets_hdf5['Target'][0,:]))
    InputsTargets = pd.Series()
    Norm_pT = np.sqrt(np.multiply(InputsTargets_hdf5['Target'][0,:], InputsTargets_hdf5['Target'][0,:]) + np.multiply(InputsTargets_hdf5['Target'][1,:], InputsTargets_hdf5['Target'][1,:]))
    for k, v in zip(keys, values):
            InputsTargets[k] = np.delete(v, list(removeMaxOutlier_xy(InputsTargets_hdf5)), 1)
            print(type(InputsTargets[k]))
            print(k)
    print(type(InputsTargets))
    
    
    IndTest = np.arange(len(InputsTargets['PF'][0,:]))
    Targets = np.transpose(InputsTargets['Target'])
    NN_Diff_x = np.subtract(DFName['NN_x'], DFName['Boson_x'])
    NN_Diff_y = np.subtract(DFName['NN_y'], DFName['Boson_y'])
    NN_Output_applied = h5py.File("%sNN_Output_applied_%s.h5"%(filesD,ll), "r")
    predictions = NN_Output_applied["MET_Predictions"]
    predictions = np.delete(predictions, list(removeMaxOutlier_xy(InputsTargets_hdf5)), 0)
    print("Laenge InputsTargets", InputsTargets['Puppi'].shape)
    plt.hist(np.subtract(InputsTargets['Puppi'][0,IndTest], Targets[IndTest,0]), bins=nbinsHist, range=[-50, 50], label='PUPPI', histtype='step', lw=3.0,  ec=colors_InOut[4])
    plt.hist(np.subtract(InputsTargets['PF'][0,IndTest], Targets[IndTest,0]), bins=nbinsHist, range=[-50, 50], label='PF', histtype='step', lw=3.0,  ec=colors_InOut[1], linewidth=3.0)
    plt.hist(NN_Diff_x[:], bins=nbinsHist, range=[-50, 50], label='NN', histtype='step', lw=3.0,  ec=colors_InOut[6], linewidth=3.0)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$ \\Delta p_{T,x}  $ in GeV', fontsize=22)
    plt.xlim(-50,50)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
    #plt.text('$p_T$ range restriction')

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18) 
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_Delta_x.png"%(plotsD), bbox_inches="tight")
    plt.close()
    
    NN_pT = np.sqrt(np.square(DFName['NN_x'])+np.square(DFName['NN_y']))
    Puppi_pT = np.sqrt(np.square(InputsTargets['Puppi'][0,IndTest])+np.square(InputsTargets['Puppi'][1,IndTest]))
    PF_pT = np.sqrt(np.square(InputsTargets['PF'][0,IndTest])+np.square(InputsTargets['PF'][1,IndTest]))
    Target_pT = np.sqrt(np.square(DFName['Boson_x'])+np.square(DFName['Boson_y']))

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)


    Histogram_pT_Dev(Puppi_pT,  'Puppi', 4, Target_pT)
    Histogram_pT_Dev(PF_pT,  'PF', 1, Target_pT)
    Histogram_pT_Dev(NN_pT, 'NN', 6, Target_pT)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$ \\Delta p_T $ in GeV', fontsize=22)
    plt.xlim(-60, 60)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)    
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.savefig("%sHist_Delta_pT.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    nbinsHist = 40

    plt.hist(np.subtract(InputsTargets['Puppi'][1,IndTest], Targets[IndTest,1]), bins=nbinsHist, range=[-50, 50], label='PUPPI', histtype='step', lw=3.0,  ec=colors_InOut[4])
    plt.hist(np.subtract(InputsTargets['PF'][1,IndTest], Targets[IndTest,1]), bins=nbinsHist, range=[-50, 50], label='PF', histtype='step', lw=3.0,  ec=colors_InOut[1], linewidth=3.0)
    plt.hist(NN_Diff_y[:], bins=nbinsHist, range=[-50, 50], label='NN', histtype='step', lw=3.0,  ec=colors_InOut[6], linewidth=3.0)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeString_high))

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$ \\Delta p_{T,y}   $ in GeV', fontsize=22)
    plt.xlim(-50,50)
    #plt.ylabel('$\sigma \\left( \\frac{u_{\perp}}{p_{T}^Z} \\right) $ in GeV')
    #plt.title(' Histogram Deviation to Target  $\Delta p_{T,y}$')
    #plt.text('$p_T$ range restriction')

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)   
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    #plt.ylim(ylimResMVAMin, ylimResMax)
    plt.savefig("%sHist_Delta_y.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams
    
    labelsize = 18

    Histogram_Mass_ax(getTransverseMass('NN'),'NN', 6, ScaleErr, ax1, 20, 200)
    Histogram_Mass_ax(getTransverseMass('PF'), 'PF', 1, ScaleErr, ax1, 20, 200)
    Histogram_Mass_ax(getTransverseMass('Puppi'), 'PUPPI', 4, ScaleErr, ax1, 20, 200)
    ax1.axvline(x=MassW)
    ax1.set_xlim(MassMin, MassMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)

    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)
    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [getTransverseMass('NN'),
        getTransverseMass('PF'),
        getTransverseMass('Puppi')]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.mean(getTransverseMass('NN')), np.mean(getTransverseMass('PF')), np.mean(getTransverseMass('Puppi'))]
    bplot1 = ax2.boxplot(x, labels=y, vert=False, patch_artist=True, showfliers=False,  medianprops=medianprops, showmeans=False, usermedians=usermedians)
    plt.yticks([1, 2, 3], y, size="x-large")
    #bplot1.tick_params(axis='both', which='minor', labelsize=18)
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    #plt.subplots_adjust(left=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(MassMin, MassMax)
    plt.xlabel('$  m_W^T $')
    plt.savefig("%sHist_Mass_wBoxplots_mean.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams
    
    labelsize = 18

    Histogram_Mass_ax_median(getTransverseMass('NN'),'NN', 6, ScaleErr, ax1, 20, 200)
    Histogram_Mass_ax_median(getTransverseMass('PF'), 'PF', 1, ScaleErr, ax1, 20, 200)
    Histogram_Mass_ax_median(getTransverseMass('Puppi'), 'PUPPI', 4, ScaleErr, ax1, 20, 200)
    ax1.axvline(x=MassW)
    ax1.set_xlim(MassMin, MassMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)

    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', title=LegendTitleMedian, numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)
    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [getTransverseMass('NN'),
        getTransverseMass('PF'),
        getTransverseMass('Puppi')]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.median(getTransverseMass('NN')), np.median(getTransverseMass('PF')), np.median(getTransverseMass('Puppi'))]
    conf_intervals = [getCIStartEnd(getTransverseMass('NN')), \
                        getCIStartEnd(getTransverseMass('PF')), \
                        getCIStartEnd(getTransverseMass('Puppi'))]
    bplot1 = ax2.boxplot(x, labels=y, vert=False, patch_artist=True, showfliers=False,  medianprops=medianprops, showmeans=False, usermedians=usermedians, conf_intervals=conf_intervals)
    plt.yticks([1, 2, 3], y, size="x-large")
    #bplot1.tick_params(axis='both', which='minor', labelsize=18)
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    #plt.subplots_adjust(left=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(MassMin, MassMax)
    plt.xlabel('$  m_W^T $')
    plt.savefig("%sHist_Mass_wBoxplots_median_68CI.png"%(plotsD), bbox_inches="tight")
    plt.close()


    if Wswitch=='true':
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        loadAxesSettings(ax)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        nbinsHist = 40


        plt.hist(getTransverseMass('Puppi')-DFName['genBosonTransMass'],  range=[DeltaMassMin, DeltaMassMax], label='Puppi, mean=%.2f $\pm$ %.2f'%(np.mean(getTransverseMass('Puppi')-DFName['genBosonTransMass']), np.std(getTransverseMass('Puppi')-DFName['genBosonTransMass'])), histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[4])
        plt.hist(getTransverseMass('PF')-DFName['genBosonTransMass'],  range=[DeltaMassMin, DeltaMassMax], label='PF, mean=%.2f $\pm$ %.2f'%(np.mean(getTransverseMass('PF')-DFName['genBosonTransMass']), np.std(getTransverseMass('PF')-DFName['genBosonTransMass'])), histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[1])
        plt.hist(getTransverseMass('NN')-DFName['genBosonTransMass'],  range=[DeltaMassMin, DeltaMassMax], label='NN, mean=%.2f $\pm$ %.2f'%(np.mean(getTransverseMass('NN')-DFName['genBosonTransMass']), np.std(getTransverseMass('NN')-DFName['genBosonTransMass'])), histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[6])


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()

        plt.ylabel('Counts', fontsize=22)
        plt.xlabel('$\Delta m_W^T$  in  $\\frac{GeV}{c^2}$', fontsize=22)

        legend = plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.00), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
        plt.setp(legend.get_title(),fontsize='x-large')
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(color='black', linestyle='-', linewidth=0.1)
        plt.xlim(DeltaMassMin, DeltaMassMax)
        plt.savefig("%sHist_Diff_TransMass.png"%(plotsD), bbox_inches="tight")
        plt.close()





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams
    
    labelsize = 18

    Histogram_Mass_ax_median(getTransverseMass('NN'),'NN', 6, ScaleErr, ax1, 20, 40)
    Histogram_Mass_ax_median(getTransverseMass('PF'), 'PF', 1, ScaleErr, ax1, 20, 40)
    Histogram_Mass_ax_median(getTransverseMass('Puppi'), 'PUPPI', 4, ScaleErr, ax1, 20, 40)
    ax1.axvline(x=MassW)
    ax1.set_xlim(MassMin, MassMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)

    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', title=LegendTitleMedian, numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)

    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    Ind = (DFName['Boson_Pt']>20) & (DFName['Boson_Pt']<40)
    y = ["NN","PF","Puppi"]
    x = [getTransverseMass('NN')[Ind],
        getTransverseMass('PF')[Ind],
        getTransverseMass('Puppi')[Ind]]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.median(getTransverseMass('NN')[Ind]), np.median(getTransverseMass('PF')[Ind]), np.median(getTransverseMass('Puppi')[Ind])]
    conf_intervals = [getCIStartEnd(getTransverseMass('NN')[Ind]), \
                        getCIStartEnd(getTransverseMass('PF')[Ind]), \
                        getCIStartEnd(getTransverseMass('Puppi')[Ind])]
    bplot1 = ax2.boxplot(x, labels=y, vert=False, patch_artist=True, showfliers=False,  medianprops=medianprops, showmeans=False, usermedians=usermedians, conf_intervals=conf_intervals)
    plt.yticks([1, 2, 3], y, size="x-large")
    #bplot1.tick_params(axis='both', which='minor', labelsize=18)
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    #plt.subplots_adjust(left=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(MassMin, MassMax)
    plt.xlabel('$  m_W^T $')
    plt.savefig("%sHist_Mass_wBoxplots_median_68CI_pT20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams
    
    labelsize = 18

    Histogram_Mass_ax(getTransverseMass('NN'),'NN', 6, ScaleErr, ax1, 20, 40)
    Histogram_Mass_ax(getTransverseMass('PF'), 'PF', 1, ScaleErr, ax1, 20, 40)
    Histogram_Mass_ax(getTransverseMass('Puppi'), 'PUPPI', 4, ScaleErr, ax1, 20, 40)
    ax1.axvline(x=MassW)
    ax1.set_xlim(MassMin, MassMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)

    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)

    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [getTransverseMass('NN'),
        getTransverseMass('PF'),
        getTransverseMass('Puppi')]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.mean(getTransverseMass('NN')[(DFName['Boson_Pt']>20) & (DFName['Boson_Pt']<40)]), np.mean(getTransverseMass('PF')[(DFName['Boson_Pt']>20) & (DFName['Boson_Pt']<40)]), np.mean(getTransverseMass('Puppi')[(DFName['Boson_Pt']>20) & (DFName['Boson_Pt']<40)])]
    

    bplot1 = ax2.boxplot(x, labels=y, vert=False, patch_artist=True, showfliers=False,  medianprops=medianprops, showmeans=False, usermedians=usermedians)
    plt.yticks([1, 2, 3], y, size="x-large")
    #bplot1.tick_params(axis='both', which='minor', labelsize=18)
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    #plt.subplots_adjust(left=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(MassMin, MassMax)
    plt.xlabel('$  m_W^T $')
    plt.savefig("%sHist_Mass_wBoxplots_mean_pT20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams
    DeltaPtNuMIn, DeltaPtNuMax = -60, 60
    labelsize = 18

    ax1.hist(getNeutrino('Puppi'), bins=nbinsHistBin, range=[DeltaPtNuMIn, DeltaPtNuMax], label='PUPPI', histtype='step', lw=3.0,  ec=colors_InOut[4])
    ax1.hist(getNeutrino('PF'), bins=nbinsHistBin, range=[DeltaPtNuMIn, DeltaPtNuMax], label='PF', histtype='step', lw=3.0,  ec=colors_InOut[1])
    ax1.hist(getNeutrino('NN'), bins=nbinsHistBin, range=[DeltaPtNuMIn, DeltaPtNuMax], label='NN', histtype='step', lw=3.0,  ec=colors_InOut[6])

    ax1.set_xlim(DeltaPtNuMIn, DeltaPtNuMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)
   

    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)
    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [getNeutrino('NN'),
        getNeutrino('PF'),
        getNeutrino('Puppi')]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.mean(getNeutrino('NN')), np.mean(getNeutrino('PF')), np.mean(getNeutrino('Puppi'))]
    conf_intervals1 = [getCI(getNeutrino('NN')), \
                        getCI(getNeutrino('PF')), \
                        getCI(getNeutrino('Puppi'))]
   
    
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    ax2.errorbar(usermedians, [1, 2, 3], xerr=np.array(conf_intervals1).T.tolist(), fmt='k|', ecolor=colors, elinewidth=10.0, markersize=20, markerfacecolor='black', markeredgewidth=2.)
    plt.yticks([1, 2, 3], y, size="x-large")
    ax2.set_ylim([0, 4])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(DeltaPtNuMIn, DeltaPtNuMax)
    plt.xlabel('$  |\\vec{p}_T^{\:miss}-\\vec{p}_T^{\:\\nu}| $')
    plt.savefig("%sHist_DeltaPtNu_Errorbar_68CI_mean.png"%(plotsD), bbox_inches="tight")
    plt.close()
    print("Mean  ptmiss-ptnu von NN, PF, Puppi", usermedians)
    print("34 CI  ptmiss-ptnu mean von NN, PF, Puppi",  getCI(getNeutrino('NN')) , getCI(getNeutrino('PF')), getCI(getNeutrino('Puppi')))
    print("34 CI  ptmiss-ptnu median von NN, PF, Puppi",  getCImed(getNeutrino('NN')) , getCImed(getNeutrino('PF')), getCImed(getNeutrino('Puppi')))
    print("Standard deviation  ptmiss-ptnu von NN, PF, Puppi", np.std(getNeutrino('NN')), np.std(getNeutrino('PF')), np.std(getNeutrino('Puppi')))








    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams

    labelsize = 18

    Histogram_Response_ax('Puppi_LongZ', 'PUPPI', 4, ScaleErr, ax1)
    Histogram_Response_ax('PF_LongZ', 'PF', 1, ScaleErr, ax1)
    Histogram_Response_ax('NN_LongZ', 'NN', 6, ScaleErr, ax1)

    ax1.set_xlim(ResponseMin, ResponseMax)
    handles, labels = ax1.get_legend_handles_labels()

    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)


    y = ["NN","PF","Puppi"]
    x = [Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200),
        Histogram_Response_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200),
        Histogram_Response_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200)]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.mean(Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)), np.mean(Histogram_Response_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)), np.mean(Histogram_Response_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200))]
    conf_intervals = [getCIStartEnd(Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)), \
                        getCIStartEnd(Histogram_Response_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)), \
                        getCIStartEnd(Histogram_Response_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200))]
                       
    bplot1 = ax2.boxplot(x, labels=y, vert=False, patch_artist=True, showfliers=False,  medianprops=medianprops, showmeans=False, usermedians=usermedians, conf_intervals=conf_intervals)
    plt.yticks([1, 2, 3], y, size="x-large")
    #bplot1.tick_params(axis='both', which='minor', labelsize=18)
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    #plt.subplots_adjust(left=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax)
    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi))
    plt.savefig("%sHist_Response_wBoxplots_mean_68CI.png"%(plotsD), bbox_inches="tight")
    plt.close()
    print("Mean  Response von NN, PF, Puppi", usermedians)
    print("34 CI Start Ende Response von NN", getCI(Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)))
    print("STD Start Ende Response von NN", np.std(Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)))





    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams

    labelsize = 18

    Histogram_Resolution_para_ax('Puppi_LongZ', 'PUPPI', 4, ScaleErr, ax1)
    Histogram_Resolution_para_ax('PF_LongZ', 'PF', 1, ScaleErr, ax1)
    Histogram_Resolution_para_ax('NN_LongZ', 'NN', 6, ScaleErr, ax1)

    ax1.set_xlim(ResolutionParaMin, ResolutionParaMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)  

    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
   
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.offsetText.set_fontsize(18)  


    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [Histogram_Resolution_para_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200),
        Histogram_Resolution_para_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200),
        Histogram_Resolution_para_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200)]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.mean(Histogram_Resolution_para_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)),np.mean(Histogram_Resolution_para_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)),np.mean(Histogram_Resolution_para_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200))]
    conf_intervals2 = [getCI(Histogram_Resolution_para_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)), \
                        getCI(Histogram_Resolution_para_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)), \
                        getCI(Histogram_Resolution_para_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200))]
   
    
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    ax2.errorbar(usermedians, [1, 2, 3], xerr=np.array(conf_intervals2).T.tolist(), fmt='k|', ecolor=colors, elinewidth=10.0, markersize=20, markerfacecolor='black', markeredgewidth=2.)
    plt.yticks([1, 2, 3], y, size="x-large")
    ax2.set_ylim([0, 4])
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax)
    plt.xlabel('$  u_{\parallel}+\\vec{p}_T^{\ %s} $ in GeV'%(chi))
    plt.savefig("%sHist_Resolution_para_Errorbar_68CI_mean.png"%(plotsD), bbox_inches="tight")
    plt.close()
    print("Mean  Resolution para von NN, PF, Puppi", usermedians)
    print("34 CI  Resolution para von NN, PF, Puppi",  getCI(Histogram_Resolution_para_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)) , getCI(Histogram_Resolution_para_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)) , getCI(Histogram_Resolution_para_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200)) )
    print("Standard deviation  Resolution para von NN, PF, Puppi", np.std(Histogram_Resolution_para_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)), np.std(Histogram_Resolution_para_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)), np.std(Histogram_Resolution_para_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200)))



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams

    labelsize = 18

    Histogram_Response_ax('Puppi_LongZ', 'PUPPI', 4, ScaleErr, ax1)
    Histogram_Response_ax('PF_LongZ', 'PF', 1, ScaleErr, ax1)
    Histogram_Response_ax('NN_LongZ', 'NN', 6, ScaleErr, ax1)

    ax1.set_xlim(ResponseMin, ResponseMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.yaxis.offsetText.set_fontsize(18)  
    loadAxesSettings(ax1)
    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
   
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.offsetText.set_fontsize(18)  


    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200),
        Histogram_Response_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200),
        Histogram_Response_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200)]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians = [np.mean(Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)),np.mean(Histogram_Response_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)),np.mean(Histogram_Response_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200))]
    conf_intervals2 = [getCI(Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)), \
                        getCI(Histogram_Response_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200)), \
                        getCI(Histogram_Response_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200))]
   
    
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    ax2.errorbar(usermedians, [1, 2, 3], xerr=np.array(conf_intervals2).T.tolist(), fmt='k|', ecolor=colors, elinewidth=10.0, markersize=20, markerfacecolor='black', markeredgewidth=2.)
    plt.yticks([1, 2, 3], y, size="x-large")
    ax2.set_ylim([0, 4])
    
    loadAxesSettings(ax2)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax)
    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi))
    plt.savefig("%sHist_Response_Errorbar_68CI_mean.png"%(plotsD), bbox_inches="tight")
    plt.close()




    '''
    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(211)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams

    labelsize = 18


    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [Histogram_Resolution_para_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200),
        Histogram_Resolution_para_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200),
        Histogram_Resolution_para_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200)]
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    bplot1 = ax2.boxplot(x, labels=y, vert=False, patch_artist=True, showfliers=False,  medianprops=medianprops, showmeans=True)
    plt.yticks([1, 2, 3], y, size="x-large")
    #bplot1.tick_params(axis='both', which='minor', labelsize=18)
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    #plt.subplots_adjust(left=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionParaMin, ResolutionParaMax)
    plt.xlabel('$  u_{\parallel}+\\vec{p}_T^{\ %s} $ in GeV'%(chi))
    plt.savefig("%sResult_Resolution_para.png"%(plotsD), bbox_inches="tight")
    plt.close()
    '''


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.yaxis.offsetText.set_fontsize(18)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams

    labelsize = 18

    Histogram_Resolution_perp_ax('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, ax1)
    Histogram_Resolution_perp_ax('PF_PerpZ', 'PF', 1, ScaleErr, ax1)
    Histogram_Resolution_perp_ax('NN_PerpZ', 'NN', 6, ScaleErr, ax1)

    ax1.set_xlim(ResolutionPerpMin, ResolutionPerpMax)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.offsetText.set_fontsize(18)  


    legend = ax1.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')

    #rcParams['xtick.labelsize'] = labelsize
    #rcParams['ytick.labelsize'] = labelsize
    y = ["NN","PF","Puppi"]
    x = [Histogram_Resolution_perp_asy2('NN_PerpZ', 'NN', 6, ScaleErr, 20, 200),
        Histogram_Resolution_perp_asy2('PF_PerpZ', 'PF', 1, ScaleErr, 20, 200),
        Histogram_Resolution_perp_asy2('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, 20, 200)]
    conf_intervals3 = [getCI(Histogram_Resolution_perp_asy2('NN_PerpZ', 'NN', 6, ScaleErr, 20, 200)), \
                        getCI(Histogram_Resolution_perp_asy2('PF_PerpZ', 'PF', 1, ScaleErr, 20, 200)), \
                        getCI(Histogram_Resolution_perp_asy2('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, 20, 200))]
           
    df = pd.DataFrame(x, index=y)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    usermedians3 = [np.mean(Histogram_Resolution_perp_asy2('NN_PerpZ', 'NN', 6, ScaleErr, 20, 200)),np.mean(Histogram_Resolution_perp_asy2('PF_PerpZ', 'PF', 1, ScaleErr, 20, 200)),np.mean(Histogram_Resolution_perp_asy2('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, 20, 200))]
    
    colors = [colors_InOut[6], colors_InOut[1],colors_InOut[4]]
    ax2.errorbar(usermedians3, [1, 2, 3], xerr=np.array(conf_intervals3).T.tolist(), fmt='k|', ecolor=colors, elinewidth=10.0, markersize=20, markerfacecolor='black', markeredgewidth=2.)
    plt.yticks([1, 2, 3], y, size="x-large")
    ax2.set_ylim([0, 4])
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResolutionPerpMin, ResolutionPerpMax)
    plt.xlabel('$  u_{\perp} $ in GeV')
    plt.savefig("%sHist_Resolution_perp_Errorbar_68CI_mean.png"%(plotsD), bbox_inches="tight")
    plt.close()
    print("Mean  Resolution perp von NN, PF, Puppi", usermedians3)
    print("34 CI  Resolution perp von NN, PF, Puppi",  getCI(Histogram_Resolution_perp_asy2('NN_PerpZ', 'NN', 6, ScaleErr, 20, 200)) , getCI(Histogram_Resolution_perp_asy2('PF_PerpZ', 'PF', 1, ScaleErr, 20, 200)) , getCI(Histogram_Resolution_perp_asy2('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, 20, 200)) )
    print("Standard deviation  Resolution perp von NN, PF, Puppi", np.std(Histogram_Resolution_perp_asy2('NN_PerpZ', 'NN', 6, ScaleErr, 20, 200)), np.std(Histogram_Resolution_perp_asy2('PF_PerpZ', 'PF', 1, ScaleErr, 20, 200)), np.std(Histogram_Resolution_perp_asy2('Puppi_PerpZ', 'PUPPI', 4, ScaleErr, 20, 200)))


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(211)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    from matplotlib import rcParams

    labelsize = 18
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    y = ["PF","Puppi", "NN"]
    x = [Histogram_Response_asy2('PF_LongZ', 'PF', 1, ScaleErr, 20, 200),
         Histogram_Response_asy2('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 200),
         Histogram_Response_asy2('NN_LongZ', 'NN', 6, ScaleErr, 20, 200)]
    df = pd.DataFrame(x, index=y)
    bplot1 = plt.boxplot(x, labels=y, vert=False, patch_artist=True, showfliers=False, showmeans=True)
    plt.yticks([1, 2, 3], y, size="x-large")
    #bplot1.tick_params(axis='both', which='minor', labelsize=18)
    colors = [colors_InOut[1],colors_InOut[4], colors_InOut[6]]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    #plt.subplots_adjust(left=0.25)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    #handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax)
    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi))
    plt.savefig("%sBoxplot.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    nbinsHist = 40




    PF_Diff_x, PF_Diff_y = np.subtract(InputsTargets['PF'][0,IndTest], Targets[IndTest,0]), np.subtract(InputsTargets['PF'][1,IndTest], Targets[IndTest,1])
    PF_Diff_norm = np.sqrt(np.multiply(PF_Diff_x,PF_Diff_x)+np.multiply(PF_Diff_y,PF_Diff_y))
    Puppi_Diff_x, Puppi_Diff_y = np.subtract(InputsTargets['Puppi'][0,IndTest], Targets[IndTest,0]), np.subtract(InputsTargets['Puppi'][1,IndTest], Targets[IndTest,1])
    Puppi_Diff_norm = np.sqrt(np.multiply(Puppi_Diff_x,Puppi_Diff_x)+np.multiply(Puppi_Diff_y,Puppi_Diff_y))

    plt.hist(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y)),  range=[0,100], label='PUPPI', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[4])
    plt.hist(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y)),  range=[0,100], label='PF', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[1])
    plt.hist(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y)),  range=[0,100], label='NN', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[6])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()

    plt.ylabel('Counts', fontsize=22)
    plt.xlabel('$p_T^{\:\mathrm{miss}}=|\\vec{u}+\\vec{p}_T^{\ Z}|$ in GeV', fontsize=22)
    plt.xlim(0,100)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(0, 70)   
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.savefig("%sHist_Diff_norm.png"%(plotsD), bbox_inches="tight")
    plt.close()
    print("Puppi Diff norm mean, std, CI left, median, CI right", np.mean(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y))), np.std(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y))), np.percentile(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y)),50-34), np.percentile(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y)),50), np.percentile(np.sqrt(np.multiply(Puppi_Diff_x, Puppi_Diff_x)+np.multiply(Puppi_Diff_y, Puppi_Diff_y)),50+34))
    print("PF Diff norm mean, std, CI left, median, CI right", np.mean(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y))), np.std(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y))), np.percentile(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y)),50-34), np.percentile(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y)),50), np.percentile(np.sqrt(np.multiply(PF_Diff_x, PF_Diff_x)+np.multiply(PF_Diff_y, PF_Diff_y)),50+34))
    print("NN Diff norm mean, std, CI left, median,  CI right", np.mean(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y))), np.std(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y))), np.percentile(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y)),50-34), np.percentile(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y)),50), np.percentile(np.sqrt(np.multiply(NN_Diff_x, NN_Diff_x)+np.multiply(NN_Diff_y, NN_Diff_y)),50+34))
    

    if False:
        #W+Jets result
        PF_Diff_x, PF_Diff_y = np.subtract(InputsTargets['PF'][0,IndTest], Targets[IndTest,0]), np.subtract(InputsTargets['PF'][1,IndTest], Targets[IndTest,1])
        PF_Diff_norm = np.sqrt(np.multiply(PF_Diff_x,PF_Diff_x)+np.multiply(PF_Diff_y,PF_Diff_y))
        Puppi_Diff_x, Puppi_Diff_y = np.subtract(InputsTargets['Puppi'][0,IndTest], Targets[IndTest,0]), np.subtract(InputsTargets['Puppi'][1,IndTest], Targets[IndTest,1])
        Puppi_Diff_norm = np.sqrt(np.multiply(Puppi_Diff_x,Puppi_Diff_x)+np.multiply(Puppi_Diff_y,Puppi_Diff_y))
        pT_miss_Puppi_W = getpTMiss('Puppi')
        pT_miss_PF_W = getpTMiss('PF')
        pT_miss_NN_W = getpTMiss('NN')
           
        pT_miss_Puppi_Z = getpTMissZ('Puppi')
        pT_miss_PF_Z = getpTMissZ('PF')
        pT_miss_NN_Z = getpTMissZ('NN')
        
        
        mT_Puppi_W = getTransverseMass('Puppi')
        mT_PF_W = getTransverseMass('PF')
        mT_NN_W = getTransverseMass('NN')
           
        mT_Puppi_Z = getTransverseMassZ('Puppi')
        mT_PF_Z = getTransverseMassZ('PF')
        mT_NN_Z = getTransverseMassZ('NN')  
        print(" Mean mT NN ", np.mean(getTransverseMassZ('NN')))      
        
        def plot_roc(good_pdf, bad_pdf, ax, labelName, col):
            #Total
            total_bad = np.sum(bad_pdf)
            total_good = np.sum(good_pdf)
            #Cumulative sum
            cum_TP = 0
            cum_FP = 0
            #TPR and FPR list initialization
            TPR_list=[]
            FPR_list=[]
            #Iteratre through all values of nbinsHist
            x=np.linspace(0, 1, num=nbinsHist) 
            for i in range(len(x)):
                #We are only interested in non-zero values of bad
                if bad_pdf[i]>0:
                    cum_TP+=bad_pdf[len(x)-1-i]
                    cum_FP+=good_pdf[len(x)-1-i]
                FPR=cum_FP/total_good
                TPR=cum_TP/total_bad
                TPR_list.append(TPR)
                FPR_list.append(FPR)
            #Calculating AUC, taking the 100 timesteps into account
            auc=np.sum(TPR_list)/nbinsHist
            print(" Auc ", auc, " of ", labelName)
            #Plotting final ROC curve 
            ax.plot(FPR_list, TPR_list, color=colors_InOut[col], label="%s AUC=%.3f"%(labelName, auc))
            if col==4:
                ax.plot(x,x, "--")
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                ax.set_ylabel('TPR', fontsize=18)
                ax.set_xlabel('FPR', fontsize=18)
                ax.grid()
     
        
        
        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        loadAxesSettings(ax)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        nbinsHist = 40


        pTmissMin, pTmissMax = 0, 200


        plt.hist(pT_miss_Puppi_W,  range=[pTmissMin, pTmissMax], label='PUPPI', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[4], normed=True)
        plt.hist(pT_miss_PF_W,  range=[pTmissMin, pTmissMax], label='PF', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[1], normed=True)
        plt.hist(pT_miss_NN_W,  range=[pTmissMin, pTmissMax], label='NN', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[6], normed=True)
        
        plt.hist(pT_miss_Puppi_Z,  range=[pTmissMin, pTmissMax], label=None, histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[4], linestyle='--', normed=True)
        plt.hist(pT_miss_PF_Z,  range=[pTmissMin, pTmissMax], label=None, histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[1], linestyle='--', normed=True)
        plt.hist(pT_miss_NN_Z,  range=[pTmissMin, pTmissMax], label=None, histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[6], linestyle='--', normed=True)
        

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()

        plt.ylabel('Density', fontsize=22)
        plt.xlabel('$|\\vec{p}_T^{\:miss}|$ in GeV', fontsize=22)
        plt.xlim(0,100)

        legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
        plt.setp(legend.get_title(),fontsize='x-large')
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(color='black', linestyle='-', linewidth=0.1)
        plt.xlim(pTmissMin, pTmissMax)   
        plt.setp(legend.get_texts(), fontsize='18')
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_fontsize(18)
        plt.savefig("%sHist_pT_miss_ZW.png"%(plotsD), bbox_inches="tight")
        plt.close()


        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        loadAxesSettings(ax)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        nbinsHist = 40


        pTmissMin, pTmissMax = 0, 200


        plt.hist(mT_Puppi_W,  range=[pTmissMin, pTmissMax], label='PUPPI', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[4], normed=True)
        plt.hist(mT_PF_W,  range=[pTmissMin, pTmissMax], label='PF', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[1], normed=True)
        plt.hist(mT_NN_W,  range=[pTmissMin, pTmissMax], label='NN', histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[6], normed=True)
        
        plt.hist(mT_Puppi_Z,  range=[pTmissMin, pTmissMax], label=None, histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[4], linestyle='--', normed=True)
        plt.hist(mT_PF_Z,  range=[pTmissMin, pTmissMax], label=None, histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[1], linestyle='--', normed=True)
        plt.hist(mT_NN_Z,  range=[pTmissMin, pTmissMax], label=None, histtype='step', lw=3.0,  bins=nbinsHist, ec=colors_InOut[6], linestyle='--', normed=True)
        

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()

        plt.ylabel('Density', fontsize=22)
        plt.xlabel('$m_T$ in GeV', fontsize=22)
        plt.xlim(0,100)

        legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
        plt.setp(legend.get_title(),fontsize='x-large')
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(color='black', linestyle='-', linewidth=0.1)
        plt.xlim(pTmissMin, pTmissMax)   
        plt.setp(legend.get_texts(), fontsize='18')
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_fontsize(18)
        plt.savefig("%sHist_mT_ZW.png"%(plotsD), bbox_inches="tight")
        plt.close()

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        bad_pdf_Puppi, _ = np.histogram(pT_miss_Puppi_W,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        bad_pdf_PF, _  = np.histogram(pT_miss_PF_W,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        bad_pdf_NN, _  = np.histogram(pT_miss_NN_W,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        good_pdf_Puppi, _ = np.histogram(pT_miss_Puppi_Z,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        good_pdf_PF, _  = np.histogram(pT_miss_PF_Z,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        good_pdf_NN, _  = np.histogram(pT_miss_NN_Z,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)        
        plot_roc(good_pdf_Puppi, bad_pdf_Puppi, ax, 'PUPPI', 4)
        plot_roc(good_pdf_PF, bad_pdf_PF, ax, 'PF', 1)
        plot_roc(good_pdf_NN, bad_pdf_NN, ax, 'NN', 6)
        ax.yaxis.offsetText.set_fontsize(18)
        plt.savefig("%sROC_pTMiss_ZW.png"%(plotsD), bbox_inches="tight")
        plt.close()

        fig=plt.figure(figsize=(10,6))
        fig.patch.set_facecolor('white')
        ax = plt.subplot(111)
        bad_pdf_Puppi, _ = np.histogram(mT_Puppi_W,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        bad_pdf_PF, _  = np.histogram(mT_PF_W,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        bad_pdf_NN, _  = np.histogram(mT_NN_W,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        good_pdf_Puppi, _ = np.histogram(mT_Puppi_Z,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        good_pdf_PF, _  = np.histogram(mT_PF_Z,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)
        good_pdf_NN, _  = np.histogram(mT_NN_Z,  range=[pTmissMin, pTmissMax], bins=nbinsHist, normed=True)        
        plot_roc(good_pdf_Puppi, bad_pdf_Puppi, ax, 'PUPPI', 4)
        plot_roc(good_pdf_PF, bad_pdf_PF, ax, 'PF', 1)
        plot_roc(good_pdf_NN, bad_pdf_NN, ax, 'NN', 6)
        ax.yaxis.offsetText.set_fontsize(18)
        plt.savefig("%sROC_mT_ZW.png"%(plotsD), bbox_inches="tight")
        plt.close()
        

    fig=plt.figure(figsize=(10,10))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    Histogram_Response('Puppi_LongZ', 'PUPPI', 4, ScaleErr)
    Histogram_Response('PF_LongZ', 'PF', 1, ScaleErr)
    Histogram_Response('NN_LongZ', 'NN', 6, ScaleErr)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.00), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax)   
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.savefig("%sHist_Response.png"%(plotsD), bbox_inches="tight")
    plt.close()



    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    Histogram_Response_filled('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.00), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax)   
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.savefig("%sHist_Response_filled.png"%(plotsD), bbox_inches="tight")
    plt.close()

    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    Histogram_Response_NN('NN_LongZ', 'NN', 6, ScaleErr)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(1.05, 1.00), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax)   
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.savefig("%sHist_Response_NN.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    Histogram_Response_bin('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 20, 40)
    Histogram_Response_bin('NN_LongZ', 'NN', 6, ScaleErr, 20, 40)
    Histogram_Response_bin('PF_LongZ', 'PF', 1, ScaleErr, 20, 40)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)

    legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0., title='$20 \mathrm{GeV}  < p_T^{\ %s} \leq 40 \mathrm{GeV}$'%(chi), fontsize='x-large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter) 
    ax.yaxis.offsetText.set_fontsize(18)  
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax) 
    plt.savefig("%sHist_Response_20_40.png"%(plotsD), bbox_inches="tight")
    plt.close()


    fig=plt.figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)
    loadAxesSettings(ax)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    Histogram_Response_bin('Puppi_LongZ', 'PUPPI', 4, ScaleErr, 60, 80)
    Histogram_Response_bin('NN_LongZ', 'NN', 6, ScaleErr, 60, 80)
    Histogram_Response_bin('PF_LongZ', 'PF', 1, ScaleErr, 60, 80)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

    plt.xlabel('$  \\frac{u_{\parallel}}{p_T^{\ %s}}} $'%(chi), fontsize=22)
    plt.ylabel('Counts', fontsize=22)

    legend = plt.legend(ncol=1, handles=handles, bbox_to_anchor=(1.05, 1.00), loc=4, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	)
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.xlim(ResponseMin, ResponseMax)   
    plt.setp(legend.get_texts(), fontsize='18')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18)
    plt.savefig("%sHist_Response_60_80.png"%(plotsD), bbox_inches="tight")
    plt.close()

    if Wswitch=='true':
        #nPV
    
    
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
        pltmax = 75
        plt.hist(DFName["NVertex"], bins=75, range=[pltmin, pltmax], histtype='step', lw=3.0,  linewidth=2.0, ec='black', normed=True, label='Fall17')
        

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

        plt.xlim(pltmin,50)
        plt.ylabel('Density', fontsize=22)
        plt.xlabel('$N_{\mathrm{Vtx}}$', fontsize=22)


        legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.90), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
        plt.setp(legend.get_title(),fontsize='x-large')
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(color='black', linestyle='-', linewidth=0.1)    
        plt.setp(legend.get_texts(), fontsize='18')
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_fontsize(18)
        plt.savefig("%s%s.png"%(plotsD,'NVertex'), bbox_inches="tight")
        plt.close()
    else:
    
        FileOld = "%sNN_Input_apply_xy_old.h5"%(filesDir)
        InputsOld, TargetsOld = loadInputsTargets(FileOld)
        
                
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
        pltmax = 75
        plt.hist(DFName["NVertex"], bins=75, range=[pltmin, pltmax], histtype='step', lw=3.0,  linewidth=2.0, ec='black', normed=True, label='Fall17')
        plt.hist(InputsOld["NVertex"], bins=75, range=[pltmin, pltmax], histtype='step', lw=3.0,  linewidth=2.0, ec='mediumblue', normed=True, label='Summer17')
        

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        handles, labels = ax.get_legend_handles_labels()
        ##handles.insert(0,mpatches.Patch(color='none', label=pTRangeStringNVertex))

        plt.xlim(pltmin,50)
        plt.ylabel('Density', fontsize=22)
        plt.xlabel('$N_{\mathrm{Vtx}}$', fontsize=22)


        legend = plt.legend(ncol=1, bbox_to_anchor=(0.95, 0.90), loc=1, borderaxespad=0., fontsize='x-large', numpoints=1, framealpha=1.0	, handles=[Line2D([], [], linewidth=3.0, c=h.get_edgecolor()) for h in handles],  labels=labels)
        plt.setp(legend.get_title(),fontsize='x-large')
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(color='black', linestyle='-', linewidth=0.1)    
        plt.setp(legend.get_texts(), fontsize='18')
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_fontsize(18)
        loadAxesSettings(ax)
        plt.savefig("%s%s.png"%(plotsD,'NVertex'), bbox_inches="tight")
        plt.close()        



if __name__ == "__main__":
    rootPath = sys.argv[1]
    filesDir =  sys.argv[2]
    plotDir = sys.argv[3]
    plotDir = plotDir+"Thesis/"
    ll = sys.argv[4]
    Wswitch = sys.argv[5]
    DFName_wOutliers = loadData(filesDir, rootPath, ll, Wswitch)
    DFName = removeMaxOutlier(DFName_wOutliers)
    print("keys DFName", DFName.keys())
    getPlotsOutput(rootPath, filesDir, plotDir, DFName)
