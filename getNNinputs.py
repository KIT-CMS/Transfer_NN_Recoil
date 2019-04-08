import numpy as np
import root_numpy as rnp
import ROOT
import pandas as pd
import sys
import h5py
from scipy import optimize
import matplotlib as mpl
from matplotlib.lines import Line2D
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True) 
formatter.set_scientific(True) 
formatter.set_powerlimits((-2,2)) 

pTMin, pTMax = 20, 200
VertexMin, VertexMax = 2, 50

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

def pol2kar(norm, phi):
    x = np.cos(phi)*norm
    y = np.sin(phi)*norm
    return(x, y)

def kar2pol(x, y):
    rho = np.sqrt(np.multiply(x,x) + np.multiply(y,y))
    phi = np.arctan2(y, x)
    return(rho, phi)

def angularrange(Winkel):
    if isinstance(Winkel, (list, tuple, np.ndarray)):
        for i in range(0, len(Winkel) ):
            Winkel[i]=((Winkel[i]+np.pi)%(2*np.pi)-(np.pi))
    else:
        Winkel=((Winkel+np.pi)%(2*np.pi)-(np.pi))
    return(Winkel)

def pandasaddxy(pandas, metDefs, Wswitch):
    if Wswitch =="false":
        keys = metDefs+["l1","l2","genbosonpt"]
    else:
         keys = metDefs+["l1","genbosonpt"]   
    for key in keys:
        if key=="l1":
            columnpT = "pt_1"
            columnphi = "phi_1"
            columnx = "Px_1"
            columny = "Py_1"
        elif key=="l2":
            columnpT = "pt_2"
            columnphi = "phi_2"
            columnx = "Px_2"
            columny = "Py_2"
        elif key=="genbosonpt":
            columnpT = "genbosonpt"
            columnphi = "genbosonphi"
            columnx = "genbosonpx"
            columny = "genbosonpy"
        else:
            columnpT = "met"+key+"Pt"
            columnphi = "met"+key+"Phi"
            columnx = "met"+key+"Px"
            columny = "met"+key+"Py"
        valx, valy = pol2kar(pandas[columnpT], pandas[columnphi])
        if key not in ["l1","l2","genbosonpt"]:
            pandas[columnx] = pd.Series(-valx, index=pandas.index)
            pandas[columny] = pd.Series(-valy, index=pandas.index)
        else:
            pandas[columnx] = pd.Series(valx, index=pandas.index)
            pandas[columny] = pd.Series(valy, index=pandas.index)
    return pandas

def getFctParameters(x_data, y_data):
    x_data = x_data.astype(np.float64)
    y_data = y_data.astype(np.float64)

    param_bounds=([0,0,0,0],[np.inf,np.inf,np.inf,np.inf])
    p0=[ 1446.4, 5.1, 4.8, 0.91]
    #p0=[500000, 2, 2, 2]  #BU damit hat es funktioniert
    params, params_covariance = optimize.curve_fit(Function, x_data, y_data, p0=p0)
    return(params)

def Function(x, a, b, c, n):
    return a * x * b**(b*n) * np.divide(np.exp(-n**2/2) , (b-n + np.divide(x,c))**(b*n) )

def getParametrization(Boson_Pt):
    n, _ = np.histogram(Boson_Pt, bins=18, normed=True)
    Parameters = getFctParameters((_[:-1]+_[1:])/2,n)
    Population = Function(Boson_Pt, Parameters[0], Parameters[1], Parameters[2], Parameters[3])  
    return Population

def getrecoil(pandas):
    RecoilMetsKeys = ["met", "metPuppi", "metNoPU", "metPUCorrected", "metTrack"]
    dfRecoil = pd.DataFrame()
    for key in RecoilMetsKeys:
        #recoil defininition
        dfRecoil[key+"Px"] = pd.Series(pandas[key+"Px"]-pandas["Px_1"]-pandas["Px_2"], index=pandas.index)
        dfRecoil[key+"Py"] = pd.Series(pandas[key+"Py"]-pandas["Py_1"]-pandas["Py_2"], index=pandas.index)
        dfRecoil[key+"SumEt"] = pd.Series(pandas[key+"SumEt"] - np.sqrt(np.square(pandas["Px_1"])+np.square(pandas["Py_1"])) - np.sqrt(np.square(pandas["Px_2"])+np.square(pandas["Py_2"])), index=pandas.index)
    #collect remaining columns where no changes are made
    dfRecoil["Px_1"] = pd.Series(pandas["Px_1"], index=pandas.index)
    dfRecoil["Boson_Pt"] = pd.Series(pandas["genbosonpt"], index=pandas.index)
    dfRecoil["Boson_Phi"] = pd.Series(pandas["genbosonphi"], index=pandas.index)
    dfRecoil["genbosonpx"] = pd.Series(pandas["genbosonpx"], index=pandas.index)
    dfRecoil["genbosonpy"] = pd.Series(pandas["genbosonpy"], index=pandas.index)
    dfRecoil["Px_2"] = pd.Series(pandas["Px_2"], index=pandas.index)
    dfRecoil["Py_2"] = pd.Series(pandas["Py_2"], index=pandas.index)
    dfRecoil["Py_1"] = pd.Series(pandas["Py_1"], index=pandas.index)
    dfRecoil["npv"] = pd.Series(pandas["npv"], index=pandas.index)
    dfRecoil["metPUPx"] = pd.Series(pandas["metPUPx"], index=pandas.index)
    dfRecoil["metPUPy"] = pd.Series(pandas["metPUPy"], index=pandas.index)
    dfRecoil["metPUSumEt"] = pd.Series(pandas["metPUSumEt"], index=pandas.index)
    return dfRecoil



def getrecoilW(pandas):
    RecoilMetsKeys = ["met", "metPuppi", "metNoPU", "metPUCorrected", "metTrack"]
    dfRecoil = pd.DataFrame()
    for key in RecoilMetsKeys:
        #recoil defininition
        dfRecoil[key+"Px"] = pd.Series(pandas[key+"Px"]-pandas["Px_1"], index=pandas.index)
        dfRecoil[key+"Py"] = pd.Series(pandas[key+"Py"]-pandas["Py_1"], index=pandas.index)
        dfRecoil[key+"SumEt"] = pd.Series(pandas[key+"SumEt"] - np.sqrt(np.square(pandas["Px_1"])+np.square(pandas["Py_1"])), index=pandas.index)
    #collect remaining columns where no changes are made
    dfRecoil["Px_1"] = pd.Series(pandas["Px_1"], index=pandas.index)
    dfRecoil["Boson_Pt"] = pd.Series(pandas["genbosonpt"], index=pandas.index)
    dfRecoil["Boson_Phi"] = pd.Series(pandas["genbosonphi"], index=pandas.index)
    dfRecoil["genbosonpx"] = pd.Series(pandas["genbosonpx"], index=pandas.index)
    dfRecoil["genbosonpy"] = pd.Series(pandas["genbosonpy"], index=pandas.index)
    dfRecoil["genbosonmasst"] = pd.Series(pandas["genbosonmasst"], index=pandas.index)
    dfRecoil["Py_1"] = pd.Series(pandas["Py_1"], index=pandas.index)
    dfRecoil["npv"] = pd.Series(pandas["npv"], index=pandas.index)
    dfRecoil["metPUPx"] = pd.Series(pandas["metPUPx"], index=pandas.index)
    dfRecoil["metPUPy"] = pd.Series(pandas["metPUPy"], index=pandas.index)
    dfRecoil["metPUSumEt"] = pd.Series(pandas["metPUSumEt"], index=pandas.index)
    return dfRecoil


def writehdf5(DataF, dset, plotsD):
    IdxpTCut = (DataF['Boson_Pt']>pTMin) & (DataF['Boson_Pt']<=pTMax) & (DataF['npv']<=VertexMax) & (DataF['npv']>VertexMin)

    #Get Population of respective pT bin und inverse it to get probability
    Population = getParametrization(DataF['Boson_Pt'][IdxpTCut])
    weights = np.divide(1.0, Population)

    set_BosonPt = dset.create_dataset("Boson_Pt",  dtype='f',
        data=[ DataF['Boson_Pt'][IdxpTCut]])

    dset_PF = dset.create_dataset("PF",  dtype='f',
        data=[DataF['metPx'][IdxpTCut], DataF['metPy'][IdxpTCut],
        DataF['metSumEt'][IdxpTCut]])

    dset_Track = dset.create_dataset("Track",  dtype='f',
        data=[DataF['metTrackPx'][IdxpTCut], DataF['metTrackPy'][IdxpTCut],
        DataF['metTrackSumEt'][IdxpTCut]])

    dset_NoPU = dset.create_dataset("NoPU",  dtype='f',
        data=[DataF['metNoPUPx'][IdxpTCut], DataF['metNoPUPy'][IdxpTCut],
        DataF['metNoPUSumEt'][IdxpTCut]])

    dset_PUCorrected = dset.create_dataset("PUCorrected",  dtype='f',
        data=[DataF['metPUCorrectedPx'][IdxpTCut], DataF['metPUCorrectedPy'][IdxpTCut],
        DataF['metPUCorrectedSumEt'][IdxpTCut]])

    dset_PU = dset.create_dataset("PU",  dtype='f',
        data=[DataF['metPUPx'][IdxpTCut], DataF['metPUPy'][IdxpTCut],
        DataF['metPUSumEt'][IdxpTCut]])

    dset_Puppi = dset.create_dataset("Puppi",  dtype='f',
        data=[DataF['metPuppiPx'][IdxpTCut], DataF['metPuppiPy'][IdxpTCut],
        DataF['metPuppiSumEt'][IdxpTCut]])

    dset_NoPV = dset.create_dataset("NVertex",  dtype='f',data=[DataF['npv'][IdxpTCut]] )
    
    if Wswitch=="true":
        dset_tmass = dset.create_dataset("genBosonTransMass",  dtype='f',data=[DataF['genbosonmasst'][IdxpTCut]] )    
    else: 
        dset_Target = dset.create_dataset("Lepton2",  dtype='f',
                data=[DataF["Px_2"][IdxpTCut],
                DataF["Py_2"][IdxpTCut]])
                
    dset_Target = dset.create_dataset("Lepton1",  dtype='f',
            data=[DataF["Px_1"][IdxpTCut],
            DataF["Py_1"][IdxpTCut]])

    dset_Target = dset.create_dataset("Target",  dtype='f',
            data=[-DataF["genbosonpx"][IdxpTCut],
            -DataF["genbosonpy"][IdxpTCut]])

    dset_Weight = dset.create_dataset("weights",  dtype='f',
            data=[weights])
    dset.close()


def loadRoot(rootFile,ll,filesD, Wswitch, metDefs):
  treename = ll+"_nominal/ntuple"

  #collect NN input and target data
  variables = ["Pt","Phi","SumEt"]
  branches = []
  for defs in metDefs:
      for var in variables:
          branches = branches + ["met%s%s"%(defs,var)]
  if Wswitch=="true":
      branchesadd = ["npv", "pt_1", "phi_1", "genbosonpt", "genbosonphi", "genbosonmasst"]
  else: 
      branchesadd = ["npv", "pt_1", "phi_1", "pt_2","phi_2", "genbosonpt", "genbosonphi"]
  branches = branches+branchesadd
  array = rnp.root2array(rootFile, treename=treename, branches=branches)
  pandas = pd.DataFrame.from_records(array.view(np.recarray))
  return pandas

def getInputs(rootFile,ll,filesD, plotsD, Wswitch):
  treename = ll+"_nominal/ntuple"

  #collect NN input and target data
  metDefs = ["", "Puppi","NoPU","PU","PUCorrected","Track"]
  pandas = loadRoot(rootFile,ll,filesD, Wswitch, metDefs)

  pandasxy = pd.DataFrame()
  pandasrecoil = pd.DataFrame()
  pandasxy = pandasaddxy(pandas, metDefs, Wswitch)
  if Wswitch == "false":
      pandasrecoil = getrecoil(pandas)
  elif Wswitch == "true":
      pandasrecoil = getrecoilW(pandas)
  else:
      print("Error, there was no W switch provided. Please select W switch in shell script.")
      sys.exit()

  #Write inputs in hdf5 file for NN
  writeInputs_apply = h5py.File("%sNN_Input_apply_%s.h5"%(filesD,ll), "w")
  writehdf5(pandasrecoil, writeInputs_apply, plotsD)



if __name__ == "__main__":
    ll = sys.argv[1]
    rootFile = sys.argv[2]
    filesD = sys.argv[3]
    plotsD = sys.argv[4]
    Wswitch = sys.argv[5]
    getInputs(rootFile,ll,filesD, plotsD, Wswitch)
