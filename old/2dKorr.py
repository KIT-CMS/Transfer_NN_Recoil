import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
import matplotlib.cm as cm
import h5py
import sys
import pandas as pd

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True) 
formatter.set_scientific(True) 
formatter.set_powerlimits((-2,2)) 
plt.ion()

colors_In = ['#ed2024', '#4800B7', '#2C6AA8', '#00A878', '#DE3E1E', '#FF8360', '#1BE7FF' , '#312F2F', '#000072']
colors_InOut = cm.brg(np.linspace(0, 1, 8))
 
def loadInputsTargets(file):
    InputsTargets = h5py.File("%s"%(file), "r")
    print("InputsTargets keys", InputsTargets.keys())

    Target =  InputsTargets['Target']
    return (InputsTargets, Target) 

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
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_fontsize(18) 
    ax.xaxis.offsetText.set_fontsize(18) 
    plt.grid(color='grey', linestyle='-', linewidth=0.4)

# Define a function to make the ellipses
def ellipse(ra,rb,ang,x0,y0,Nb=100):
    xpos,ypos=x0,y0
    radm,radn=ra,rb
    an=ang
    co,si=np.cos(an),np.sin(an)
    the=linspace(0,2*np.pi,Nb)
    X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
    Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos
    return X,Y
 
def compareInputs(plotDir, filesDir):
    #FileOld = "%sNN_Input_apply_xy_old.h5"%(filesDir)
    FileNew = "%sNN_Input_apply_mm.h5"%(filesDir)
    #InputsOld, TargetsOld = loadInputsTargets(FileOld)
    InputsOld, TargetsOld = loadInputsTargets(FileNew)
    nbinsHistBin = 50 

    # Define the x and y data 
    # For example just using random numbers
    x = InputsOld["PUCorrected"][0,:]
    y = InputsOld["Puppi"][0,:]
    
    print("Korrelationskoeffizient", np.corrcoef(x,y))
    
    # Set up default x and y limits
    xlims = [-50,50]
    ylims = [-50,50]

    # Set up your x and y labels
    xlabel = 'PV_NU_x in GeV'
    ylabel = 'PUPPI_x in GeV'

    # Define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02

    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left+0.045, bottom_h, 0.465, 0.25] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram

    # Set up the size of the figure
    fig = plt.figure(1, figsize=(9.5,9))

    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram


    #load Settings
    loadAxesSettings(axHistx)
    loadAxesSettings(axHisty)

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Find the min/max of the data
    xmin = -50
    xmax = +50
    ymin = -50
    ymax = +50

    # Make the 'main' temperature plot
    # Define the number of bins
    nxbins = 50
    nybins = 50
    nbins = 100

    xbins = linspace(start = xmin, stop = xmax, num = nxbins)
    ybins = linspace(start = ymin, stop = ymax, num = nybins)
    xcenter = (xbins[0:-1]+xbins[1:])/2.0
    ycenter = (ybins[0:-1]+ybins[1:])/2.0
    aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)

    H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins))
    X = xcenter
    Y = ycenter
    Z = H

    # Plot the temperature data
    cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],
           interpolation='nearest', origin='lower',aspect=aspectratio))

    # Plot the temperature plot contours
    contourcolor = 'white'
    xcenter = np.mean(x)
    ycenter = np.mean(y)
    ra = np.std(x)
    rb = np.std(y)
    ang = 0
    
    '''
    X,Y=ellipse(ra,rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",ms=1,linewidth=2.0)
    axTemperature.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points', horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25)

    X,Y=ellipse(2*ra,2*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor,ms=1,linewidth=2.0)
    axTemperature.annotate('$2\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)

    X,Y=ellipse(3*ra,3*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor, ms=1,linewidth=2.0)
    axTemperature.annotate('$3\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)
    '''
    #Plot the axes labels
    axTemperature.set_xlabel(xlabel,fontsize=22)
    axTemperature.set_ylabel(ylabel,fontsize=22)

    #Make the tickmarks pretty
    ticklabels = axTemperature.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    ticklabels = axTemperature.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    #Set up the plot limits
    axTemperature.set_xlim(xlims)
    axTemperature.set_ylim(ylims)


    #Plot the histograms
    axHistx.hist(x, bins=nxbins, range=xlims, label='PV_NU_x', histtype='step', linewidth=3.0, ec=colors_In[5], normed=True)
    axHisty.hist(y, bins=nybins, range=ylims, orientation='horizontal', label='PUPPI_x', histtype='step', linewidth=3.0, ec=colors_InOut[4], normed=True)

    #Set up the histogram limits
    axHistx.set_xlim( xlims )
    axHistx.set_ylabel('Density', fontsize=18)
    axHisty.set_xlabel('Density', fontsize=18)
    axHisty.xaxis.set_label_position("top")
    #axHisty.xaxis.set_tick_params(labeltop='on')
    axHisty.set_ylim( ylims )
    axHistx.grid(color='grey', linestyle='-', linewidth=0.4)

    #Make the tickmarks pretty
    ticklabels = axHistx.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    #Make the tickmarks pretty
    ticklabels = axHisty.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    #Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    axHistx.yaxis.set_major_locator(MaxNLocator(4))
    

    
    #Show the plot
    plt.draw()

    # Save to a File
    filename = 'myplot'
    plt.savefig("%s%s.png"%(plotDir,filename))
    plt.close()    



    # Define the x and y data 
    # For example just using random numbers
    x = InputsOld["PUCorrected"][0,:]
    y = InputsOld["Puppi"][0,:]
    
    print("Korrelationskoeffizient", np.corrcoef(x,y))
    
    # Set up default x and y limits
    xlims = [-50,50]
    ylims = [-50,50]

    # Set up your x and y labels
    xlabel = 'PV_NU_x in GeV'
    ylabel = 'PUPPI_x in GeV'

    # Define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02

    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left+0.045, bottom_h, 0.465, 0.25] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram

    # Set up the size of the figure
    fig = plt.figure(1, figsize=(9.5,9))

    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram


    #load Settings
    loadAxesSettings(axHistx)
    loadAxesSettings(axHisty)

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Find the min/max of the data
    xmin = -50
    xmax = +50
    ymin = -50
    ymax = +50

    # Make the 'main' temperature plot
    # Define the number of bins
    nxbins = 50
    nybins = 50
    nbins = 100

    xbins = linspace(start = xmin, stop = xmax, num = nxbins)
    ybins = linspace(start = ymin, stop = ymax, num = nybins)
    xcenter = (xbins[0:-1]+xbins[1:])/2.0
    ycenter = (ybins[0:-1]+ybins[1:])/2.0
    aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
    
    N1_1 = (-x+y+5>=0)
    N1_2 = (x-y+5>=0)
    Bedingung = N1_1 & N1_2
    H, xedges,yedges = np.histogram2d(y[Bedingung],x[Bedingung],bins=(ybins,xbins))
    # for x in xedges[:-1]:
    #     for y in yedged[:-1]:
    #        if  -x+y-20<0:
    #            index = nonzero(xedges == x)[0][0]
    #            H[index] = 0
    X = xcenter
    Y = ycenter
    Z = H
    r = np.corrcoef(x[Bedingung],y[Bedingung])
    print("Korrelationskoeffizient for relu (-x+y-20)", np.corrcoef(x[Bedingung],y[Bedingung]))
    # Plot the temperature data
    cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],
           interpolation='nearest', origin='lower',aspect=aspectratio))

    # Plot the temperature plot contours
    contourcolor = 'white'
    xcenter = np.mean(x)
    ycenter = np.mean(y)
    ra = np.std(x)
    rb = np.std(y)
    ang = 0

    #Plot the axes labels
    axTemperature.set_xlabel(xlabel,fontsize=22)
    axTemperature.set_ylabel(ylabel,fontsize=22)

    #Make the tickmarks pretty
    ticklabels = axTemperature.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    ticklabels = axTemperature.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    #Set up the plot limits
    axTemperature.set_xlim(xlims)
    axTemperature.set_ylim(ylims)
    

    #Plot the histograms
    axHistx.hist(x[Bedingung], bins=nxbins, range=xlims, label='PV_NU_x', histtype='step', linewidth=3.0, ec=colors_In[5], normed=True)
    axHisty.hist(y[Bedingung], bins=nybins, range=ylims, orientation='horizontal', label='PUPPI_x', histtype='step', linewidth=3.0, ec=colors_InOut[4], normed=True)

    #Set up the histogram limits
    axHistx.set_xlim( xlims )
    axHistx.set_ylabel('Density', fontsize=18)
    axHisty.set_xlabel('Density', fontsize=18)
    axHisty.xaxis.set_label_position("top")
    #axHisty.xaxis.set_tick_params(labeltop='on')
    axHisty.set_ylim( ylims )
    axHistx.grid(color='grey', linestyle='-', linewidth=0.4)

    #Make the tickmarks pretty
    ticklabels = axHistx.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    #Make the tickmarks pretty
    ticklabels = axHisty.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        #label.set_family('serif')

    #Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    axHistx.yaxis.set_major_locator(MaxNLocator(4))
    

    
    #Show the plot
    plt.draw()

    # Save to a File
    filename = 'Relu'
    plt.savefig("%s%s%s.png"%(plotDir,filename, r))
    plt.close()   

    
    
    import seaborn as sns
    from collections import OrderedDict
    fig=plt.figure(figsize=(14,8))
    fig.patch.set_facecolor('white')
    sns.set(font_scale = 1.5)
    MET_definitions = ['PF', 'ChargedPV', 'PV', 'PV_NU', 'PU', 'PUPPI']
    MET_definitions2 = ['PF', 'Track', 'NoPU', 'PUCorrected', 'PU', 'Puppi']
    Variables = ['x','y','SumPt']
    Variables = [Variables[:] for _ in range(6)]
    MET_definitions = np.repeat(MET_definitions,3)
    Variables = [item for sublist in Variables for item in sublist]
    Inputstring = [MET+'_'+Variable for MET,Variable in zip(MET_definitions,Variables)]
    Inputstring = np.append(Inputstring, 'NVertex')    
    x2derivates = pd.DataFrame()
    for i in range(len(MET_definitions2)):
        for j in range(len(['x','y','SumPt'])):
            print("i", i)
            print("j", j)
            x2derivates[Inputstring[i*3+j]] = InputsOld[MET_definitions2[i]][j,:]
    x2derivates['NVertex'] = np.transpose(InputsOld['NVertex'][:])
    print(x2derivates)
    x2derivates = x2derivates.reindex(sorted(x2derivates.columns), axis=1)
    mask = np.zeros_like(x2derivates.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask, k=+1)] = True
    with sns.axes_style("white"):
        sns_plot = sns.heatmap(x2derivates.corr(),  annot=True, mask=mask, square=False, fmt='.1f', cmap="RdYlGn", linewidths=.5, cbar_kws={"orientation": "vertical"})
        sns_plot.set_xticklabels(sns_plot.get_xticklabels(),rotation="vertical")
        sns_plot.set_yticklabels(sns_plot.get_yticklabels(),rotation="horizontal")
        plt.savefig("%sKorrMatrix.png"%(plotDir), bbox_inches="tight")
        plt.close()    
	
if __name__ == "__main__":
    plotsDir = sys.argv[1]
    filesDir = sys.argv[2]
    compareInputs(plotsDir,filesDir) 
