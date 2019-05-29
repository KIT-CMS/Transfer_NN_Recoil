xy#!/bin/bash

LCG_RELEASE=93

if uname -a | grep ekpdeepthought
then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_${LCG_RELEASE}/x86_64-ubuntu1604-gcc54-dbg/setup.sh
    #git clone git://github.com/rootpy/root_numpy.git
    #cd root_numpy
    #python setup.py install
    #cd ..
    #pip uninstall tensorflow-gpu
    #pip install --user tensorflow-gpu==1.3.0
    #pip install --user python-pcl
    #pip install --user matplotlib
    #pip install --user root_numpy==4.7.3
    #brew install --HEAD root
    #pip install --user scipy==0.18
    #pip install --user --upgrade h5py
    #pip uninstall numpy root_numpy --user
    #pip install --upgrade --ignore-installed --no-cache-dir numpy root_numpy
    #pip install --upgrade --user numpy==1.10.1
    #pip install --user --no-cache root_numpy
    #pip install --user --upgrade pandas
    #pip install --user root_numpy --upgrade
    #pip install --user numpy==1.6.1
    #pip install root_numpy --upgrade
    #pip install --user python==2.7
    #pip install --user numpy==1.11.3 pandas==0.20.2 --no-binary=':all:' --verbose
    #pip install --user backports.functools_lru_cache
    export PATH=/usr/local/cuda-8.0/bin/:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES="2"
else
    source /cvmfs/sft.cern.ch/lcg/views/LCG_${LCG_RELEASE}/x86_64-slc6-gcc62-opt/setup.sh
fi
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/

