#!/bin/bash
source /storage/b/tkopf/jdl_maerz/setup_maerz.sh
training=true
WSample=false
lepton=mm
trainingsname="training${training}_WSample${WSample}_Channel${lepton}_PlotSummer17Thesis"
loss_fct="relResponseAsypTPVRange"
files_di=/storage/b/tkopf/NTupleApplication/files/
plots_di=/storage/b/tkopf/NTupleApplication/plots/
#WJets mit Boson Transverse Mass
#mergedFile=/storage/b/tkopf/mergedAnalysisFiles/MVAMet_genMt_2_27_11_2018/WJetsToLNu_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_madgraph-pythia8_v2/WJetsToLNu_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_madgraph-pythia8_v2.root
#WJets
#mergedFile=/ceph/akhmet/merged_files_from_naf/MVAMet_23_11_2018/WJetsToLNu_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_madgraph-pythia8_v2/WJetsToLNu_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_madgraph-pythia8_v2.root
#WJets1tauausJet
#mergedFile=/storage/b/tkopf/mergedAnalysisFiles/MVAMet_23_11_2018/WJetsToLNu_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_madgraph-pythia8_ext1-v2/WJetsToLNu_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_madgraph-pythia8_ext1-v2.root 
#DY
mergedFile=/storage/b/akhmet/merged_files_from_naf/MVAMet_09_11_2018/DYJetsToLLM50_RunIIFall17MiniAODv2_PU2017RECOSIMstep_13TeV_MINIAOD_madgraph-pythia8_ext1-v1/DYJetsToLLM50_RunIIFall17MiniAODv2_PU2017RECOSIMstep_13TeV_MINIAOD_madgraph-pythia8_ext1-v1.root


if [ ! -d "$files_di${trainingsname}_$lepton/" ]; then
	mkdir $files_di${trainingsname}_${lepton}/
	mkdir $files_di${trainingsname}_${lepton}/CV/
fi
files_directory=$files_di${trainingsname}_${lepton}/
cp NN_Input_apply_xy_old.h5 $files_directory
#cp NN_Input_apply_xy_old_npv24.h5 $files_directory
#cp NN_Output_applied_xy_old.h5 $files_directory
#cp preprocessing_input.pickle $files_directory
#cp checkpoint $files_directory


if [ ! -d "$plots_di${trainingsname}_${lepton}/" ]; then
	mkdir $plots_di${trainingsname}_${lepton}/
	mkdir $plots_di${trainingsname}_${lepton}/derivates/
	mkdir $plots_di${trainingsname}_${lepton}/Thesis/
fi
plots_directory=$plots_di${trainingsname}_${lepton}/

#python getNNinputs.py $lepton $mergedFile $files_directory $plots_directory $WSample
#python comparePhaseSpace.py $plots_directory $files_directory 
#python Plot_Inputs_Summer17.py $plots_directory $files_directory 
#python 2dKorr.py $plots_directory $files_directory 
if [ "$training"  = true ]; then
	#python Training.py $files_directory $loss_fct $lepton $plots_directory
	python Training_CV.py $files_directory $loss_fct $lepton $plots_directory
fi
#python applyNNmodel.py $lepton $files_directory $training $loss_fct
#python compareOutput.py $plots_directory $files_directory
#python prepareOutputFile.py $files_directory $lepton $plots_directory
#python Plot.py $mergedFile $files_directory $plots_directory $lepton $WSample

cp *.py $plots_directory
cp Start*.sh $plots_directory
cp -r $plots_directory /ekpwww/web/tkopf/public_html/NTupleApplication/
cp /portal/ekpbms1/home/tkopf/index.php $plots_directory
cp /portal/ekpbms1/home/tkopf/index.php /ekpwww/web/tkopf/public_html/NTupleApplication/${trainingsname}_${lepton}/
cp /portal/ekpbms1/home/tkopf/index.php /ekpwww/web/tkopf/public_html/NTupleApplication/${trainingsname}_${lepton}/Thesis/
cp /portal/ekpbms1/home/tkopf/index.php /ekpwww/web/tkopf/public_html/NTupleApplication/${trainingsname}_${lepton}/derivates/
