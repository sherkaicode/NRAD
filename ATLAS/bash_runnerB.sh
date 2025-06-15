#!/bin/bash
#SBATCH --job-name=Data16
#SBATCH --partition=tartarus
#SBATCH --output=Data16.txt
#SBATCH -e error_Data16.txt
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

export PATH=$HOME/Titan0/programs/usr/bin:$PATH
export X11_X11_INCLUDE_PATH=$HOME/Titan0/programs/usr/include:$X11_X11_INCLUDE_PATH
export X11_X11_LIB=$HOME/Titan0/programs/usr/lib:$X11_X11_LIB

source $HOME/Titan0/programs/bin/thisroot.sh

export PYTHONPATH=$HOME/Titan0/madgraph/MG5_aMC_v3_5_6/HEPTools/lhapdf6_py3/local/lib/python3.10/dist-packages:$PYTHONPATH


export PYTHIA8=$HOME/Titan0/Pythia/pythia8313
export DELPHES_PPATH=$HOME/Titan0/madgraph/MG5_aMC_v3_5_6/Delphes

export LD_LIBRARY_PATH=$HOME/Titan0/programs/usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHIA8/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/Titan0/madgraph/MG5_aMC_v3_5_6/HEPTools/lhapdf6_py3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DELPHES_PPATH

export ROOT_INCLUDE_PATH=$DELPHES_DIR:$DELPHES_PPATH/external:$ROOT_INCLUDE_PATH

cd $HOME/Titan0/NRAD/ATLAS
echo "Running Script"
python3 -u periodB_script.py >> periodB.log 2>&1
echo "Done Running Script"
