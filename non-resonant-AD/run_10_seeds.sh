#!/bin/bash
#SBATCH --job-name=SPP_NRAD_MI
#SBATCH --partition=tartarus
#SBATCH --output=SPP_NRAD_MI.txt
#SBATCH -e e_SPP_NRAD_MI.txt
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

export LD_LIBRARY_PATH=$HOME/Titan0/programs/usr/lib:$LD_LIBRARY_PATH
export PATH=$HOME/Titan0/programs/usr/bin:$PATH
export X11_X11_INCLUDE_PATH=$HOME/Titan0/programs/usr/include:$X11_X11_INCLUDE_PATH
export X11_X11_LIB=$HOME/Titan0/programs/usr/lib:$X11_X11_LIB

NRAD_PATH="/home/aegis/Titan0/NRAD/non-resonant-AD"

cd "$NRAD_PATH"
echo "Activate Environment"
source ../../ML_HEP/mlenv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No virtual environment is activated. Please activate your 'mlenv' environment."
    exit 1
fi

# Check if the virtual environment path contains "mlenv"
if [[ "$VIRTUAL_ENV" != *"mlenv"* ]]; then
    echo "Error: The activated virtual environment is not 'mlenv'. Current VIRTUAL_ENV: $VIRTUAL_ENV"
    exit 1
fi

echo "Virtual environment 'mlenv' is activated. Continuing..."

echo "Generate Base"
python gen_base.py

echo "Running 10 Seeds"
for i in {1..10}
do
	echo "Creating signal injection for Seed $i"
	python gen_siginj.py -g $i
	echo "Run Program for all signal injections"
	for sig_inj in 0 0.004 0.008 0.012 0.016 0.02 0.024
	do
		echo "Training Model"
		cd "Train_Models/"
		python context_weights.py -s $sig_inj -g $i
		python reweight.py -s $sig_inj -g $i
		python generate.py -s $sig_inj -g $i
		python morph.py -s $sig_inj -g $i
		cd ..
		echo "Generate Samples"
		cd "Gen_Sample/"
		python sample_context_weights.py -s $sig_inj -g $i
		python sample_reweight.py -s $sig_inj -g $i
		python sample_generate.py -s $sig_inj -g $i
		python sample_morph.py -s $sig_inj -g $i
		cd ..
		
		echo "Train CWoLa for SR"
		python CWoLa_SR.py -s $sig_inj -g $i
		
	done
	echo "Train CWoLa for CR (How well the background is created if signal = 0)"
	python CWoLa_CR.py -g $i
done


