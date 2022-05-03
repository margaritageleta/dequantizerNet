#!/bin/bash
source $HOME/dequantizerNet/environment.sh
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            experiment) experiment=${VALUE} ;;     
            *)   
    esac    
done

if [[ -z $experiment ]]; then echo "Missing experiment number."; exit 1; fi

echo "Executing experiment #$experiment"
rm -rf $LOG_PATH/exp$experiment
mkdir -p $LOG_PATH/exp$experiment
cp $ROOT_PATH/params.yaml $LOG_PATH/exp$experiment/
touch $LOG_PATH/exp$experiment/exp$experiment.log
chmod +rwx $LOG_PATH/exp$experiment/exp$experiment.log

sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=DQNET$experiment
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=100G
#SBATCH --time=1-00:00
#SBATCH -o $LOG_PATH/exp$experiment/exp$experiment.log
#SBATCH -e $LOG_PATH/exp$experiment/exp$experiment.err
ml load python/3.6.1
#source env/bin/activate
ml load cuda
ml load py-pytorch/1.4.0_py36
ml load py-numpy/1.19.2_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load py-h5py/3.1.0_py36
ml load py-scikit-learn/0.24.2_py36
# -C GPU_MEM:16GB
if python3 $ROOT_PATH/src/trainer.py \
--params $LOG_PATH/exp$experiment/params.yaml \
--num $experiment;
then echo "Success!"
else echo "Fail!"; fi
EOT

sleep .5
echo "LOG DIR: $LOG_PATH/exp$experiment/exp$experiment.log"
