#!/bin/bash
#SBATCH --partition=capella
#SBATCH --account=p_ml_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --array=1
#SBATCH --output=logs/board_%A_%a.out

WS=/data/horse/ws/giba824c-TE_AN
CODE=$WS/Env_LSTM-TD3
SIF=$CODE/carla_td3.sif

module load apptainer

export APPTAINER_BINDPATH="$WS"
mkdir -p $WS/logs

PORT=$((2000 + SLURM_ARRAY_TASK_ID * 10))
HOMEDIR=$WS/carla_home/task_${SLURM_ARRAY_TASK_ID}
mkdir -p $HOMEDIR
echo "Seed $SLURM_ARRAY_TASK_ID  PORT $PORT"

apptainer exec --nv --home $HOMEDIR \
  --env PYTHONPATH=$CODE \
  --env PROJECT_ROOT=$CODE \
  --env WSDIR=$WS \
  --env CARLA_PORT=$PORT \
  --env SEED=$SLURM_ARRAY_TASK_ID \
  --env SHOW_PREVIEW=0 \
  --env SDL_VIDEODRIVER=offscreen \
  --env PYTHONUNBUFFERED=1 \
  $SIF bash -c "cd $CODE && python train_manager_lstm_td3.py $SLURM_ARRAY_TASK_ID"

