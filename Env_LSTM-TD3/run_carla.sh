#!/bin/bash
#SBATCH --partition=capella
#SBATCH --account=p_ml_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=1-5
#SBATCH --output=logs/board_%A_%a.out

module --force purge
module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5 Python/3.11.3 CUDA/12.6.0
source ../.env/bin/activate

current_time=$(date +%Y%m%d%H%M%S)

for i in 1
do
    PORT=$((2000 + $i * $SLURM_ARRAY_TASK_ID * 10))
    SEED=$((i * $SLURM_ARRAY_TASK_ID))

    nohup apptainer exec --nv /data/cat/ws/giba824c-TE_AN/Env_LSTM-TD3/carla_0.9.15.sif /home/carla/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=$PORT > logs/runs/carla_run_${current_time}_$SEED.log 2>&1 &

    sleep 60

    nohup python train_lstm_td3.py $SEED $PORT &
done

sleep infinity