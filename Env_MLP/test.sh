module --force purge
module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5 Python/3.11.3 CUDA/12.6.0
source ../.env/bin/activate

current_time=$(date +%Y%m%d%H%M%S)

for i in 1
do
    PORT=$((2000 + $i * 10))
    SEED=$i

    nohup apptainer exec --nv /data/horse/ws/giba824c-TE_AN/Env_LSTM-TD3/carla_0.9.15.sif /home/carla/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=$PORT > logs/runs/carla_run_${current_time}_$i.log 2>&1 &

    sleep 60

    nohup python train_lstm_td3.py $SEED $PORT &
done

# To forcefully kill all CARLA processes after training
# kill $(ps -ax | grep '[c]arla' | awk '{print $1}')