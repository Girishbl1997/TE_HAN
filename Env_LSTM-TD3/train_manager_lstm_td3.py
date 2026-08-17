import os
import time
import subprocess
import sys
import signal

CARLA_EXE_PATH = r"C:/sim/CarlaUE4.exe"

TOTAL_TIMESTEPS = 1_000_000
#SEEDS = [1, 2, 3]

CARLA_SH = os.environ.get("CARLA_ROOT","/home/carla") + "/CarlaUE4.sh" 
PORT = int(os.environ.get("CARLA_PORT", "2010"))

def kill_carla(port=2010):
    print("Clearing memory: Force closing CARLA server...")
    # os.system("taskkill /F /IM CarlaUE4.exe /T 2>NUL")
    # os.system("taskkill /F /IM CarlaServer.exe /T 2>NUL")
    os.system(f"pkill -9 -f 'carla-rpc-port={port}'")  #for linux
    time.sleep(3)

def start_carla():
    print(f"Booting CARLA server on port {PORT}...", flush=True)
    logpath = os.path.join(os.environ.get("WS","."), "logs",f"carla_{PORT}.log")
    logf = open(logpath, "w")
    # subprocess.Popen([CARLA_EXE_PATH, "-quality-level=Low", "-RenderOffScreen" ]) 
    subprocess.Popen([CARLA_SH, "-RenderOffScreen", "-nosound", f"-carla-rpc-port={PORT}"],
                      env={**os.environ, "SDL_VIDEODRIVER":"offscreen"}, stdout=logf, stderr=subprocess.STDOUT)  # for linux
    time.sleep(40) 

def run_training(SEED):

    print(f"Launching training | seed {SEED } |")
    try:
        result = subprocess.run(["python", "train_lstm_td3.py", str(SEED)])
        if result.returncode != 0:
            print(f"Training script encountered an error:{result.returncode}")
            kill_carla(); sys.exit(1)

    except KeyboardInterrupt:
        print("Training interrupted by user. Shutting down CARLA server.")
        kill_carla(); sys.exit(0)

if __name__ == "__main__":
    
        print(f"Starting Pipeline: seeds =  ")
        signal.signal(signal.SIGTERM, lambda*_: (kill_carla(), sys.exit(0)))
    
    #for SEED in SEEDS:
        SEED = int(sys.argv[1])
        #kill_carla()
        #print(f"[seed {SEED} ] ")       
        start_carla()
        run_training(SEED)
        kill_carla()
        
        #print(f"\n ALL SEEDS COMPLETE. {TOTAL_TIMESTEPS} timesteps/seed x {len(SEEDS)} seeds.")