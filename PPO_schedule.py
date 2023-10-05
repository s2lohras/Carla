import gymnasium as gym
import envs
from parameters import *
from stable_baselines3 import PPO
import os
from callbacks import SpeedLoggerCallback, DeviationLoggerCallback, CollisionLoggerCallback
from stable_baselines3.common.callbacks import CallbackList

Learn_From_Scratch = 1

def ent_coef_schedule(epoch, total_epochs):
    start_ent_coef = 0.1
    end_ent_coef = 0.01
    progress = epoch / total_epochs
    return start_ent_coef - (start_ent_coef - end_ent_coef) * progress


if __name__ == '__main__':

    models_dir = "models/PPO"
    log_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    speed_callback = SpeedLoggerCallback(check_freq=100, log_dir=log_dir)
    deviation_callback = DeviationLoggerCallback()
    collision_callback = CollisionLoggerCallback()
    callbacks = CallbackList([speed_callback, deviation_callback, collision_callback])

    env = gym.make("CarlaEnv-v1") 
    print("----- ENV CREATED -----")

    if Learn_From_Scratch:
        model = PPO("CnnPolicy", env,
                    learning_rate=3e-4, 
                    clip_range=0.2, 
                    clip_range_vf=0.2, 
                    ent_coef=0.1,  # Initial value; we'll change it during training
                    verbose=1, tensorboard_log=log_dir)
        print("----- MODEL CREATED -----")
    else:
        load_models_dir = "models/PPO_Town07_sp38_50_CNN_2nd"
        model_path = f"{load_models_dir}/390000.zip"
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
        print("----- MODEL LOADED -----")

    TIMESTEPS = 10000
    TOTAL_EPOCHS = 100

    for i in range(1, TOTAL_EPOCHS + 1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callbacks)
        
        # Update the entropy coefficient after each epoch
        new_ent_coef = ent_coef_schedule(i, TOTAL_EPOCHS)
        model.ent_coef = new_ent_coef

        model.save(f"{models_dir}/{TIMESTEPS*i}")
        print("---------------- model saved ----------------")

    print("---------------- end of training ----------------")
    env.close_env()
