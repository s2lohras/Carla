from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os

class SpeedLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SpeedLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env.envs[0]
            
            v = env.vehicle.get_velocity()
            speed_kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))

            # Log the speed
            self.logger.record('train/00_speed_kmh', speed_kmh)
            self.logger.dump(self.n_calls)

        return True


class DeviationLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(DeviationLoggerCallback, self).__init__(verbose)
        self.episode_deviations = []

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        # Get the current deviation
        current_deviation = env.distance_from_center  
        self.episode_deviations.append(current_deviation)

        # Check if an episode has ended
        done = self.locals.get('done')
        if done: #'episode' in self.locals:
            mean_deviation = np.mean(self.episode_deviations)
            
            # Log the mean deviation
            self.logger.record('train/00_mean_episode_deviation', mean_deviation)
            self.logger.dump(self.n_calls)
            
            # Reset the episode deviations
            self.episode_deviations = []

        return True
    

class CollisionLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CollisionLoggerCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        done = self.locals.get('done')
        if done:      
            collision_rate = env.num_collision / env.num_episodes
            self.logger.record('train/00_collision_rate', collision_rate)
            self.logger.dump(self.n_calls)
        
        return True


