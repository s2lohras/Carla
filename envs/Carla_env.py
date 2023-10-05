from xmlrpc.client import TRANSPORT_ERROR
import gymnasium as gym
from gymnasium import error, spaces, utils
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
import torch as T
import random
import time
import matplotlib.pyplot as plt

from MitM import apply_gaussian_noise


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# from parameters import *

# IM_WIDTH = 640
# IM_HEIGHT = 480
# IM_WIDTH = 160
# IM_HEIGHT = 120
IM_WIDTH = 320
IM_HEIGHT = 240
IM_CHANELS = 3

REPLAY_MEMORY_SIZE = 5_000
MINIBATCH_SIZE = 16
MIN_REWARD = -200
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
SHOW_PREVIEW = False
SECONDS_PER_EPISODE = 10
SHOW_CAM = SHOW_PREVIEW
STEER_AMT = 1.0
front_camera = None

SAVE_IMAGES = 0

DESTINATION = 50
# TOWN= None #"Town10"
# SPOWN_POINT=4
TOWN= "Town07"
SPOWN_POINT=38
# TOWN= "Town02"
# SPOWN_POINT=1
RANDOM_START_POINT = False

ATTACK = 0


class CarlaEnv(gym.Env):   


    def __init__(self):
        print("-------------------------------- Efficient initializing of Carla Environment!")

        self.SHOW_CAM = SHOW_CAM
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.client.load_world(TOWN)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        spectator = self.world.get_spectator()

        if TOWN=="Town02":
            spectator.set_transform(carla.Transform(carla.Location(x=0, y=250, z=100), carla.Rotation(pitch=-90, yaw=0, roll=0)))
        elif TOWN=="Town07":
            spectator.set_transform(carla.Transform(carla.Location(x=-50, y=-100, z=300), carla.Rotation(pitch=-90, yaw=0, roll=0)))
        else:
            spectator.set_transform(carla.Transform(carla.Location(x=-100, y=0, z=100), carla.Rotation(pitch=-90, yaw=0, roll=0)))

        
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        self.im_chanels = IM_CHANELS
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_height, self.im_width, IM_CHANELS), dtype=np.uint8)
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)  # two actions: Throttle & Steering
        # self.action_space = spaces.Box(low=0, high=1, shape=(1,1), dtype=np.float32)   # one action: Throttle

        self.num_collision = 0
        self.terminal = 0
        self.num_episodes = 0
        self.kmh = 0

        self.sensor_list = list()
        self.actor_list = list()

        
        if RANDOM_START_POINT:
            self.V_transform = random.choice(self.world.get_map().get_spawn_points())
        else:
            self.V_transform = self.map.get_spawn_points()[SPOWN_POINT]

                
        self.vehicle = self.world.spawn_actor(self.model_3, self.V_transform)
        self.actor_list.append(self.vehicle)
        print("--- Agent generated and transformed to this point: /n", self.V_transform)        

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.last_position = None
        self.total_distance_travelled = 0.0
        
        print("-------------------------------- End of initialization!")


    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)

        self.collision_hist = []
        self.terminal = 0
        self.distance_from_center = float(0.0)
        self.episode_start = time.time()

        # Set the vehicle back to its original spawn point or put it in a random point
        if RANDOM_START_POINT:
            self.V_transform = random.choice(self.world.get_map().get_spawn_points())
        else:
            self.V_transform = self.map.get_spawn_points()[SPOWN_POINT]
        self.vehicle.set_transform(self.V_transform)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.last_position = self.vehicle.get_location()
        self.total_distance_travelled = 0.0

        self.state = self.observation()

        return np.array(self.state), {}

    
    def step(self, action):


        if self.terminal:
            return self.reset()
        
        # self._iter_in_episod += 1
        reward_step = 0
        self.emgBr_flag = 0

        self.takeAction(action)

        v = self.vehicle.get_velocity()
        self.kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        self.terminal, self.terminalType = self.terminalCheck()

        current_position = self.vehicle.get_location()
        distance_this_step = self.distance_between_positions(current_position, self.last_position)
        self.total_distance_travelled += distance_this_step
        self.last_position = current_position
    


        reward = self.reward(self.terminal, self.terminalType, action)
        reward_step += reward

        if not self.terminal:
            self.state = self.observation()

        return np.array(self.state), reward_step, self.terminal, False, {}
       
         
    def observation(self):
        state = self.front_camera
        return state
    

    def reward(self, done, terminalType, action):
        # Reward 1
        reward = 0.0
      
        # Reward for progress: The faster the car goes (within limits), the higher the reward
        if 1 <= self.kmh <= 20:
            reward += self.kmh * 1  # Example: At 30 km/h, reward is +3.0 per time step
        else: # Speed too slow or too fast            
            reward -= 1 

        # Reward for staying in lane
        self.distance_from_center , wrong_lane = self.get_distance_to_centerline()
        reward -= self.distance_from_center
        if wrong_lane:
            reward -= 10


        # If end of episode (successfully or not)
        if done:
            if terminalType == "end of episode":
                reward += 100  # Bonus for completing the episode
            elif terminalType == "Too Long":
                reward -=-10
            elif terminalType == "Collision!":
                reward -= 100

        return reward        
  

    def terminalCheck(self):
        self.terminal = 0
        self.terminalType = None

        if len(self.collision_hist) != 0:
            self.terminal = 1
            self.terminalType = "Collision!"
            print(" - - - - - - - - - - Collision! - - - - - - - - - -")
            self.num_collision += 1
            self.num_episodes +=1

        
        if self.total_distance_travelled >= DESTINATION:
            self.terminal = 1
            self.terminalType = "end of episode"
            print(" - - - - - - - - - - Reached to the destination! - - - - - - - - - -")
            self.num_episodes +=1

        
        if (self.episode_start + SECONDS_PER_EPISODE < time.time()) and (self.kmh < 1):
            self.terminal = 1
            self.terminalType = "Too Long"
            print(" - - - - - - - - - - Too Long! - - - - - - - - - -")
            self.num_episodes +=1

        return self.terminal, self.terminalType

        
    def takeAction(self, action):
       
        throttle, steering = action
    
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steering = float(steering)
        
        self.vehicle.apply_control(control)


    def collision_data(self, event):
        self.collision_hist.append(event)

    
    def process_img(self, image):

        i = np.array(image.raw_data)      
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]

        if ATTACK: 
            perturbed_image = apply_gaussian_noise(i3)
            self.front_camera = perturbed_image

            if SAVE_IMAGES:

                save_folder = 'camera_images'
                os.makedirs(save_folder, exist_ok=True)
                filename = os.path.join(save_folder, f'image_{time.time()}.png')
                cv2.imwrite(filename, (i3).astype('uint8'))

                save_folder2 = 'perturbed_image'
                os.makedirs(save_folder, exist_ok=True)
                filename = os.path.join(save_folder2, f'image_{time.time()}.png')
                cv2.imwrite(filename, (perturbed_image).astype('uint8'))


        else:
            self.front_camera = i3

        # if self.SHOW_CAM:
        #     cv2.imshow("", i3)
        #     cv2.waitKeyEx(1)

    
    def get_distance_to_centerline(self):
        center_waypoints = self.map.generate_waypoints(2.0)
        vehicle_location = self.vehicle.get_transform().location
        vehicle_forward_vector = self.vehicle.get_transform().get_forward_vector()
        # vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        min_distance = float('inf')
        wrong_lane = 0
        closest_waypoint = None

        for waypoint in center_waypoints:
            # Draw waypoints as small blue spheres for visualization.
            # self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
            #                             color=carla.Color(r=0, g=0, b=255), life_time=0.1,
            #                             persistent_lines=True)

            distance = vehicle_location.distance(waypoint.transform.location)
            
            if distance < min_distance:
                min_distance = distance
                closest_waypoint = waypoint
                # closest_waypoint_location = waypoint.transform.location

        # Check if the vehicle is facing the wrong way
        waypoint_forward_vector = closest_waypoint.transform.get_forward_vector()
        if vehicle_forward_vector.dot(waypoint_forward_vector) < 0:
            wrong_lane = 1
            # min_distance *= 3  # Double the penalty. Adjust as needed.

        # Draw a line from the vehicle to the closest waypoint for visualization.
        # if closest_waypoint is not None:
        #     self.world.debug.draw_line(vehicle_location, closest_waypoint_location, thickness=0.2, color=carla.Color(255, 0, 0), life_time=0.3)

        # print(f"Distance to centerline: {min_distance} meters")
        return min_distance, wrong_lane

    
    def close_env(self):
        if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            print("xxxxxxxxxxxxxxxxxx - REMOVE EVERYTHING IN CLOSE ENV - xxxxxxxxxxxxxxxxxx")
    

    def distance_between_positions(self, loc1, loc2):
        # Assuming loc1 and loc2 are carla.Location objects
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        dz = loc1.z - loc2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)