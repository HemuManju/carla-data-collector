import math

from collections import deque

import numpy as np
from gym.spaces import Box, Discrete, Dict

from torchvision import transforms


try:
    import carla
except ModuleNotFoundError:
    pass

from .base_experiment import ContinuousBaseExperiment
from .utils import post_process_image


class FrontRGBContinuousExperiment(ContinuousBaseExperiment):
    def get_observation_space(self):
        if self.config["hero"]["sensors_process"]["gray_scale"]:
            num_of_channels = 1
        else:
            num_of_channels = 3
        obs_space = Dict(
            {
                "images": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.config['seq_length'],
                        1,
                        self.config["hero"]["sensors"]["rgb"]["image_size_x"],
                        self.config["hero"]["sensors"]["rgb"]["image_size_y"],
                    ),
                    dtype=np.float32,
                ),
                "command": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )
        return obs_space

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = hero.get_location().z < -0.5

        # Collision
        if self.n_collision > 3:
            self.collided = True

        return (
            self.done_time_idle
            or self.done_falling
            or self.done_time_episode
            or self.collided
        )

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False
        self.collided = False

        # hero variables
        self.last_location = None
        self.last_velocity = 0

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        # Collision and heading deviation
        self.n_collision = 0
        self.last_heading_deviation = 0

    def preprocess_sensor_data(self, sensor_data):
        # Check if there is a collision
        if "collision" in sensor_data.keys():
            self.n_collision += 1

    def get_observation(self, sensor_data, core=None):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """

        # Preprocess sensor data
        self.preprocess_sensor_data(sensor_data)
        image = post_process_image(
            sensor_data["rgb"],
            normalized=self.config["hero"]["sensors_process"]["normalized"],
            grayscale=self.config["hero"]["sensors_process"]["gray_scale"],
            rotation=-90,
        )

        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        # Create a temporary variable images stack
        images_stack = image

        if self.frame_stack >= 2:
            images_stack = np.concatenate([self.prev_image_0, images_stack], axis=2)
        if self.frame_stack >= 3 and images_stack is not None:
            images_stack = np.concatenate([self.prev_image_1, images_stack], axis=2)
        if self.frame_stack >= 4 and images_stack is not None:
            images_stack = np.concatenate([self.prev_image_2, images_stack], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image
        images_stack = np.swapaxes(images_stack, 0, 2)
        self.obs["images"] = images_stack

        return self.obs, {}

    def compute_reward(self, observation, core):
        """Computes the reward"""

        def compute_angle(u, v):
            return -math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])

        # Hero-related variables
        hero = core.hero
        map_ = core.map
        hero_location = hero.get_location()
        hero_speed = self.get_speed(hero)
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]

        # Negative reward for collision
        if self.n_collision > 1:
            r_collision = -1
        else:
            r_collision = 0

        # Reward for steering
        r_steer = -hero.get_control().steer ** 2

        # Negative reward for overspeed
        if hero_speed > self.config["constraints"]["desired_speed"]:
            r_fast = -1
        else:
            r_fast = 0

        # Reward for longitudinal speed and cost of lateral acceleration
        r_long_speed = hero_speed
        r_lat = -abs(hero.get_control().steer) * hero_speed**2

        # La duracion de estas infracciones deberia ser 2 segundos?
        # Penalize if not inside the lane
        r_out = 0
        closest_waypoint = map_.get_waypoint(
            hero_location, project_to_road=False, lane_type=carla.LaneType.Any
        )
        if (
            closest_waypoint is None
            or closest_waypoint.lane_type not in self.allowed_types
        ):
            r_out = -1
            self.last_heading_deviation = math.pi
        else:
            if not closest_waypoint.is_junction:
                # Direction waypoint heading
                wp_heading = closest_waypoint.transform.get_forward_vector()
                wp_heading = [wp_heading.x, wp_heading.y]
                angle = compute_angle(hero_heading, wp_heading)
                self.last_heading_deviation = abs(angle)

                # Update longitudinal speed
                r_long_speed = hero_speed * abs(math.cos(angle))

                if np.dot(hero_heading, wp_heading) < 0:
                    # We are going in the wrong direction
                    r_out = -1
                else:
                    if abs(math.sin(angle)) > 0.4:
                        if self.last_action is None:
                            self.last_action = carla.VehicleControl()

                        if self.last_action.steer * math.sin(angle) >= 0:
                            r_out = -1
            else:
                self.last_heading_deviation = 0

        reward = (
            200 * r_collision
            + 10 * r_fast
            + 5 * r_steer
            + r_out
            + r_long_speed
            + 0.2 * r_lat
            - 0.1
        )

        return reward
