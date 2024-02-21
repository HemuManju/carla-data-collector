
import carla

from core.carla_core import CarlaCore


class CarlaServer:
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """

    def __init__(self, config):
        """Initializes the environment"""
        self.cfg = config

        self.core = CarlaCore(self.cfg['carla_server'])
        self.core.setup_experiment(self.cfg['experiment'])
        self.setup_client()
        self.reset()

    def setup_client(self):
        # Setup simulation
        self.core.client.set_timeout(2.0)
        sim_world = self.core.client.get_world()

        if self.cfg['simulation']['sync']:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            sim_world.apply_settings(settings)
        return None

    def close(self):
        # Finally kill the process
        self.core.kill_process()

    def get_client(self):
        return self.core.client

    def get_world(self):
        return self.core.client.get_world()

    def get_hero(self):
        return self.hero

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.cfg['vehicle'])

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        return sensor_data

    def set_weather(self, weather):
        # Choose the weather of the simulation
        weather = getattr(carla.WeatherParameters, weather)
        self.core.world.set_weather(weather)

    def change_town(self, town):
        # self.core.setup_experiment(self.cfg['experiment'], map_name=town)
        raise NotImplementedError

    def step(self, control):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        sensor_data = self.core.tick(control)
        return sensor_data
