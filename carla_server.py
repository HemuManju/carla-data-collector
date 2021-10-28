import carla

from core.carla_core import CarlaCore


class CarlaServer():
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """
    def __init__(self, config):
        """Initializes the environment"""
        self.config = config

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.config['experiment'])
        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.config['vehicle'])

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        return sensor_data

    def set_weather(self, weather):
        # Choose the weather of the simulation
        weather = getattr(carla.WeatherParameters, weather)
        self.core.world.set_weather(weather)

    def change_town(self, town):
        self.core.world = self.core.client.load_world(
            map_name=town,
            reset_settings=False,
            map_layers=carla.MapLayer.All if
            self.config['carla']["enable_map_assets"] else carla.MapLayer.NONE)

    def step(self, control):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        sensor_data = self.core.tick(control)
        return sensor_data
