import random
import os
from tqdm import tqdm

import json

from core.carla_core import kill_all_servers
from carla_server import CarlaServer
from core.helper import inspect

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

from post_process import PostProcess
from data_writer import WebDatasetWriter

from utils import create_directory


class DataCollector():

    def __init__(self, config):
        self.cfg = config

        # Setup carla path
        os.environ["CARLA_ROOT"] = "/home/hemanth/Carla/CARLA_0.9.11"
        self.server = CarlaServer(config=self.cfg)
        self.post_process = PostProcess(config=self.cfg)
        self.writer = WebDatasetWriter(config=self.cfg)
        self.world = None
        self.steps = 0

        return None

    def _setup_agent(self, client):
        # Get the vehicle and set the type
        self.world = client.get_world()
        hero = self.server.hero
        if self.cfg['vehicle']['agent'] == "Basic":
            agent = BasicAgent(
                hero, target_speed=self.cfg['vehicle']['target_speed'])
        else:
            agent = BehaviorAgent(hero,
                                  behavior=self.cfg['vehicle']['behavior'])

        # Set the agent destination
        spawn_points = self.world.get_map().get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

        return agent, spawn_points

    def _setup_client(self):
        if self.cfg['simulation']['seed']:
            random.seed(self.cfg['simulation']['seed'])

        # Setup simulation
        client = self.server.core.client
        client.set_timeout(2.0)
        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if self.cfg['simulation']['sync']:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
        return client

    def _run_simulation(self, agent, spawn_points, data_writer):
        for i in tqdm(range(self.cfg['simulation']['steps'])):
            control = agent.run_step()

            # Get different kinds of data
            vehicle_data = agent.get_vehicle_data(control)
            traffic_data = agent.get_traffic_data()
            waypoint_data = agent.get_waypoint_data()
            sensor_data = self.server.step(control)

            # Write data at regular intervals
            if i % self.cfg['simulation']['data_write_freq'] == 0:
                # Process the data
                data = self.post_process.process(sensor_data,
                                                 waypoint_data,
                                                 vehicle_data=vehicle_data,
                                                 traffic_data=traffic_data)
                data_writer.write(data, i)

            if agent.done():
                agent.set_destination(random.choice(spawn_points).location)

        return None

    def _save_configuration(self, client, file_name, write_path):

        # Get file configuration
        client_config = inspect(client)

        # Add carla configuration
        client_config['carla_config'] = self.cfg['carla']

        # Add vehicle configuration
        client_config['vehicle_config'] = self.cfg['vehicle']

        # Add simulation configuration
        client_config['simulation_config'] = self.cfg['simulation']

        # Save the configuration
        save_path = write_path + file_name + '_configuration.json'
        with open(save_path, 'w') as fp:
            json.dump(client_config, fp, indent=4)
        fp.close()

    def collect(self, file_name, write_path):
        """
        Main loop of the simulation. It handles updating all the HUD information,
        ticking the agent and, if needed, the self.world.
        """

        client = self._setup_client()
        agent, spawn_points = self._setup_agent(client)

        # Create a data directory
        create_directory(write_path)

        try:
            # Iterate over weather
            for weather in self.cfg['experiment']['weather']:
                self.server.set_weather(weather)

                # Get the new file name
                file_name = '_'.join([weather, self.cfg['experiment']['town']])

                # Save the configuration
                self._save_configuration(client, file_name, write_path)

                # Setup the writer
                self.writer._setup_data_directory(file_name, write_path)

                # Run the simulation
                self._run_simulation(agent, spawn_points, self.writer)

            # Finally close the writer
            self.writer.close()

        except KeyboardInterrupt:
            self.writer.close()
            kill_all_servers()
            print('Data collection interrupted')

        finally:
            print('Finished data collection')
            kill_all_servers()
