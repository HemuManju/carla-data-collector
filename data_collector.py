import random
import os
from tqdm import tqdm
import itertools

import json

import ray
import multiprocessing

from core.carla_core import kill_all_servers
from carla_server import CarlaServer
from core.helper import inspect

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

from pre_process import PreProcessData
from data_writer import WebDatasetWriter

from utils import create_directory


def save_configuration(config, client, write_path):
    # Get file configuration
    client_config = inspect(client)

    # Add the configuration
    for key in config:
        if key not in ['experiment']:
            client_config[key + '_config'] = config[key]

    # Save the configuration
    file_name = config['experiment']['town']
    save_path = write_path + file_name + '_configuration.json'
    with open(save_path, 'w') as fp:
        json.dump(client_config, fp, indent=4)
    fp.close()
    return None


class AgentManager():

    def __init__(self, config, server):
        self.cfg = config
        self.server = server
        self.world = None
        self.steps = 0
        return None

    def setup_agent(self, behavior=None):
        # Get the vehicle and set the type
        world = self.server.get_world()
        hero = self.server.get_hero()

        # Set the behavior type
        if behavior is None:
            agent = BasicAgent(
                hero, target_speed=self.cfg['vehicle']['target_speed'])
        else:
            agent = BehaviorAgent(hero, behavior=behavior)

        # Set the agent destination
        spawn_points = world.get_map().get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)
        return agent, spawn_points

    def reset_agent(self):
        self.server.reset()
        agent, spawn_points = self.setup_agent()
        return agent, spawn_points

    def collect_data(self, agent, pre_process=None):
        control = agent.run_step()

        # Get different kinds of data
        vehicle_data = agent.get_vehicle_data(control)
        traffic_data = agent.get_traffic_data()
        waypoint_data = agent.get_waypoint_data()
        sensor_data = self.server.step(control)

        if pre_process is not None:
            data = pre_process.process(sensor_data,
                                       waypoint_data,
                                       vehicle_data=vehicle_data,
                                       traffic_data=traffic_data)
        else:
            data = {
                **sensor_data,
                **waypoint_data,
                **traffic_data,
                **vehicle_data
            }
        return data


class DataCollector():

    def __init__(self, config, write_path):
        self.cfg = config
        self.write_path = write_path

        # Setup carla path and server
        os.environ["CARLA_ROOT"] = "/home/hemanth/Carla/CARLA_0.9.11"
        self.server = CarlaServer(config=self.cfg)

        # Setup agent, writer and preprocess
        self.agent_manager = AgentManager(config=self.cfg, server=self.server)
        self.pre_process = PreProcessData(config=self.cfg)
        self.writer = WebDatasetWriter(config=self.cfg)

        # Create a directory and save the configuration
        create_directory(write_path)

        # Save the configuration
        client = self.agent_manager.server.get_client()
        save_configuration(self.cfg, client, self.write_path)

        return None

    def write_loop(self, file_name, agent, spawn_points):
        # Create the tar file
        self.writer.create_tar_file(file_name, self.write_path)

        steps = self.cfg['data_writer']['steps']
        for i in range(steps):

            # Collect the data from agent
            data = self.agent_manager.collect_data(agent, self.pre_process)

            # Write data at regular intervals
            if i % self.cfg['data_writer']['data_write_freq'] == 0:
                self.writer.write(data, i)

            # Change the destination if done
            if agent.done():
                agent.set_destination(random.choice(spawn_points).location)

            # Reset if collision has happened
            if data['collision']:
                agent, spawn_points = self.agent_manager.reset_agent()
                agent.set_destination(random.choice(spawn_points).location)
        return None

    def collect(self):
        """
        Main loop of the simulation. It handles updating all the HUD information,
        ticking the agent.
        """
        try:
            # Iterate over weather and behavior
            combinations = list(
                itertools.product(self.cfg['experiment']['weather'],
                                  self.cfg['vehicle']['behavior']))
            for weather, behavior in tqdm(combinations):
                self.server.set_weather(weather)
                agent, spawn_points = self.agent_manager.setup_agent(behavior)

                # Get the new file name
                file_name = '_'.join(
                    [self.cfg['experiment']['town'], weather, behavior])

                # Run the simulation
                self.write_loop(file_name, agent, spawn_points)

            # Finally close the writer
            self.writer.close()

        except KeyboardInterrupt:
            self.writer.close()
            kill_all_servers()
            print('Data collection interrupted')

        finally:
            print('Finished data collection')
            kill_all_servers()


class ParallelDataCollector():

    def __init__(self, config, write_path, number_collectors=1):
        self.cfg = config
        self.write_path = write_path
        self.number_collector = number_collectors
        return None

    def single_instance_collector(self, weather, behavior):

        # Set the weather and agent behavior
        data_collector = DataCollector(self.cfg, self.write_path)
        data_collector.server.set_weather(weather)
        agent, spawn_points = data_collector.agent_manager.setup_agent(
            behavior)

        # Get the new file name
        file_name = '_'.join(
            [self.cfg['experiment']['town'], weather, behavior])

        # Run the simulation
        data_collector.write_loop(file_name, agent, spawn_points)
        data_collector.writer.close()
        data_collector.server.close()
        print('_' * 32 + 'Process Done' + '_' * 32)
        return None

    def collect(self):
        try:
            # Iterate over weather and behavior
            with multiprocessing.Pool(processes=self.number_collector) as pool:
                pool.starmap(
                    self.single_instance_collector,
                    itertools.product(self.cfg['experiment']['weather'],
                                      self.cfg['vehicle']['behavior']))
        except KeyboardInterrupt:
            self.writer.close()
            kill_all_servers()
            print('Data collection interrupted')

        finally:
            print('Finished data collection')
            kill_all_servers()
