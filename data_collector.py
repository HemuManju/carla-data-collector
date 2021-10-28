import random
import os
from tqdm import tqdm

from core.carla_core import kill_all_servers
from carla_server import CarlaServer

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

from post_process import PostProcess
from data_writer import WebWriter


class DataCollector():
    def __init__(self, config, file_name, write_path):
        self.cfg = config
        # Setup carla path
        os.environ["CARLA_ROOT"] = "/home/hemanth/Carla/CARLA_0.9.11"
        self.server = CarlaServer(self.cfg)
        self.post_process = PostProcess(config)
        self.writer = WebWriter(config=config,
                                file_name=file_name,
                                write_path=write_path)
        self.world = None
        self.steps = 0
        return None

    def collect(self):
        """
        Main loop of the simulation. It handles updating all the HUD information,
        ticking the agent and, if needed, the self.world.
        """
        try:
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

            for i in tqdm(range(self.cfg['simulation']['steps'])):
                control = agent.run_step()
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
                    self.writer.write(data, i)

                # # Weather Change
                # if i % 500 == 0:
                #     weather = random.choice(self.cfg['experiment']['weather'])
                #     self.server.set_weather(weather)

                if agent.done():
                    agent.set_destination(random.choice(spawn_points).location)

            # Close the writer
            self.writer.close()

        finally:
            print('Finished data collection')
            kill_all_servers()
