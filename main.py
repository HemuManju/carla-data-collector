import yaml

from core.carla_core import kill_all_servers
from data_collector import DataCollector

# Run the simulation
kill_all_servers()
config = yaml.load(open('experiment_config.yaml'), Loader=yaml.SafeLoader)

collector = DataCollector(config, file_name='data', write_path='data/')
collector.collect()
