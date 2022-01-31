import yaml

from core.carla_core import kill_all_servers

from data_collector import DataCollector
from data_reader import WebDatasetReader
from post_process import Replay

from utils import skip_run

# Run the simulation
kill_all_servers()
config = yaml.load(open('experiment_config.yaml'), Loader=yaml.SafeLoader)

with skip_run('skip', 'collect_data') as check, check():
    collector = DataCollector(config)
    collector.collect(file_name='data', write_path='../../../Desktop/data/')

with skip_run('skip', 'read_data') as check, check():
    reader = WebDatasetReader(
        config=None,
        file_path='../../../Desktop/data/ClearNoon_Town05_Opt.tar')
    reader.create_movie()
