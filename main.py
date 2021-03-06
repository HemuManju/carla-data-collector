import yaml

from core.carla_core import kill_all_servers

from modules.data_collector import DataCollector, ParallelDataCollector
from modules.data_reader import WebDatasetReader

from utils import skip_run

# Run the simulation
kill_all_servers()
config = yaml.load(open('experiment_config.yaml'), Loader=yaml.SafeLoader)

with skip_run('skip', 'collect_data') as check, check():
    collector = DataCollector(config, write_path='../../../Desktop/data/')
    collector.collect()

with skip_run('skip', 'parallel_collect_data') as check, check():
    collector = ParallelDataCollector(
        config, write_path='../../../Desktop/data/', number_collectors=2
    )
    collector.collect()

with skip_run('skip', 'read_data') as check, check():
    reader = WebDatasetReader(
        config=None,
        file_path='/home/hemanth/Desktop/data/Town01_ClearNoon_normal_000000.tar',
    )
    reader.create_movie()

