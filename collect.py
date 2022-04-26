import yaml

from core.carla_core import kill_all_servers

from modules.data_collector import DataCollector, ParallelDataCollector
from modules.data_reader import WebDatasetReader

from utils import skip_run

# Run the simulation
kill_all_servers()
config = yaml.load(open('experiment_config.yaml'), Loader=yaml.SafeLoader)


def main(config):
    if config['collector']['parallel_collect']:
        collector = ParallelDataCollector(
            config,
            write_path=config['collector']['data_write_path'],
            number_collectors=config['collector']['number_collectors'],
        )
    else:
        collector = DataCollector(
            config, write_path=config['collector']['data_write_path']
        )

    collector.collect()


if __name__ == '__main__':

    try:
        main(config)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
