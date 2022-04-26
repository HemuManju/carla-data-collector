import yaml

from core.carla_core import kill_all_servers

from modules.data_collector import DataCollector, ParallelDataCollector
from modules.data_reader import WebDatasetReader

from utils import skip_run

# Run the simulation
kill_all_servers()
config = yaml.load(open('experiment_config.yaml'), Loader=yaml.SafeLoader)


def main(config):
    reader = WebDatasetReader(
        config=None, file_path=config['reader']['data_read_path'],
    )
    if config['reader']['create_movie']:
        reader.create_movie()


if __name__ == '__main__':

    try:
        main(config)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
