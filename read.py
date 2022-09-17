import yaml

from core.carla_core import kill_all_servers

from modules.data_reader import WebDatasetReader

from utils import find_tar_files

# Run the simulation
kill_all_servers()
config = yaml.load(open('experiment_config.yaml'), Loader=yaml.SafeLoader)


def main(config, file_path=None):
    if file_path is None:
        file_path = config['reader']['data_read_path']
    reader = WebDatasetReader(config=None, file_path=file_path)
    if config['reader']['create_movie']:
        reader.create_movie()


if __name__ == '__main__':

    try:
        paths = find_tar_files(config['reader']['data_read_path'], pattern='')
        main(config, file_path=paths[0])
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
