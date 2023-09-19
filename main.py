import yaml
import torch
import numpy as np
from torchvision.transforms import Resize

import matplotlib.pyplot as plt

from core.carla_core import kill_all_servers

from modules.data_collector import DataCollector, ParallelDataCollector
from modules.data_reader import WebDatasetReader, Summary
from modules.data_writer import AutoEncoderDatasetWriter

from utils import skip_run, find_tar_files, show_image, labels_to_cityscapes_palette


# Run the simulation
kill_all_servers()
config = yaml.load(open('experiment_config.yaml'), Loader=yaml.SafeLoader)

with skip_run('skip', 'collect_data') as check, check():
    writer = AutoEncoderDatasetWriter(config)
    collector = DataCollector(
        config,
        write_path='/home/hemanth/Desktop/carla-data-test/',
        navigation_type='one_curve',
        writer=writer,
    )
    collector.collect(single_scenario=True)

with skip_run('run', 'collect_data') as check, check():
    config['experiment']['town'] = 'Town01'
    collector = DataCollector(
        config,
        write_path='/home/hemanth/Desktop/carla-data-test/',
        navigation_type='straight',
    )
    collector.collect(single_scenario=True)


with skip_run('skip', 'parallel_collect_data') as check, check():
    collector = ParallelDataCollector(
        config, write_path='../../../Desktop/data/', number_collectors=2
    )
    collector.collect()

with skip_run('skip', 'read_data') as check, check():
    reader = WebDatasetReader(
        config=None,
        file_path='/home/hemanth/Desktop/data/one_curve/Town01_HardRainSunset_cautious_000000.tar',
    )
    dataset = reader.get_dataset()

    for data in dataset:
        print(data['json']['gnss']['values'])
        print(data['json']['velocity'])
        print(data['front.jpeg'])
        break
        test = np.array(data['json']['semseg']).reshape(256, 256)
        t = labels_to_cityscapes_palette(test)
        t = np.transpose(t, (2, 1, 0))
        t = torch.from_numpy(t)
        t = Resize(size=(128, 128))(t)
        plt.imshow(np.transpose(t, (1, 2, 0)), origin='lower')
        plt.pause(0.01)

with skip_run('skip', 'summary_data') as check, check():
    paths = find_tar_files(config['reader']['data_read_path'], pattern='')
    reader = WebDatasetReader(config=None, file_path=paths)
    samples = reader.get_dataset()

    summary = Summary(config=None)
    summary.summarize(samples)
