# Carla (0.9.11) dataset collector :car: :floppy_disk:

Script used for collecting data on CARLA version 0.9.11 and save it is as [webdataset](https://github.com/webdataset/webdataset)

Types of data captured include RGB, vehicle data (speed, throttle, etc.) and traffic data. Other types of data (semantic segmentation, lidar, ...) are also implemented. The scripts are modified from carla rllib integration repository (https://github.com/carla-simulator/rllib-integration)


## Getting started
### Prerequisites
* Python carla package and its dependencies

\*Until now I've only tested on Ubuntu 20.04. It might work on other OS, but that's not certain.

### Installation
1. Clone this repo
```
git clone https://github.com/HemuManju/carla-data-collector
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Carla installation:

Refer to https://carla.readthedocs.io/en/latest/getting_started/ and https://github.com/carla-simulator/carla/blob/master/Docs/download.md

4. Change the settings in [experiment_config.yaml](experiment_config.yaml)


5. Run the [collect.py](collect.py) file
```
python collect.py
```
7. To record different data types (e.g. semantic segmentation, lidar, ...), add the configuration in [experiment_config.yaml](experiment_config.yaml) and change ```sample``` function in the [data_writer.py](data_writer.py) file. Currently, the [data_writer.py](data_writer.py) saves only the RGB images and other sensor data.
8. To create a movie from collected data, run the [read.py](read.py) file after chaning the data read path in the [experiment_config.yaml](experiment_config.yaml) file.

```
python read.py
```

The datasets are stored in ```data``` directory

TODO:
- [ ] Implement an automated version of [data_writer.py](data_writer.py) where the different data image types are automatically saved
- [ ] Add other traffic data (e.g. pedistrian information, obstacles, ...)
- [ ] Add a noiser module
- [x] Add reset function if the planning fails.
- [x] Add an example code to read [webdataset](https://github.com/webdataset/webdataset)
- [x] Add a functionality to create a mp4 video file
- [x] Add replay functionality
- - - -

readme file modified from: https://github.com/AlanNaoto/carla-dataset-runner
