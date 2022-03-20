#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import collections.abc
import os
import shutil
import glob
import sys

import cv2
import numpy as np

from tensorboard import program

try:
    sys.path.append(
        glob.glob(
            '../carla/dist/carla-*%d.%d-%s.egg'
            % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64',
            )
        )[0]
    )
except IndexError:
    pass

import carla
from datetime import datetime
import re
import socket


def get_ip(host):
    if host in ['localhost', '127.0.0.1']:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(('10.255.255.255', 1))
            host = sock.getsockname()[0]
        except RuntimeError:
            pass
        finally:
            sock.close()
    return host


def find_weather_presets():
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), x) for x in presets]


def inspect(client):

    world = client.get_world()
    time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    actors = world.get_actors()
    s = world.get_settings()

    weather = 'Custom'
    current_weather = world.get_weather()
    for preset, name in find_weather_presets():
        if current_weather == preset:
            weather = name

    if s.fixed_delta_seconds is None:
        frame_rate = 'variable'
    else:
        frame_rate = '%.2f ms (%d FPS)' % (
            1000.0 * s.fixed_delta_seconds,
            1.0 / s.fixed_delta_seconds,
        )

    config = {}

    config['version'] = client.get_server_version()
    config['map'] = world.get_map().name
    config['weather'] = weather
    config['time'] = time
    config['frame_rate'] = frame_rate
    config['rendering'] = 'disabled' if s.no_rendering_mode else 'enabled'
    config['sync mode'] = 'disabled' if not s.synchronous_mode else 'enabled'
    config['spectator'] = len(actors.filter('spectator'))
    config['static'] = len(actors.filter('static.*'))
    config['traffic'] = len(actors.filter('traffic.*'))
    config['vehicles'] = len(actors.filter('vehicle.*'))
    config['walkers'] = len(actors.filter('walker.*'))

    return config


def post_process_image(image, normalized=True, grayscale=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :param grayscale
    :return: image
    """
    if isinstance(image, list):
        image = image[0]
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.float32)


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


def find_latest_checkpoint(directory):
    """
    Finds the latest checkpoint, based on how RLLib creates and names them.
    """
    start = directory
    checkpoint_path = ""
    max_checkpoint_int = -1

    path = os.walk(start)
    for root, directories, files in path:
        for directory in directories:
            if directory.split('_')[0] in "checkpoint_":
                # Find the checkpoint with least number
                checkpoint_int = int(directory.split('_')[1])
                if checkpoint_int > max_checkpoint_int:
                    max_checkpoint_int = checkpoint_int
                    name = "/checkpoint-" + str(checkpoint_int)
                    checkpoint_path = root + '/' + directory + name

    if not checkpoint_path:
        raise FileNotFoundError(
            "Could not find any checkpoint, make sure that you have selected the correct folder path"
        )

    return checkpoint_path


def get_checkpoint(training_directory, name, restore=False, overwrite=False):
    if name is not None:
        training_directory = os.path.join(training_directory, name)

    if overwrite and restore:
        raise RuntimeError(
            "Both 'overwrite' and 'restore' cannot be True at the same time"
        )

    if overwrite:
        if os.path.isdir(training_directory):
            shutil.rmtree(training_directory)
            print("Removing all contents inside '" + training_directory + "'")
        return None

    if restore:
        return find_latest_checkpoint(training_directory)

    return None


def launch_tensorboard(logdir, host="localhost", port="6006"):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--host", host, "--port", port])
    url = tb.launch()  # noqa
