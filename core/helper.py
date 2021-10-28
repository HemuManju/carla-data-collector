#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import collections.abc
import os
import shutil

import cv2
import numpy as np

from tensorboard import program


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
            "Both 'overwrite' and 'restore' cannot be True at the same time")

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
    tb.configure(
        argv=[None, "--logdir", logdir, "--host", host, "--port", port])
    url = tb.launch()  # noqa
