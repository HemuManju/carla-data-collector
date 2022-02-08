import os
import json

from PIL import Image as im

import webdataset as wds

from utils import get_nonexistant_path


class WebDatasetWriter():
    def __init__(self, config) -> None:
        self.cfg = config
        self.sink = None

    def _is_jsonable(self, x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    def _get_serializable_data(self, data):
        keys_to_delete = []
        for key, value in data.items():
            if not self._is_jsonable(value):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del data[key]
        return data

    def create_tar_file(self, file_name, write_path):
        # Check if file already exists, increment if so
        if self.cfg['data_writer']['shard_write']:
            path_to_file = write_path + file_name + '_%06d.tar'
        else:
            path_to_file = write_path + file_name + '.tar'

        # Create a folder
        write_path = get_nonexistant_path(path_to_file)

        # Create a tar file
        if self.cfg['data_writer']['shard_write']:
            max_count = self.cfg['data_writer']['shard_maxcount']
            self.sink = wds.ShardWriter(write_path,
                                        maxcount=max_count,
                                        compress=True)
        else:
            self.sink = wds.TarWriter(write_path, compress=True)

    def sample(self, data, index):
        image_data = im.fromarray(data['rgb'])
        del data['rgb']  # No longer needed

        # Find only serializable data
        remaining_data = self._get_serializable_data(data)
        return {
            "__key__": "sample%06d" % index,
            'jpeg': image_data,
            'json': remaining_data
        }

    def write(self, data, index):
        if self.sink is None:
            raise FileNotFoundError(
                'Please call create_tar_file() method before calling the write method'
            )
        self.sink.write(self.sample(data, index))

    def close(self):
        self.sink.close()
