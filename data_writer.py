import os
import json

from PIL import Image as im

import webdataset as wds


def get_nonexistant_path(fname_path):
    """
        Get the path to a filename which does not exist by incrementing path.

        Examples
        --------
        >>> get_nonexistant_path('/etc/issue')
        '/etc/issue-1'
        >>> get_nonexistant_path('whatever/1337bla.py')
        'whatever/1337bla.py'
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}_{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}_{}{}".format(filename, i, file_extension)
    return new_fname


class WebWriter():
    def __init__(self, config, file_name, write_path) -> None:
        # Check if file already exists, increment if so
        path_to_file = write_path + file_name + '.tar'
        write_path = get_nonexistant_path(path_to_file)

        # Create a tar file
        self.sink = wds.TarWriter(write_path, compress=True)

        # Save the configuration
        with open(write_path.split('.')[0] + '_configuration.json', 'w') as fp:
            json.dump(config, fp, indent=4)
        fp.close()

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
        self.sink.write(self.sample(data, index))

    def close(self):
        self.sink.close()
