import os
import sys
from contextlib import contextmanager


from pathlib import Path
import natsort


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """

    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>12}  {:>2}  {:>}'.format('Skipping the block', '|', f))
            raise SkipWith()
        else:
            p.print_run('{:>12}  {:>3}  {:>1}'.format('Running the block', '|', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


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


def get_nonexistant_shard_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    if not os.path.isfile(fname_path % 0):
        return fname_path
    start_index = 1
    while os.path.exists(fname_path % start_index):
        start_index += 1
    return start_index


def create_directory(write_path):
    if not os.path.exists(write_path):

        # Create a new directory because it does not exist
        os.makedirs(write_path)
        print("Created new data directory!")


def find_tar_files(read_path, pattern):
    files = [str(f) for f in Path(read_path).glob('*.tar') if f.match(pattern + '*')]
    return natsort.natsorted(files)
