import os


def create_directory(write_path):
    if not os.path.exists(write_path):

        # Create a new directory because it does not exist
        os.makedirs(write_path)
        print("Created new data directory!")
