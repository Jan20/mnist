from os import mkdir
from os.path import exists


def create_dataset_directories(parent_dir: str, classes: [str]) -> [str]:
    """
    Checks, whether directories exist, in which training samples can be stored
    at. If not, new directories are getting generated.

    @param parent_dir: Directory, at which a new class dir will be created.
    @param classes: Name of a training class for which a dir gets created.
    @return: Path to the class directory.
    """
    class_directories: [str] = []
    datasets_dir = f'{parent_dir}/datasets'
    raw_data_dir = f'{parent_dir}/datasets/raw_data'

    # Creates a dataset dir if not already present.
    if not exists(datasets_dir):
        mkdir(datasets_dir)

    if not exists(raw_data_dir):
        mkdir(raw_data_dir)

    for class_name in classes:
        class_directories.append(f'{raw_data_dir}/{class_name}')

    for class_directory in class_directories:
        if not exists(class_directory):
            mkdir(class_directory)

    return class_directories


def create_training_directories(parent_dir: str, classes: [str]) -> [str]:
    """
    Takes a target directory, intended host datasets for training, validation
    and test data. Each of the three datasets will contain a dedicated
    directory for storing images of dogs and cats, respectively.

    @param parent_dir: Directory in which the datasets directory are be placed.
    @param classes:
    @return: List of strings, containing the generated directories.
    """
    datasets_dir = f'{parent_dir}/datasets'
    destination_directories = []

    if not exists(datasets_dir):
        mkdir(datasets_dir)

    dirs = [f'{datasets_dir}/train',
            f'{datasets_dir}/val',
            f'{datasets_dir}/test']

    # Adds the paths of the envisioned child directories of the top-level
    # training, validation and testing directories to the list of
    # destination directories.
    for class_name in classes:
        for directory in dirs:
            destination_directories.append(f'{directory}/{class_name}')

    # Creates training, validation and test directories.
    for directory in dirs:
        if not exists(directory):
            mkdir(directory)

    # Creates all sub-directories of the top-level training, validation and
    # test directories.
    for directory in destination_directories:
        if not exists(directory):
            mkdir(directory)

    return destination_directories

