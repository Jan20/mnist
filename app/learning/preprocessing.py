from os import listdir
from pathlib import Path
from shutil import copyfile
from keras_preprocessing.image import ImageDataGenerator
from app.generator.sample_generator import fetch_training_classes
from app.generator.sample_generator import fetch_training_samples
from app.learning.directory_setup import create_dataset_directories
from app.learning.directory_setup import create_training_directories
from app.generator.image_generator import generate_training_images


def preprocess_data(generate_samples: bool = True) -> (ImageDataGenerator,
                                                       ImageDataGenerator,
                                                       ImageDataGenerator):
    """
    Loads Mnist's handwritten digits dataset and transforms it into training,
    validation and test sets.

    @return: Tuple, containing train, validation and test samples.
    """
    parent_dir: str = f'{Path(__file__).parent}'

    if not generate_samples:
        return create_generators(parent_dir, width=28, height=28)

    classes: [str] = fetch_training_classes()

    create_dataset_directories(parent_dir, classes)

    dest_dirs = create_training_directories(parent_dir, classes)

    generate_training_images(parent_dir, fetch_training_samples())

    copy_files(parent_dir, dest_dirs, classes, 5000)

    return create_generators(parent_dir, width=28, height=28)


def copy_files(parent_dir: str, dest_dirs: [str], classes: [str],
               limit: int = None) -> None:
    """
    Takes a source directory, containing training images of cats as well as
    dogs, a category, which can either be cat or dog and a list of destination
    dirs, containing directories to store training, validation and testing
    samples for cat and dog images, respectively.

    @param parent_dir: Source directory, from which samples are taken.
    @param classes: String, that either equal cat or dog.
    @param dest_dirs: List of training, validation and test dirs.
    @param limit:
    @return: None.
    """
    if limit is None:
        # TODO: Refactor this piece of code into a dedicated function.
        limit = 100000
        for class_name in classes:
            class_directory = f'{parent_dir}/datasets/raw_data/{class_name}'
            number_of_files = len(listdir(class_directory))
            if number_of_files < limit:
                limit = number_of_files

    # Splits the files per class to one half into training, a quarter in
    # validation and finally a quarter in test dir.
    i: int = 0

    val_limit = limit / 2
    test_limit = val_limit + limit / 4

    for class_name in classes:
        for file_index in range(limit):
            dest_dir = i
            if file_index >= val_limit:
                dest_dir = i + 1
            if file_index >= test_limit:
                dest_dir = i + 2

            path: str = f'{parent_dir}/datasets/raw_data'
            copyfile(f'{path}/{class_name}/{class_name}_{file_index}.png',
                     f'{dest_dirs[dest_dir]}/{class_name}_{file_index}.png')
        i = i + 3


def create_generators(parent_dir: str, width: int, height: int) -> \
                     (ImageDataGenerator, ImageDataGenerator,
                      ImageDataGenerator):
    """
    Selects

    @param parent_dir:
    @param height:
    @param width:
    @return:
    """
    train_dir: str = f'{parent_dir}/datasets/train'
    val_dir: str = f'{parent_dir}/datasets/val'
    test_dir: str = f'{parent_dir}/datasets/test'

    train_gen: ImageDataGenerator = ImageDataGenerator(rescale=1./255)
    val_gen: ImageDataGenerator = ImageDataGenerator(rescale=1./255)
    test_gen: ImageDataGenerator = ImageDataGenerator(rescale=1./255)

    train_gen = train_gen.flow_from_directory(directory=train_dir,
                                              target_size=(width, height),
                                              batch_size=32,
                                              class_mode='categorical')

    val_gen = val_gen.flow_from_directory(directory=val_dir,
                                          target_size=(width, height),
                                          batch_size=32,
                                          class_mode='categorical')

    test_gen = test_gen.flow_from_directory(directory=test_dir,
                                            target_size=(width, height),
                                            batch_size=32,
                                            class_mode='categorical')

    return train_gen, val_gen, test_gen
