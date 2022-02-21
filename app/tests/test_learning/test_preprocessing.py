import unittest
from os import listdir
from os.path import exists, isfile
from pathlib import Path
from shutil import rmtree

from keras.datasets import mnist

from app.learning.directory_setup import create_dataset_directories
from app.generator.image_generator import generate_training_images
from app.learning.preprocessing import copy_files
from app.learning.preprocessing import create_training_directories


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.parent_dir: str = f'{Path(__file__).parent}'
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        create_dataset_directories(self.parent_dir, self.classes)

        mnist_data, _ = mnist.load_data()

        self.training_data = (mnist_data[0][:300], mnist_data[1][:300])

        generate_training_images(dataset=self.training_data,
                                 dest_dir=self.parent_dir)

        self.dest_dirs: [str] = create_training_directories(self.parent_dir,
                                                            self.classes)

    def test_copy_files(self):
        """
        Checks, whether the copy files function actually copies all expected
        files from a source directory into training, validation and test
        directories.

        @return: None
        """
        copy_files(self.parent_dir, self.dest_dirs, self.classes, 20)

        dest_dir = f'{self.parent_dir}/datasets'

        self.assertEqual(len(listdir(f'{dest_dir}/train/0')), 10)
        self.assertEqual(len(listdir(f'{dest_dir}/train/1')), 10)
        self.assertEqual(len(listdir(f'{dest_dir}/val/0')), 5)
        self.assertEqual(len(listdir(f'{dest_dir}/val/1')), 5)
        self.assertEqual(len(listdir(f'{dest_dir}/test/0')), 5)
        self.assertEqual(len(listdir(f'{dest_dir}/test/1')), 5)

        self.assertTrue(isfile(f'{dest_dir}/train/0/0_0.png'))
        self.assertTrue(isfile(f'{dest_dir}/train/1/1_0.png'))
        self.assertTrue(isfile(f'{dest_dir}/val/0/0_10.png'))
        self.assertTrue(isfile(f'{dest_dir}/val/1/1_10.png'))
        self.assertTrue(isfile(f'{dest_dir}/test/0/0_15.png'))
        self.assertTrue(isfile(f'{dest_dir}/test/1/1_15.png'))

    def tearDown(self):
        if exists(f'{self.parent_dir}/datasets'):
            rmtree(f'{self.parent_dir}/datasets')


if __name__ == '__main__':
    unittest.main()
