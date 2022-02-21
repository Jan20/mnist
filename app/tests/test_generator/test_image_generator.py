import unittest
from os import listdir
from os.path import exists, isfile
from pathlib import Path
from shutil import rmtree
from keras.datasets import mnist

from app.generator.image_generator import generate_training_images
from app.learning.preprocessing import create_dataset_directories


class TestImageGenerator(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets the Image generator tests up.

        @return: None
        """
        self.parent_dir = f'{Path(__file__).parent}'
        classes: [str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        create_dataset_directories(self.parent_dir, classes)

        mnist_data, _ = mnist.load_data()

        self.training_data = (mnist_data[0][:100], mnist_data[1][:100])

    def test_generate_training_images(self) -> None:
        """
        Tests, whether the generate_training_images function creates the
        expected number of training images, stored in the correct
        directory.

        @return: None
        """
        generate_training_images(self.parent_dir, self.training_data)

        base_path: str = f'{self.parent_dir}/datasets/raw_data'

        self.assertTrue(isfile(f'{base_path}/0/0_0.png'))
        self.assertEqual(len(listdir(f'{base_path}/0')), 13)
        self.assertEqual(len(listdir(f'{base_path}/1')), 14)
        self.assertEqual(len(listdir(f'{base_path}/2')), 6)
        self.assertEqual(len(listdir(f'{base_path}/3')), 11)
        self.assertEqual(len(listdir(f'{base_path}/4')), 11)
        self.assertEqual(len(listdir(f'{base_path}/5')), 5)
        self.assertEqual(len(listdir(f'{base_path}/6')), 11)
        self.assertEqual(len(listdir(f'{base_path}/7')), 10)
        self.assertEqual(len(listdir(f'{base_path}/8')), 8)
        self.assertEqual(len(listdir(f'{base_path}/9')), 11)

    def tearDown(self):
        if exists(f'{self.parent_dir}/datasets'):
            rmtree(f'{self.parent_dir}/datasets')


if __name__ == '__main__':
    unittest.main()
