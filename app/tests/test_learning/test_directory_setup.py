import unittest
from os.path import exists
from pathlib import Path
from shutil import rmtree

from app.learning.preprocessing import create_dataset_directories
from app.learning.preprocessing import create_training_directories


class TestDirectorySetup(unittest.TestCase):

    def setUp(self):
        self.parent_dir: str = f'{Path(__file__).parent}'
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def test_create_dataset_directories(self):
        """
        Checks, whether the create_raw_data_dir function creates the intended
        raw_data directory structure, including subdirectories for training
        classes.

        @return: None
        """
        create_dataset_directories(self.parent_dir, self.classes)

        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/0'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/1'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/2'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/3'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/4'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/5'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/6'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/7'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/8'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/raw_data/9'))

    def test_create_training_dirs(self):
        """
        Checks, whether the create_training_directories function creates indeed
        the intended directory structure.

        @return: None
        """
        result = create_training_directories(self.parent_dir, self.classes)

        self.assertEqual(result[0].split('/')[-2:], ['train', self.classes[0]])
        self.assertEqual(result[1].split('/')[-2:], ['val', self.classes[0]])
        self.assertEqual(result[2].split('/')[-2:], ['test', self.classes[0]])

        self.assertEqual(result[3].split('/')[-2:], ['train', self.classes[1]])
        self.assertEqual(result[4].split('/')[-2:], ['val', self.classes[1]])
        self.assertEqual(result[5].split('/')[-2:], ['test', self.classes[1]])

        self.assertTrue(exists(f'{self.parent_dir}/datasets/train/1'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/val/1'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/test/1'))

        self.assertTrue(exists(f'{self.parent_dir}/datasets/train/2'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/val/2'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/test/2'))

        self.assertTrue(exists(f'{self.parent_dir}/datasets/train/3'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/val/3'))
        self.assertTrue(exists(f'{self.parent_dir}/datasets/test/3'))

    def tearDown(self):
        if exists(f'{self.parent_dir}/datasets'):
            rmtree(f'{self.parent_dir}/datasets')


if __name__ == '__main__':
    unittest.main()
