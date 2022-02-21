import unittest

from app.learning.learning import train_model


class Test_Preprocessing(unittest.TestCase):

    def test_load_train_data(self):

        train_model(store_model=True)


if __name__ == '__main__':
    unittest.main()
