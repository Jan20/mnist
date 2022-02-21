import unittest
import pickle


class Test_Processor(unittest.TestCase):

    def test_test(self):

        objects = []
        with (open("./tests/history.pickle", "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        
        print(objects)

        self.assertEqual(True, True)


if __name__ == '__main__':

    unittest.main()
