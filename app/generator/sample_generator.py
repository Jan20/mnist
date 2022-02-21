from typing import Tuple, Any

from tensorflow.keras.datasets import mnist


def fetch_training_classes() -> [str]:
    """
    Returns the classes in which training samples may fall in.

    @return: List of training classes.
    """
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def fetch_training_samples() -> Tuple[Any, Any]:
    """
    Loads Mnist's handwritten digits dataset and returns it's training samples
    as tuple, consisting of a matrix, containing the handwritten digits as well
    as the class, in which the samples falls into.

    @return: Tuple, containing handwritten digits and there respective class.
    """
    mnist_data, _ = mnist.load_data()

    return mnist_data[0], mnist_data[1]
