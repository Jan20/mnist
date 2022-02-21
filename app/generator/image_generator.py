from os import listdir
from typing import Tuple, Any
import matplotlib.pyplot as plt
from PIL import Image


def generate_training_images(dest_dir: str, dataset: Tuple[Any, Any]) -> None:
    """
    Takes splits taken from an investment universe, sort them
    by the best performing selection and returns the 5 best performing
    selection, together with a date as a Panda's Series.

    @param dataset:
    @param dest_dir: Directory, at which the training image should be stored.
    @return: Series, containing a date and 5 best-performing selection.
    """
    for index, image in enumerate(dataset[0]):
        path: str = f'{dest_dir}/datasets/raw_data/{dataset[1][index]}'
        file_name = f'{path}/{dataset[1][index]}_{len(listdir(path))}.png'
        img = Image.fromarray(image)
        img.save(file_name)


def generate_image_via_imshow(dataset: Tuple[Any, Any], dest_dir: str) -> None:
    """
    Takes splits taken from an investment universe, sort them
    by the best performing selection and returns the 5 best performing
    selection, together with a date as a Panda's Series.

    @param dataset:
    @param dest_dir: Directory, at which the training image should be stored.
    @return: Series, containing a date and 5 best-performing selection.
    """
    for i, image in enumerate(dataset[0]):
        path: str = f'{dest_dir}/datasets/raw_data/{dataset[1][i]}'
        plt.imshow(image, cmap=plt.cm.Pastel1)
        annotate_graph()
        file_name = f'{path}/{dataset[1][i]}_{len(listdir(path))}.png'
        plt.gcf().set_size_inches(0.5, 0.5)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)


def annotate_graph() -> None:
    """
    Takes a matplotlib graph, sets the ordinate to a fix range from -100 to
    150. A drop to -100 signals lost of 100 percent of a stock's value, while
    150 refers to an increase of a stock's value by 150 percent.

    @return: None.
    """
    plt.gca().set_axis_off()
    plt.margins(0, 0)
