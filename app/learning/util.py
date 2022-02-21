from datetime import datetime
from os import mkdir
from os.path import exists
from pathlib import Path

from tensorflow.keras.models import Sequential


def save_model(model: Sequential) -> None:
	"""
	Provides a convenience function to store trained neural networks
	in a trained_models directory.

	:param model: Trained deep learning model.
	:return: None
	"""
	base_dir: Path = Path(__file__).parent.parent

	if not exists(f'{base_dir}/trained_models'):
		mkdir(f'{base_dir}/trained_models')

	date = datetime.today().strftime('%Y-%m-%d-%H:%M')

	model.save(f'{base_dir}/trained_models/model_{date}.h5')
