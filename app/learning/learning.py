import tensorflow as tf
from datetime import datetime
from app.learning.model import create_model
from app.learning.preprocessing import preprocess_data
from app.learning.util import save_model


def train_model(store_model: bool = True) -> None:

    """
    Trains a deep neural network.

    @param store_model: Indicates whether the trained model should be stored.
    @return: None
    """
    train_gen, val_gen, test_gen = preprocess_data(False)

    model = create_model()

    log_dir = f'logs/fit/{datetime.now().strftime("%Y-%m-%d_%H:%M")}'
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

    model.fit(train_gen,
              epochs=10,
              steps_per_epoch=200,
              validation_data=val_gen,
              verbose=1,
              callbacks=[tensorboard])

    if store_model:
        save_model(model=model)


if __name__ == '__main__':
    train_model(False)
