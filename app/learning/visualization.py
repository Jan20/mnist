import keras.callbacks
import matplotlib.pyplot as plt
from keras import models
from keras.models import load_model
from keras.preprocessing import image
from numpy import expand_dims, ndarray, zeros, clip


def display_progress(history: keras.callbacks.History):
    """
    Visualizes the training and validation progress.

    @param history:
    @return: None
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def visualize_graph(image_path: str) -> ndarray:

    img = image.load_img(path=image_path, target_size=(500, 500))

    image_tensor = image.img_to_array(img)

    image_tensor = expand_dims(image_tensor, axis=0)

    image_tensor /= 255.

    print(image_tensor.shape)

 #   plt.imshow(image_tensor[0])
 #   plt.show()

    return image_tensor


def visualize_feature_map(image_path: str, model_path: str) -> None:

    model = load_model(model_path)

    image_tensor = visualize_graph(image_path=image_path)

    layer_outputs = [layer.output for layer in model.layers[:8]]

    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(image_tensor)

    first_layer_activation = activations[0]

    print(first_layer_activation.shape)

    plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
    plt.show()


def visualize_feature_maps(image_path: str, model_path: str) -> None:

    model = load_model(model_path)

    layer_names = []

    image_tensor = visualize_graph(image_path=image_path)

    layer_outputs = [layer.output for layer in model.layers[:8]]

    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(image_tensor)

    for layer in model.layers[:12]:

        layer_names.append(layer.name)

    images_per_row = 16

    test = zip(layer_names, activations)

    for layer_name, layer_activation in zip(layer_names, activations):

        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row

        display_grid = zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):

            for row in range(images_per_row):

                channel_image = layer_activation[0, :, :, col * images_per_row + row]

                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

                scale = 1. / size

        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))

        plt.title(layer_name)
        plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
