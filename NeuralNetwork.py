import numpy as np
from PIL import Image
import tensorflow as tf


def image2table(im: Image.Image, im_size: (int, int)):
    im_resized: Image.Image = im.resize(im_size)
    table_image = []
    for i in range(im_size[0]):
        table_image.append([])
        for j in range(im_size[1]):
            pixel = sum(im_resized.getpixel((i, j))) / (3 * 255.0)
            table_image[i].append(pixel)
        pass
    return table_image


def index_highest(table):
    highest_v = max(table[0])
    for i in range(len(table[0])):
        if highest_v == table[0][i]:
            return i


class CharDetector:
    def __init__(self, img_size: () = (20, 20), out_number: int = 2):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(input_dim=img_size[0] * img_size[1], output_dim=int((img_size[0] * img_size[1]))))
        self.model.add(tf.keras.layers.SimpleRNN(int((img_size[0] * img_size[1]))))
        self.model.add(tf.keras.layers.Dense(img_size[0] * img_size[1], activation='relu'))
        self.model.add(tf.keras.layers.Dense(out_number, activation='sigmoid'))

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        self._images: [] = []
        self._accuracyImages: [] = []

        self._labels: [] = []
        self._accuracyLabels: [] = []

        self.__img_size = img_size
        pass

    def __load_single_image(self, image_name: "", label: int, images, labels):
        loading_image: Image.Image = Image.open(image_name)
        img = []
        for line in image2table(loading_image, self.__img_size):
            img += line
        images.append(img)
        labels.append(label)

    def __load_images(self, image_label_pairs, images, labels):
        for pair in image_label_pairs:
            self.__load_single_image(pair[0], pair[1], images, labels)

    def load_train_images(self, image_label_pairs):
        self.__load_images(image_label_pairs, self._images, self._labels)

    def load_accuracy_images(self, image_label_pairs):
        self.__load_images(image_label_pairs, self._accuracyImages, self._accuracyLabels)

    def train(self):
        self.model.fit(self._images, self._labels, epochs=25)

    def check_accuracy(self):
        test_loss, test_acc = self.model.evaluate(self._accuracyImages, self._accuracyLabels, verbose=2)
        return test_acc

    def predict(self, img_name: ""):
        im = Image.open(img_name)
        img = []
        for line in image2table(im, self.__img_size):
            img += line
        predictions = self.model.predict([img])
        return predictions
