
# tutorial url: https://www.tensorflow.org/tutorials/keras/basic_classification

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# obtem os dados para o treinamento e testes
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# configura camadas da rede
# camada 1: formata os dados de entrada
# camada 2: cria 128 neuronios para o aprendizado
# camada 3: retorna o resultado com um array de 10 possibilidades
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

# configura o compilador do modelo
model.compile(optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics = ['accuracy'])

# executa o aprendizado
model.fit(train_images, train_labels, epochs = 5)

# testa modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)

print(predictions[0])

maxConfidence = np.argmax(predictions[0])

print('Type: ', class_names[maxConfidence])

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    label = "{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100 * np.max(predictions_array),
                                        class_names[true_label],
                                        color = color)
    plt.xlabel(label)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


#num_rows = 5
#num_cols = 3
#num_images = num_rows * num_cols

#plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))
#for i in range(num_images):
#    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#    plot_image(i, predictions, test_labels, test_images)
#    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#    plot_value_array(i, predictions, test_labels)

#plt.show()

img = test_images[153]

print(img.shape)

img = np.expand_dims(img, 0)

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)
plt.show()

print(np.argmax(predictions_single[0]))